import os

import torch
from torchvision import transforms
import fastai
import fastprogress
from fastai.vision import ImageDataBunch, models, cnn_learner, accuracy, Learner
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from PIL import Image

from mole_dataset import MoleDataset
from utils import load_images_labels, BensProcessing
from configurations import find_best_lr, plot_images, train, unfreeze_cnn_layers, img_size, cache_location

if __name__ == '__main__':

    # Disable progress bar if wanted
    fastprogress.fastprogress.NO_BAR = True
    master_bar, progress_bar = fastprogress.force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar

    if not os.path.exists(cache_location):
        os.mkdir(cache_location)
        os.mkdir(os.path.join(cache_location, 'malignant'))
        os.mkdir(os.path.join(cache_location, 'benign'))

    image_files, labels = load_images_labels()

    training_image_paths, testing_image_paths, training_labels, testing_labels = train_test_split(image_files, labels, random_state=42, test_size=0.1)

    # TODO use kaggle preprocessing https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping

    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(size=(img_size, img_size)), transforms.RandomHorizontalFlip(), BensProcessing(), transforms.ToTensor(),
         # transforms.Normalize((0.7, 0.54, 0.50), (0.17, 0.17, 0.19))
         ])

    transform_test = transforms.Compose([transforms.Resize(size=(img_size, img_size)), BensProcessing(), transforms.ToTensor(),
                                         # transforms.Normalize((0.7, 0.54, 0.50), (0.17, 0.17, 0.19))
                                         ])
    train_dataset = MoleDataset(training_image_paths, training_labels, transform=transform_train)
    test_dataset = MoleDataset(testing_image_paths, testing_labels, transform=transform_test)

    if not os.path.exists('models/pytorch_model.pt') or train:
        print('Training Model')

        # ---------------------Use FastAi for easier training with less boilerplate code---------------------
        data = ImageDataBunch.create(train_dataset, test_dataset)

        # Pytorch bug avoids us using mobilenet_v2
        # model  = torchvision.models.mobilenet_v2(num_classes=2)
        # learner = Learner(data=data,model=model)

        # resnet 34 achieved 91 after 5 epochs(still improving)
        learner = cnn_learner(data, models.resnet50, metrics=accuracy)
        # Either train the cnn layers or not
        if unfreeze_cnn_layers:
            print('Unfreezing Model')
            learner.unfreeze()

        total_params = sum(p.numel() for p in learner.model.parameters())
        trainable_params = sum(p.numel() for p in learner.model.parameters() if p.requires_grad)
        print(f'Total Paramaters: {total_params}, Trainable Parameters: {trainable_params}')

        # We can plot some images to take a look at them
        if plot_images:
            images = next(iter(data.train_dl))

            for image in images[0]:
                img = image.transpose(0, 1).transpose(1, 2).numpy()
                print(img.max(), img.min())
                plt.imshow(img)
                plt.show()

        # We can find the best learning rate
        if find_best_lr:
            learner.lr_find()
            fig = learner.recorder.plot(return_fig=True)
            plt.show()

        learner.fit_one_cycle(10, 3e-5)  # other rates to try, 5e-07, 5e-06
        # Save it as a pytorch model, not as fastai model
        torch.save(learner.model, 'models/pytorch_model.pt')

        # ---------------------Stop using FastAi, return to native pytorch---------------------

    model = torch.load('models/pytorch_model.pt', map_location='cpu')
    new_image_path = '224_cached_images/malignant/ISIC_0000046.jpeg'
    new_image = Image.open(new_image_path)
    scores = model(transform_test(new_image).unsqueeze(0))
    prediction = int(torch.argmax(scores))
    prediction_str = 'benign' if prediction == 0 else 'malignant'
    print(prediction_str.capitalize())
    print(scores.detach().numpy())
