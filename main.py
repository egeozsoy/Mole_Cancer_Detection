import os

import numpy as np
from fastai.vision import ImageDataBunch, models, cnn_learner, accuracy
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from mole_dataset import MoleDataset
from utils import load_images_labels
from configurations import find_best_lr, plot_images, train

if __name__ == '__main__':
    image_files, labels = load_images_labels()

    training_image_paths, testing_image_paths, training_labels, testing_labels = train_test_split(image_files, labels, random_state=42, test_size=0.1)

    # TODO use kaggle preprocessing https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping

    img_size = 224

    transform_train = transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomResizedCrop(size=(img_size, img_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         # transforms.Normalize((0.7, 0.54, 0.50), (0.17, 0.17, 0.19))
         ])

    transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(),
                                         # transforms.Normalize((0.7, 0.54, 0.50), (0.17, 0.17, 0.19))
                                         ])
    train_dataset = MoleDataset(training_image_paths, training_labels, transform=transform_train)
    test_dataset = MoleDataset(testing_image_paths, testing_labels, transform=transform_test)

    if not os.path.exists('models/pytorch_model.pt') or train:
        print('Training Model')

        # ---------------------Use FastAi for easier training with less boilerplate code---------------------
        data = ImageDataBunch.create(train_dataset, test_dataset, bs=64)
        learner = cnn_learner(data, models.resnet18, metrics=accuracy)
        # Either train the cnn layers or not
        learner.unfreeze()

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

        learner.fit_one_cycle(1, 3e-5)  # other rates to try, 5e-07, 5e-06
        # Save it as a pytorch model, not as fastai model
        torch.save(learner.model, 'models/pytorch_model.pt')

        # ---------------------Stop using FastAi, return to native pytorch---------------------

    model = torch.load('models/pytorch_model.pt')
    print('Do stuff with model')
    print(model)

# TODO apply transformations
