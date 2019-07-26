import os

import numpy as np
from fastai.vision import ImageDataBunch, models, cnn_learner, accuracy
from torchvision import transforms
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from mole_dataset import MoleDataset

# --------- Start data processing -----------


training_image_paths, testing_image_paths, training_labels, testing_labels = train_test_split(image_files, labels, random_state=42, test_size=0.1)
# --------- End data processing ------------

# TODO use kaggle preprocessing

img_size = 224
transform_train = transforms.Compose(
    [transforms.ToPILImage(), transforms.RandomResizedCrop(size=(img_size, img_size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
     transforms.Normalize((0.7, 0.54, 0.50), (0.17, 0.17, 0.19))
     ])

transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(),
                                     transforms.Normalize((0.7, 0.54, 0.50), (0.17, 0.17, 0.19))
                                     ])
train_dataset = MoleDataset(training_image_paths, training_labels, transform=transform_train)
test_dataset = MoleDataset(testing_image_paths, testing_labels, transform=transform_test)

data = ImageDataBunch.create(train_dataset, test_dataset)
learner = cnn_learner(data, models.resnet18, metrics=accuracy)
learner.unfreeze()
# # Find best learning rate
learner.lr_find()
fig = learner.recorder.plot(return_fig=True)
plt.show()

# learner.fit_one_cycle(10, 1e-3)


# images = next(iter(train_loader))
# images_to_analyse = images[0].numpy()
# first_channel = images_to_analyse[:, 0, :, :]
# second_channel = images_to_analyse[:, 1, :, :]
# third_channel = images_to_analyse[:, 2, :, :]
#
# print(first_channel.mean())
# print(second_channel.mean())
# print(third_channel.mean())
# print(first_channel.std())
# print(second_channel.std())
# print(third_channel.std())

# (img[0].numpy()*255)[:,0,:,:].std()
# plt.imshow(img[0][0].transpose(0,1).transpose(1,2).numpy())
