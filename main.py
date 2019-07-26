import os

import numpy as np

from fastai.vision import ImageDataBunch, models, cnn_learner, accuracy
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage import io
from matplotlib import pyplot as plt


class MoleDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.c = 2

    def __getitem__(self, index: int):
        img_name = self.image_paths[index]
        image = io.imread(img_name)
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.image_paths)


# --------- Start data processing -----------
benign_path = '/Volumes/seagate 1/ML_Datasets/Mole_Cancer_ISIC/Data/benign'
malignant_path = '/Volumes/seagate 1/ML_Datasets/Mole_Cancer_ISIC/Data/malignant'
benign_files = []
malignant_files = []
modified_malignant_files = []

for benign_file in os.listdir(benign_path):
    if '.jpeg' in benign_file:
        benign_files.append(os.path.join(benign_path, benign_file))

for malignant_file in os.listdir(malignant_path):
    if '.jpeg' in malignant_file:
        malignant_files.append(os.path.join(malignant_path, malignant_file))

# # we have two few malignant files
while len(modified_malignant_files) < len(benign_files):
    modified_malignant_files += malignant_files

# make sure they are the same length
modified_malignant_files = modified_malignant_files[:len(benign_files)]

image_files = benign_files + modified_malignant_files
labels = [0] * len(benign_files) + [1] * len(modified_malignant_files)

image_files, labels = shuffle(image_files, labels)

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
