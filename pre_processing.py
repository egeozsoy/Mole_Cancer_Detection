# coding: utf-8
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

IMAGE_SIZE = 64

def get_images(images_path='create_mole_data/images/'):
    images_data = []

    for image_data in os.listdir(images_path):
        if 'benign' in image_data or 'malignant' in image_data:
            images_data.append(image_data)

    return images_data


def one_hot_encoding(image_data):
    if 'benign' in image_data:
        return [1, 0]
    else:
        return [0, 1]


def create_training_data(images_path='create_mole_data/images', limit=None, size=80):
    images = []
    labels = []
    images_data = get_images(images_path)

    for idx, image_data in enumerate(images_data):
        img_path = os.path.join(images_path,image_data)
        img = cv2.imread(img_path, 0)  # 0 for grayscale
        img = cv2.resize(img, (size, size))
        images.append(img)
        labels.append(one_hot_encoding(image_data))
        if limit and idx > limit:
            break
        else:
            if idx % 500 == 0:
                print(idx)

    return images, labels


def create_prediction_image(image_file_path, img_size=80):
    try:
        img = cv2.imread(image_file_path, 0)
        img = cv2.resize(img, (img_size, img_size))
        return img
    except Exception as e:
        print(e)
        return None


def visualize_images(images, labels):
    # Image.open(img_path)
    for idx, image in enumerate(images):
        fig = plt.figure()
        plt.imshow(image, cmap='gray')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        fig.text(.5, .05, labels[idx], ha='center')
        plt.show()


if __name__ == '__main__':
    images, labels = create_training_data(size=IMAGE_SIZE,images_path='/Users/egeozsoy/Documents/mole_cancer_images')
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html
    np.save('images.npy', images)
    np.save('labels.npy', labels)
    # np.load('images.npy') to load
