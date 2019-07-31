import os

import numpy as np
from sklearn.utils import shuffle
from PIL import ImageFilter
# Paths define two constant, malignant_path and benign_path with absolute paths
from paths import malignant_path, benign_path


def load_images_labels():
    benign_files = []
    malignant_files = []
    modified_malignant_files = []
    print(benign_path)
    print(malignant_path)
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
    # Create corresponding labels
    labels = [0] * len(benign_files) + [1] * len(modified_malignant_files)

    image_files, labels = shuffle(image_files, labels)

    return image_files, labels


# Currently used
class BensProcessing(object):
    '''
    # https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
    '''

    def __call__(self, img):
        blur_filter = ImageFilter.GaussianBlur(radius=10)
        blurred = np.array(img.filter(blur_filter)).astype(np.float32)
        img = np.array(img).astype(np.float32)

        final_img = ((img * 4 + blurred * (-4)) + 128) / 255

        return final_img
