import os

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from configurations import img_size, cache_location,cache_segmentation_location
from paths import segmentation_path


class MoleDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.c = 2

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img_name:str = img_path.rsplit('/', 1)[-1]
        label_str = 'benign' if label == 0 else 'malignant'
        cache_path = os.path.join(cache_location, os.path.join(label_str,img_name))

        if os.path.exists(cache_path):
            image = Image.open(cache_path)
        else:
            image = Image.open(img_path)
            image = image.resize((img_size, img_size))
            image.save(cache_path)

        img_name_without_extension = img_name.replace('.jpeg','')
        novice_segmentation_path = os.path.join(segmentation_path,'{}_novice.png'.format(img_name_without_extension))
        expert_segmentation_path = os.path.join(segmentation_path, '{}_expert.png'.format(img_name_without_extension))
        segmentation_image_path = novice_segmentation_path if os.path.exists(novice_segmentation_path) else expert_segmentation_path
        cache_segmentation_path = os.path.join(cache_segmentation_location,'{}_mask.npy'.format(img_name_without_extension))
        rectangle_segmantation_matrix = None

        if os.path.exists(cache_segmentation_path):
            rectangle_segmantation_matrix = np.load(cache_segmentation_path)

        elif os.path.exists(segmentation_image_path):
            segmentation_map = Image.open(segmentation_image_path)
            segmentation_map = segmentation_map.resize((img_size, img_size))
            segmentation_matrix = np.array(segmentation_map).astype(np.bool)

            non_zero_columns = np.nonzero(np.any(segmentation_matrix, axis=0))[0]
            non_zero_rows = np.nonzero(np.any(segmentation_matrix, axis=1))[0]

            threshold = 16
            left_range = non_zero_columns[0] - threshold
            right_range = non_zero_columns[-1] + threshold
            upper_range = non_zero_rows[0] - threshold
            bottom_range = non_zero_rows[-1] + threshold

            if left_range < 0:
                left_range = 0
            if right_range >= img_size:
                right_range = img_size - 1
            if upper_range < 0:
                upper_range = 0
            if bottom_range >= img_size:
                bottom_range = img_size - 1

            rectangle_segmantation_matrix = np.zeros((img_size, img_size, 3), dtype=np.bool)
            rectangle_segmantation_matrix[upper_range:bottom_range, left_range:right_range, :] = True
            np.save(cache_segmentation_path, rectangle_segmantation_matrix)

        if rectangle_segmantation_matrix is not None:
            image_matrix = np.array(image)
            image_matrix[~rectangle_segmantation_matrix] = np.random.rand(len(image_matrix[~rectangle_segmantation_matrix])) * 255
            # Convert back to PIL
            image = Image.fromarray(image_matrix)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.image_paths)
