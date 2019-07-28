import os

from torch.utils.data import Dataset
from PIL import Image

from configurations import img_size, cache_location


class MoleDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.c = 2

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        label = self.labels[index]

        img_name = img_path.rsplit('/', 1)[-1]
        label_str = 'benign' if label == 0 else 'malignant'
        cache_path = os.path.join(cache_location, os.path.join(label_str,img_name))

        if os.path.exists(cache_path):
            image = Image.open(cache_path)
        else:
            image = Image.open(img_path)
            image = image.resize((img_size, img_size))
            image.save(cache_path)


        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.image_paths)
