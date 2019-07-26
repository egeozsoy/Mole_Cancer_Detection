from torch.utils.data import Dataset
from skimage import io


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
