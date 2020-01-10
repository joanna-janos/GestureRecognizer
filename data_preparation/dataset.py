import typing

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class GestureDataset(Dataset):
    """ Dataset for gestures where one sample is image-label.

    Parameters
    ----------
    paths : typing.List[str]
        paths to images
    gestures : typing.List[str]
        labels (gestures names)
    transform : torchvision.transforms.Compose
         Transformations to carry out on image, default: changing to tensor

    Returns
    -------
    torch.Tensor, torch.Tensor
        tensors representing gestures' image and label
    """

    def __init__(self,
                 paths: typing.List[str],
                 gestures: typing.List[str],
                 transform=ToTensor()):
        self.paths = paths
        self.gestures = gestures
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx])
        return self.transform(img), self.gestures[idx]
