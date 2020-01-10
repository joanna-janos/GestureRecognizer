import typing

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, Resize

from data_preparation.dataset import GestureDataset


def _get_img_transformations():
    """ Get all transformations to carry out on image.
    Helpful while creating GestureDataset.

    Returns
    -------
    torchvision.transforms.Compose
        composition of transformations
    """
    return Compose([
        Resize((512, 256)),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # will be applied RANDOMLY
        RandomHorizontalFlip(p=0.3),
        RandomVerticalFlip(p=0.3),
        ToTensor()
    ])


def create_dataloader(x: typing.List[str],
                      y: typing.List[str],
                      batch_size: int,
                      shuffle: bool = True):
    """ Create data loader for given data and labels that returns given number of samples as one batch.

    Arguments
    ----------
    x : typing.List[str]
        paths to images
    y : typing.List[str]
        labels
    batch_size : int
        number of samples in one batch
    shuffle : bool
        True if data should be returned in random order, False otherwise

    Returns
    -------
    torch.utils.data.DataLoader
        data loader
    """
    transforms = _get_img_transformations()
    dataset = GestureDataset(x, y, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
