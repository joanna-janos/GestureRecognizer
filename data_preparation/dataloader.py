import typing

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, ColorJitter, RandomVerticalFlip, Resize, Lambda, RandomApply, \
    Normalize

from data_preparation.dataset import GestureDataset


def _get_img_transformations(means: typing.List[float],
                             stds: typing.List[float]
                             ) -> Compose:
    """ Get all transformations to carry out on image.
    Helpful while creating GestureDataset.

    Arguments
    ----------
    means : typing.Tuple
        Mean values for data normalization
    stds : typing.Tuple
        Std values for data normalization

    Returns
    -------
    torchvision.transforms.Compose
        composition of transformations
    """
    return Compose([
        Resize((512, 256)),
        ColorJitter(brightness=0.1, contrast=0.1),  # will be applied RANDOMLY
        RandomVerticalFlip(p=0.1),
        ToTensor(),
        Normalize(means, stds),
        RandomApply([Lambda(lambda x: x + torch.randn_like(x))], p=0.3)  # noise
    ])


def create_dataloader(x: typing.List[str],
                      y: typing.List[str],
                      batch_size: int,
                      shuffle: bool,
                      means: typing.List[float],
                      stds: typing.List[float]
                      ) -> torch.utils.data.DataLoader:
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
    means : typing.Tuple
        Mean values for data normalization
    stds : typing.Tuple
        Std values for data normalization

    Returns
    -------
    torch.utils.data.DataLoader
        data loader
    """
    transforms = _get_img_transformations(means, stds)
    dataset = GestureDataset(x, y, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
