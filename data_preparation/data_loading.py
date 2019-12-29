import typing

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, ColorJitter


class GestureDataset(Dataset):
    """ Dataset for gestures where one sample is image-label.

    Parameters
    ----------
    paths : List[str]
        paths to images with gestures
    gestures : List[str]
        labels (gestures names)
    transform : torchvision.transforms.Compose
         Transformations to carry out on image

    Returns
    -------
    torch.Tensor, torch.Tensor
        tensor representing gestures image and label
    """

    def __init__(self,
                 paths: typing.List[str],
                 gestures: typing.List[str],
                 transform=None):
        self.paths = paths
        self.gestures = gestures
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.gestures[idx])


def _get_img_transformations():
    """ Get all transformations to carry out on image.
    Helpful while creating GestureDataset.

    Returns
    -------
    torchvision.transforms.Compose
        composition of transformations
    """
    return Compose([
        ToTensor(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)  # will be applied RANDOMLY
        # TODO: consider other transformations like horizontal/vertical flip when gestures will be chosen
    ])


def _create_dataloader(x: typing.List,
                       y: typing.List,
                       batch_size: int = 64,
                       shuffle: bool = True):
    """ Create data loader for given data and labels that returns given number of samples as one batch.

    Arguments
    ----------
    x : List[str]
        data
    y : float
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


def _split_into_train_and_validation(data: typing.List,
                                     rate: float = 0.2,
                                     seed: int = 1024):
    """ Split provided data into train and validation.

    Arguments
    ----------
    data : list[(str, str)]
        Data to split
    rate : float
        Validation data percentage, default=20%
    seed : int
        seed value

    Returns
    -------
    list[str], list[str], list[str], list[str]
        train samples, validation samples, train labels, validation labels
    """
    x = [img for (img, gesture) in data]
    y = [gesture for (img, gesture) in data]
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=rate, random_state=seed)
    return x_train, x_validation, y_train, y_validation


def prepare_data(data: typing.List):
    """ Prepare data for training.
    Data (path to image - label) will be split into train and validation
    and then corresponding data loaders returning image-label will be created.

    Arguments
    ----------
    data : list[(str, str)]
        Train and validation data

    Returns
    -------
    torch.utils.data.DataLoader, torch.utils.data.DataLoader
        train and validation data loaders
    """
    x_train, x_validation, y_train, y_validation = _split_into_train_and_validation(data)
    train_dataloader = _create_dataloader(x_train, y_train)
    validation_dataloader = _create_dataloader(x_validation, y_validation, False)
    return train_dataloader, validation_dataloader
