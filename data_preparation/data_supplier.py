import typing

import torch
from sklearn.model_selection import train_test_split

from data_preparation.dataloader import create_dataloader


def _split_into_train_and_validation(data: typing.List[str],
                                     rate: float = 0.2,
                                     seed: int = 0
                                     ) -> typing.Tuple[typing.List[str], typing.List[str], typing.List[str], typing.List[str]]:
    """ Split provided data into train and validation.

    Arguments
    ----------
    data : typing.List[typing.Tuple[str, str]]
        Data to split
    rate : float
        Validation data percentage, default=20%
    seed : int
        seed value

    Returns
    -------
    typing.Tuple[typing.List[str], typing.List[str], typing.List[str], typing.List[str]]
        train samples, validation samples, train labels, validation labels
    """
    x = [img for (img, gesture) in data]
    y = [gesture for (img, gesture) in data]
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=rate, random_state=seed)
    return x_train, x_validation, y_train, y_validation


def prepare_data(data: typing.List[str],
                 batch_size: int,
                 means: typing.List[float],
                 stds: typing.List[float]
                 ) -> typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """ Prepare data for training.
    Data (path to image - label) will be split into train and validation
    and then corresponding data loaders returning image-label will be created.

    Arguments
    ----------
    data : typing.List[typing.Tuple[str, str]]
        Train and validation data
    batch_size : int
        Size of batch
    means : typing.Tuple
        Mean values for data normalization
    stds : typing.Tuple
        Std values for data normalization

    Returns
    -------
    typing.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        train and validation data loaders
    """
    x_train, x_validation, y_train, y_validation = _split_into_train_and_validation(data)
    train_dataloader = create_dataloader(x_train, y_train, batch_size, True, means, stds)
    validation_dataloader = create_dataloader(x_validation, y_validation, batch_size, False, means, stds)
    return train_dataloader, validation_dataloader
