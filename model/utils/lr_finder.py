import torch
from torch_lr_finder import LRFinder

from model.model_selector import get_model
from model.utils.loss_optimizer import get_loss_and_optimizer


def find_optimal_learning_rate(model_name: str,
                               pretrained: bool,
                               train_only_last_layer: bool,
                               train_data_loader: torch.utils.data.DataLoader,
                               validation_data_loader: torch.utils.data.DataLoader,
                               path_to_visualisations: str,
                               min_learning_rate: float,
                               max_learning_rate: float):
    """ Find learning rate based on Leslie Smith's approach
    and https://github.com/davidtvs/pytorch-lr-finder implementation.

    Arguments
    ----------
    model_name : str
        Model to train
    pretrained : bool
        True if model should be pretrained, False otherwise
    train_only_last_layer : bool
        Value indicating part of model that were trained (filename will contain information about it)
    train_data_loader: torch.utils.data.DataLoader
        Data loader used for training
    validation_data_loader: torch.utils.data.DataLoader
        Data loader used for validation
    path_to_visualisations : str
        Path where results of lr finder will be stored
    min_learning_rate : float
        Minimum learning rate used for searching
    max_learning_rate : float
        Maximum learning rate used for searching
    """
    model = get_model(model_name, train_only_last_layer, pretrained)
    criterion, optimizer = get_loss_and_optimizer(model, min_learning_rate)

    lr_finder = LRFinder(model, optimizer, criterion)
    lr_finder.range_test(
        train_loader=train_data_loader,
        val_loader=validation_data_loader,
        end_lr=max_learning_rate,
        num_iter=5,
        step_mode="exp")
    lr_finder.plot()
    # TODO: Save plot under `path_to_visualisations` directory
    lr_finder.reset()