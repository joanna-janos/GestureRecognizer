import typing

import torch


def get_loss_and_optimizer(model: torch.nn.Module,
                           learning_rate: float
                           ) -> typing.Tuple[typing.Callable, torch.optim.Optimizer]:
    """ Get loss and optimizer used for lr finder and training.
    Util created to have the same criterion and optimizer during both tasks.

    Arguments
    ----------
    model : torch.nn.Module
        This model parameters will be updated by optimizer
    learning_rate : float
        Learning rate used by optimizer

    Returns
    -------
    typing.Tuple[typing.Callable, torch.optim.Optimizer]
        Loss and optimizer
    """
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return loss, optimizer
