import torch
from torch.utils.tensorboard import SummaryWriter

from model.utils.directory import create_not_existing_directory


def setup_tensorboard(path_tensorboard: str,
                      model_name: str,
                      pretrained: bool
                      ) -> torch.utils.tensorboard.SummaryWriter:
    """ Setup tensorboard under given directory for provided model.
    Directory for storing logs:
    PATH_TENSORBOARD/MODEL_NAME/pretrained
    or PATH_TENSORBOARD/MODEL_NAME/not_pretrained

    Arguments
    ----------
    path_tensorboard : str
         Main directory for tensorboard results
    model_name : str
        Model name
    pretrained : bool
        True if model is pretrained, False otherwise

    Returns
    -------
    torch.utils.tensorboard.SummaryWriter
        Entry to log data for consumption and visualization by TensorBoard
    """
    if pretrained:
        path_to_results_wth_filename = path_tensorboard + model_name + '_pretrained'
    else:
        path_to_results_wth_filename = path_tensorboard + model_name + '_not_pretrained'

    create_not_existing_directory(path_tensorboard)
    return SummaryWriter(path_to_results_wth_filename)


def log_results(writer: torch.utils.tensorboard.SummaryWriter,
                accuracy_train: float,
                loss_train: float,
                accuracy_validation: float,
                loss_validation: float,
                epoch: int
                ):
    """ Log accuracies and losses for training and validation in given epoch.

    Arguments
    ----------
    writer : torch.utils.tensorboard.SummaryWriter
         Entry to log data for consumption and visualization by TensorBoard
    accuracy_train : float
        Training accuracy
    loss_train : float
        Training loss
    accuracy_validation : float
        Validation accuracy
    loss_validation : float
        Validation loss
    epoch : int
        Number of epochs, where above results were obtained
    """
    writer.add_scalar('Train/Accuracy', accuracy_train, epoch)
    writer.add_scalar('Train/Loss', loss_train, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy_validation, epoch)
    writer.add_scalar('Validation/Loss', loss_validation, epoch)
