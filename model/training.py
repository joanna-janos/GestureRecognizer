import typing

import torch
from tqdm import tqdm

from model.model_selector import get_model
from model.utils.directory import create_not_existing_directory
from model.utils.loss_optimizer import get_loss_and_optimizer
from model.utils.tensorboard import setup_tensorboard, log_results


def train(model: torch.nn.Module,
          data_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler,
          loss: typing.Callable,
          batch_size: int
          ):
    """ Train model on batched data using provided optimizer and loss

    Arguments
    ----------
    model : torch.nn.Module
        Model to train
    data_loader: torch.utils.data.DataLoader
        Data loader
    optimizer : torch.optim.Optimizer
        Optimizer used to update weights
    scheduler :
        Optimizer's scheduler
    loss : typing.Callable
        Criterion to calculate loss
    batch_size : int
        Size of batch
    """
    print(f'\tTraining')
    model.train()
    overall_loss = 0
    correct = 0
    for x, y in tqdm(data_loader):
        optimizer.zero_grad()
        output = model(x)
        correct += float(sum(output.argmax(dim=1) == y))
        loss_value = loss(output, y)
        overall_loss += loss_value
        loss_value.backward()
        optimizer.step()
        scheduler.step()

    accuracy_per_epoch = correct / (len(data_loader) * batch_size)
    loss_per_epoch = overall_loss / (len(data_loader) * batch_size)
    print(f'\tAccuracy: {accuracy_per_epoch}')
    print(f'\tLoss: {loss_per_epoch}')
    return accuracy_per_epoch, loss_per_epoch


def validate(model: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             loss: typing.Callable,
             batch_size: int
             ):
    """ Validate model on batched data

    Arguments
    ----------
    model : torch.nn.Module
        Model to validate
    data_loader: torch.utils.data.DataLoader
        Data loader
    loss : typing.Callable
        Criterion to calculate loss
    batch_size : int
        Size of batch
    """
    print(f'\tValidating')
    model.eval()
    overall_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in tqdm(data_loader):
            output = model(x)
            correct += float(sum(output.argmax(dim=1) == y))
            loss_value = loss(output, y)
            overall_loss += loss_value
    accuracy_per_epoch = correct / (len(data_loader) * batch_size)
    loss_per_epoch = overall_loss / (len(data_loader) * batch_size)
    print(f'\tAccuracy: {accuracy_per_epoch}')
    print(f'\tLoss: {loss_per_epoch}')
    return accuracy_per_epoch, loss_per_epoch


def save_model(path_to_saved_model: str,
               model: torch.nn.Module,
               model_name: str,
               pretrained: bool,
               accuracy: float
               ):
    """ Save trained model for future usage

    Arguments
    ----------
    path_to_saved_model: str
        Path where model will be stored
    model : torch.nn.Module
        Model to save
    model_name : str
        Model name (filename will contain it)
    pretrained : bool
        True if model is pretrained, False otherwise
    accuracy : float
        Validation accuracy
    """

    if pretrained:
        full_path_to_saved_model = path_to_saved_model + model_name + '/pretrained/'
    else:
        full_path_to_saved_model = path_to_saved_model + model_name + '/not_pretrained/'

    create_not_existing_directory(full_path_to_saved_model)
    torch.save(model.state_dict(), full_path_to_saved_model + str(accuracy) + '.pt')


def train_and_validate(model_name: str,
                       pretrained: bool,
                       train_data_loader: torch.utils.data.DataLoader,
                       validation_data_loader: torch.utils.data.DataLoader,
                       epochs: int,
                       path_to_saved_model: str,
                       batch_size: int,
                       train_only_last_layer: bool,
                       base_learning_rate: float,
                       max_learning_rate: float,
                       path_tensorboard: str
                       ):
    """ Train and validate model and then save under `path_to_saved_model` directory

    Arguments
    ----------
    model_name : str
        Model name
    pretrained : bool
        True if model should be pretrained, False otherwise
    train_data_loader : torch.utils.data.DataLoader
        Data loader used for training
    validation_data_loader : torch.utils.data.DataLoader
        Data loader used for validation
    epochs : int
        Number of epochs
    path_to_saved_model: str
        Path where model will be stored
    batch_size : int
        Size of batch
    train_only_last_layer : bool
        Value indicating part of model that were trained
    base_learning_rate : float
        Learning rate
    max_learning_rate : float
        Learning rate
    path_tensorboard : str
        Path where tensorboard results will be stored
    """
    tensorboard_writer = setup_tensorboard(path_tensorboard, model_name, pretrained)
    model = get_model(model_name, train_only_last_layer, pretrained)
    loss, optimizer = get_loss_and_optimizer(model, base_learning_rate)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_learning_rate, max_lr=max_learning_rate,
                                                  cycle_momentum=False)
    best_accuracy = 0
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}/{epochs}')
        accuracy_train, loss_train = train(model, train_data_loader, optimizer, scheduler, loss, batch_size)
        accuracy_validation, loss_validation = validate(model, validation_data_loader, loss, batch_size)
        log_results(tensorboard_writer, accuracy_train, loss_train, accuracy_validation, loss_validation, epoch)
        if accuracy_validation > best_accuracy:
            best_accuracy = accuracy_validation
            save_model(path_to_saved_model, model, model_name, pretrained, accuracy_validation)
