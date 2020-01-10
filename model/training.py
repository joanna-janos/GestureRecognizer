import typing

import torch
from tqdm import tqdm

from model.model_selector import get_model
from model.utils.directory import create_not_existing_directory


def train(model: torch.nn.Module,
          data_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss: typing.Callable,
          batch_size: int):
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

    print(f'\tAccuracy: {correct / (len(data_loader) * batch_size)}')
    print(f'\tLoss: {overall_loss / (len(data_loader) * batch_size)}')


def validate(model: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             loss: typing.Callable,
             batch_size: int):
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
    print(f'\tAccuracy: {correct / (len(data_loader) * batch_size)}')
    print(f'\tLoss: {overall_loss / (len(data_loader) * batch_size)}')


def save_model(path_to_saved_model, model_state_dict, model_name, train_only_last_layer):
    create_not_existing_directory(path_to_saved_model)
    if train_only_last_layer:
        path_to_saved_model_with_filename = path_to_saved_model + model_name + '_trained_only_last_layer.pt'
    else:
        path_to_saved_model_with_filename = path_to_saved_model + model_name + '_trained_everything.pt'
    torch.save(model_state_dict, path_to_saved_model_with_filename)


def train_and_validate(model_name: str,
                       train_data_loader: torch.utils.data.DataLoader,
                       validation_data_loader: torch.utils.data.DataLoader,
                       epochs: int,
                       path_to_saved_model: str,
                       batch_size: int,
                       train_only_last_layer: bool):
    model = get_model(model_name, train_only_last_layer)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}/{epochs}')
        train(model, train_data_loader, optimizer, loss, batch_size)
        validate(model, validation_data_loader, loss, batch_size)
    save_model(path_to_saved_model, model.state_dict(), model_name, train_only_last_layer)
