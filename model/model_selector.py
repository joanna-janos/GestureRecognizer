import torch
from torchvision import models


def _turn_off_backpropagation(model: torch.nn.Module):
    """ Turn backpropagation for all layers in given model

    Arguments
    ----------
    model : torch.nn.Module
        Model for which weights update will be turned off
    """
    for param in model.parameters():
        param.requires_grad = False


def _get_mnasnet(pretrained: bool,
                 train_only_last_layer: bool,
                 output_classes_count: int
                 ) -> torch.nn.Module:
    """ Get MnasNet and adjust to fit gesture recognition task

    Arguments
    ----------
    pretrained : bool
        Path to directory containing JPG images
    train_only_last_layer : bool
        True if pretrained model should have only last layer learnable,
        False if all weights should be adjusted during training
    output_classes_count : int
        Number of labels in classification

    Returns
    -------
    torch.nn.Module
        MnasNet
    """
    model = models.mnasnet1_0(pretrained=pretrained)
    if train_only_last_layer:
        _turn_off_backpropagation(model)

    if pretrained:
        # unfreeze last layers
        model.layers[13][0].layers[0].weight.requires_grad = True
        model.layers[13][0].layers[1].weight.requires_grad = True
        model.layers[13][0].layers[3].weight.requires_grad = True
        model.layers[13][0].layers[4].weight.requires_grad = True
        model.layers[13][0].layers[6].weight.requires_grad = True
        model.layers[13][0].layers[7].weight.requires_grad = True

        model.layers[14].weight.requires_grad = True
        model.layers[15].weight.requires_grad = True

    # change classifier to fit to current task
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(1280, 256),
        torch.nn.Dropout(p=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(256, output_classes_count)
    )

    return model


def _get_squeezenet(pretrained: bool,
                    train_only_last_layer: bool,
                    output_classes_count: int
                    ) -> torch.nn.Module:
    """ Get SqueezeNet and adjust to fit gesture recognition task

    Arguments
    ----------
    pretrained : bool
        Path to directory containing JPG images
    train_only_last_layer : bool
        True if pretrained model should have only last layer learnable,
        False if all weights should be adjusted during training
    output_classes_count : int
        Number of labels in classification

    Returns
    -------
    torch.nn.Module
        SqueezeNet
    """
    model = models.squeezenet1_0(pretrained=pretrained)
    if train_only_last_layer:
        _turn_off_backpropagation(model)

    if pretrained:
        # unfreeze last layers
        model.features[12].squeeze.weight.requires_grad = True
        model.features[12].expand1x1.weight.requires_grad = True
        model.features[12].expand3x3.weight.requires_grad = True

    # change classifier to fit to current task
    model.classifier[1] = torch.nn.Conv2d(512, output_classes_count, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = output_classes_count
    return model


def _get_mobilenet(pretrained: bool,
                   train_only_last_layer: bool,
                   output_classes_count: int
                   ) -> torch.nn.Module:
    """ Get MobileNetV2 and adjust to fit gesture recognition task

    Arguments
    ----------
    pretrained : bool
        Path to directory containing JPG images
    train_only_last_layer : bool
        True if pretrained model should have only last layer learnable,
        False if all weights should be adjusted during training
    output_classes_count : int
        Number of labels in classification

    Returns
    -------
    torch.nn.Module
        MobileNetV2
    """
    model = models.mobilenet_v2(pretrained=pretrained)
    if train_only_last_layer:
        _turn_off_backpropagation(model)

    if pretrained:
        # unfreeze last layers
        model.features[17].conv[2].weight.requires_grad = True
        model.features[17].conv[3].weight.requires_grad = True

        model.features[18][0].weight.requires_grad = True
        model.features[18][1].weight.requires_grad = True

    # change classifier to fit to current task
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.last_channel, 256),
        torch.nn.Dropout(p=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(256, output_classes_count)
    )
    return model


def get_model(model_name: str,
              train_only_last_layer: bool,
              pretrained: bool,
              output_classes_count: int = 8
              ) -> torch.nn.Module:
    """ Get model using name from torchvision.models

    Arguments
    ----------
    model_name : str
        Name of model to use in task
    train_only_last_layer : bool
        True if pretrained model should have only last layer learnable,
        False if all weights should be adjusted during training
    pretrained : bool
        Path to directory containing JPG images
    output_classes_count : int
        Number of labels in classification

    Returns
    -------
    torch.nn.Module
        Model
    """
    if model_name == 'MnasNet':
        model = _get_mnasnet(pretrained, train_only_last_layer, output_classes_count)
    elif model_name == 'SqueezeNet':
        model = _get_squeezenet(pretrained, train_only_last_layer, output_classes_count)
    elif model_name == 'MobileNetV2':
        model = _get_mobilenet(pretrained, train_only_last_layer, output_classes_count)
    else:
        raise ValueError('Incorrect model name. The only available are: MnasNet, SqueezeNet and MobileNetV2.')

    print(model)
    return model
