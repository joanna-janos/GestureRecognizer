import torch
from torchvision import models


def _turn_off_backpropagation(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def _get_mnasnet(pretrained: bool,
                 train_only_last_layer: bool,
                 output_classes_count: int
                 ):
    model = models.mnasnet1_0(pretrained=pretrained)
    if train_only_last_layer:
        _turn_off_backpropagation(model)
    model.classifier[1] = torch.nn.Linear(1280, output_classes_count)
    return model


def _get_squeezenet(pretrained: bool,
                    train_only_last_layer: bool,
                    output_classes_count: int
                    ):
    model = models.squeezenet1_0(pretrained=pretrained)
    if train_only_last_layer:
        _turn_off_backpropagation(model)
    model.classifier._modules["1"] = torch.nn.Conv2d(512, output_classes_count, kernel_size=(1, 1))
    model.num_classes = output_classes_count
    return model


def _get_mobilenet(pretrained: bool,
                   train_only_last_layer: bool,
                   output_classes_count: int
                   ):
    model = models.mobilenet_v2(pretrained=pretrained)
    if train_only_last_layer:
        _turn_off_backpropagation(model)
    model.classifier[1] = torch.nn.Linear(model.last_channel, output_classes_count)
    return model


def get_model(model_name: str,
              train_only_last_layer: bool,
              pretrained: bool = True,
              output_classes_count: int = 8
              ):
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
