# Model selection

The purpose of this project is to create a model which can be used as a component of mobile application. In this situation model need to be small, fast and still provides accurate results.

## Literature

I took into account three promising architectures:
1. [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626)
1. [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
1. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)


## Transfer learning

Above models are available in PyTorch (see: [TORCHVISION.MODELS](https://pytorch.org/docs/stable/torchvision/models.html)) so it may be valuable to use pretrained versions of them.

All of mentioned models were pretrained on [ImageNet](http://www.image-net.org).