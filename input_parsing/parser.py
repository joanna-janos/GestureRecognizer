import argparse


def parse_arguments():
    """ Parse user provided arguments. """
    parser = argparse.ArgumentParser(description='Gesture recognizer')

    parser.add_argument('--data_path',
                        type=str,
                        default='data/',
                        help='Path to data directory')

    parser.add_argument('--path_to_saved_model',
                        type=str,
                        default='trained_model/',
                        help='Path where trained model will be stored')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Size of batch')

    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs')

    parser.add_argument('--model_name',
                        type=str,
                        default='MnasNet',
                        help='Model name. The only available are: MnasNet, SqueezeNet and MobileNetV2.')

    parser.add_argument('--train_only_last_layer',
                        type=bool,
                        default=True,
                        help='True if pretrained model should have only last layer learnable,'
                             'False if all weights should be adjusted during training')

    return parser.parse_args()
