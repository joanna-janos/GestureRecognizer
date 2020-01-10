import argparse


def parse_arguments():
    """ Parse user provided arguments. """
    parser = argparse.ArgumentParser(description='Gesture recognizer')

    parser.add_argument('--data_path',
                        type=str,
                        default='data/',
                        help='Path to data directory')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Size of batch')

    parser.add_argument('--model_name',
                        type=str,
                        default='MnasNet',
                        help='Model name. The only available are: MnasNet, SqueezeNet and MobileNetV2.')

    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='Number of epochs')

    return parser.parse_args()
