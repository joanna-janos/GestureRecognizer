import argparse

from input_parsing.subparsers import find, train


def parse_arguments():
    """ Parse user provided arguments. """
    parser = argparse.ArgumentParser(description='Gesture recognizer')

    parser.add_argument('--data_path',
                        type=str,
                        default='data/',
                        required=False,
                        help='Path to data directory')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        required=False,
                        help='Size of batch')

    parser.add_argument('--model_name',
                        type=str,
                        default='MnasNet',
                        required=False,
                        choices=('MnasNet', 'SqueezeNet', 'MobileNetV2'),
                        help='Model name. The only available are: MnasNet, SqueezeNet and MobileNetV2.')

    parser.add_argument('--pretrained',
                        type=bool,
                        default=True,
                        required=False,
                        help='True if model should be pretrained, False otherwise')

    parser.add_argument('--train_only_last_layer',
                        type=bool,
                        default=True,
                        required=False,
                        help='True if pretrained model should have only last layer learnable,'
                             'False if all weights should be adjusted during training')

    subparsers = parser.add_subparsers(help="Task to perform:", dest="task")
    find(subparsers)
    train(subparsers)

    return parser.parse_args()
