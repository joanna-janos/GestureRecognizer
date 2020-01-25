import argparse

from input_parsing.subparsers import find_lr, train, find_mean_std


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
                        default=8,
                        required=False,
                        help='Size of batch')

    parser.add_argument('--model_name',
                        type=str,
                        default='SqueezeNet',
                        required=False,
                        choices=('MnasNet', 'SqueezeNet', 'MobileNetV2'),
                        help='Model name. The only available are: MnasNet, SqueezeNet and MobileNetV2.')

    parser.add_argument('--pretrained',
                        default=True,
                        action="store_false",
                        required=False,
                        help='True if model should be pretrained, False otherwise')

    parser.add_argument('--train_only_last_layer',
                        default=True,
                        action="store_false",
                        required=False,
                        help='True if pretrained model should have only last layer learnable,'
                             'False if all weights should be adjusted during training')

    parser.add_argument('--means',
                        nargs="+",
                        type=float,
                        default=[0.0, 0.0, 0.0],
                        required=False,
                        help='Mean values used to data normalization, '
                             'found by find_mean_std task: [0.7315, 0.6840, 0.6410], '
                             'default: no normalization')

    parser.add_argument('--stds',
                        nargs="+",
                        type=float,
                        default=[1.0, 1.0, 1.0],
                        required=False,
                        help='Std values used to data normalization, '
                             'found by find_mean_std task: [0.4252, 0.4546, 0.4789], '
                             'default: no normalization')

    subparsers = parser.add_subparsers(help="Task to perform:", dest="task")
    find_lr(subparsers)
    train(subparsers)
    find_mean_std(subparsers)

    return parser.parse_args()
