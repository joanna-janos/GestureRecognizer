import argparse


def find(subparsers):
    """`find` sub command parser."""

    sub_parser = subparsers.add_parser(
        "find",
        help="Find neural network learning rate using Leslie Smith's LR finder.",
    )

    sub_parser.add_argument('--path_to_visualisations',
                            type=str,
                            default='results/visualisations_lr_finder/',
                            required=False,
                            help='Path where results of lr finder will be stored')

    sub_parser.add_argument('--min_learning_rate',
                            type=float,
                            default=0.001,
                            required=False,
                            help='Minimum learning rate value used for search')

    sub_parser.add_argument('--max_learning_rate',
                            type=float,
                            default=0.01,
                            required=False,
                            help='Maximum learning rate value used for search')

    sub_parser.add_argument('--num_iter',
                            type=int,
                            default=100,
                            required=False,
                            help='Number of iterations after which test will be performed')

    sub_parser.add_argument('--step_mode',
                            type=str,
                            default='exp',
                            choices=('linear', 'exp'),
                            required=False,
                            help="Mode to perform search. 'linear' stands for original Leslie's (longer),"
                                 " while 'exp' for fastai's")


def train(subparsers):
    """`train` sub command parser."""

    sub_parser = subparsers.add_parser(
        "train",
        help="Train neural network",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    sub_parser.add_argument('--path_to_saved_model',
                            type=str,
                            default='results/trained_model/',
                            required=False,
                            help='Path where trained model will be stored')

    sub_parser.add_argument('--path_tensorboard',
                            type=str,
                            default='results/tensorboard/',
                            required=False,
                            help='Path where tensorboard results will be stored')

    sub_parser.add_argument('--epochs',
                            type=int,
                            default=4,
                            required=False,
                            help='Number of epochs')

    sub_parser.add_argument('--learning_rate',
                            type=float,
                            default=0.0001,
                            required=False,
                            help='Learning rate value')
