import argparse


def find_lr(subparsers):
    """`find_lr` sub command parser."""

    sub_parser = subparsers.add_parser(
        "find_lr",
        help="Find neural network learning rate using Leslie Smith's LR finder.",
    )

    sub_parser.add_argument('--min_learning_rate',
                            type=float,
                            default=1e-7,
                            required=False,
                            help='Minimum learning rate value used for search')

    sub_parser.add_argument('--max_learning_rate',
                            type=float,
                            default=1,
                            required=False,
                            help='Maximum learning rate value used for search')

    sub_parser.add_argument('--num_iter',
                            type=int,
                            default=200,
                            required=False,
                            help='Number of iterations after which test will be performed')

    sub_parser.add_argument('--step_mode',
                            type=str,
                            default='linear',
                            choices=('linear', 'exp'),
                            required=False,
                            help="Mode to perform search. 'linear' stands for original Leslie's (longer),"
                                 " while 'exp' for fastai's")


def find_mean_std(subparsers):
    """`find_mean_std` sub command parser."""

    subparsers.add_parser(
        "find_mean_std",
        help="Find mean and std to normalize data",
    )


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
                            default=100,
                            required=False,
                            help='Number of epochs')

    sub_parser.add_argument('--base_learning_rate',
                            type=float,
                            default=1e-3,
                            required=False,
                            help='Base learning rate value')

    sub_parser.add_argument('--max_learning_rate',
                            type=float,
                            default=1e-2,
                            required=False,
                            help='Max learning rate value')
