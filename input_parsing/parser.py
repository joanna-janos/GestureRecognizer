import argparse


def parse_arguments():
    """ Parse user provided arguments. """
    parser = argparse.ArgumentParser(description='Gesture recognizer')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to data directory')
    return parser.parse_args()
