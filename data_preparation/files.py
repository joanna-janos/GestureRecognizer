import os
import re


def get_paths_and_gestures(data_directory: str):
    """ Get paths to all images representing gestures
    and corresponding gestures names under given directory.

    Arguments
    ----------
    data_directory : str
        Path to directory containing JPG images

    Returns
    -------
    List[(str, str)]
        Paths to images and gestures names
    """
    path_gesture = []
    for r, _, f in os.walk(data_directory):
        for file in f:
            if file.endswith('.JPG'):
                path_gesture.append((os.path.join(r, file), re.search(r'\d+', file).group()))
    return path_gesture
