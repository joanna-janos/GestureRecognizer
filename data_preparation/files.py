import os
import typing


def get_paths_and_gestures(data_directory: str,
                           gestures: typing.Tuple[str] = ('1', '2', '3', '4', '5', 'A', 'O', 'U')
                           ) -> typing.List[typing.Tuple[str, str]]:
    """ Get paths to all images representing gestures and corresponding gesture name
    under given directory.

    Arguments
    ----------
    data_directory : str
        Path to directory containing JPG images
    gestures : typing.Tuple[str]
        Gestures taken into account

    Returns
    -------
    typing.List[typing.Tuple[str, str]]
        Paths to images and gestures names
    """
    path_gesture = []
    for r, _, f in os.walk(data_directory):
        if r.endswith(gestures):
            for file in f:
                if file.endswith('.JPG'):
                    path_gesture.append((os.path.join(r, file), r[-1]))
    return path_gesture
