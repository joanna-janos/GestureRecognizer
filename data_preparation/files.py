import os
import re


def get_paths_and_gesture(data_directory):
    path_gesture = []
    for r, _, f in os.walk(data_directory):
        for file in f:
            if file.endswith('.JPG'):
                path_gesture.append((os.path.join(r, file), re.search(r'\d+', file).group()))
    return path_gesture
