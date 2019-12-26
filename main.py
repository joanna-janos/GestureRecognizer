from data_preparation import data_loading, files
from input_parsing import parser

if __name__ == "__main__":
    args = parser.parse_arguments()
    paths_and_gesture = files.get_paths_and_gesture(args.data_path)
    train_dataloader, validation_dataloader = data_loading.prepare_data(paths_and_gesture)
