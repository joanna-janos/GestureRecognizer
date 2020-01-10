from data_preparation import data_supplier, files
from input_parsing import parser

if __name__ == "__main__":
    args = parser.parse_arguments()
    paths_and_gestures = files.get_paths_and_gestures(args.data_path)
    train_dataloader, validation_dataloader = data_supplier.prepare_data(paths_and_gestures, args.batch_size)
