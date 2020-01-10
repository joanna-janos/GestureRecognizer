from data_preparation import data_supplier, files
from input_parsing import parser
from model import training

if __name__ == "__main__":
    args = parser.parse_arguments()
    paths_and_gestures = files.get_paths_and_gestures(args.data_path)
    train_data_loader, validation_data_loader = data_supplier.prepare_data(paths_and_gestures, args.batch_size)
    training.train_and_validate(args.model_name, train_data_loader, validation_data_loader, args.epochs,
                                args.path_to_saved_model, args.batch_size, args.train_only_last_layer)
