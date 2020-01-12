from data_preparation import data_supplier, files
from input_parsing import parser
from model import training
from model.utils import lr_finder

if __name__ == "__main__":
    args = parser.parse_arguments()
    paths_and_gestures = files.get_paths_and_gestures(args.data_path)
    train_data_loader, validation_data_loader = data_supplier.prepare_data(paths_and_gestures, args.batch_size)

    print(f'Chosen task: {args.task}')

    if args.task == 'find':
        lr_finder.find_optimal_learning_rate(args.model_name, args.pretrained, args.train_only_last_layer,
                                             train_data_loader, args.min_learning_rate, args.max_learning_rate,
                                             args.num_iter, args.step_mode)

    if args.task == 'train':
        training.train_and_validate(args.model_name, args.pretrained, train_data_loader, validation_data_loader,
                                    args.epochs, args.path_to_saved_model, args.batch_size, args.train_only_last_layer,
                                    args.learning_rate, args.path_tensorboard)
