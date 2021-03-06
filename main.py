import torch

from data_preparation import data_supplier, files, mean_std_finder
from input_parsing import parser
from model import training
from model.utils import lr_finder

torch.manual_seed(0)

if __name__ == "__main__":
    args = parser.parse_arguments()
    paths_and_gestures = files.get_paths_and_gestures(args.data_path)
    train_data_loader, validation_data_loader = data_supplier.prepare_data(paths_and_gestures, args.batch_size,
                                                                           args.means, args.stds)
    print(f'Chosen task: {args.task}')

    if args.task == 'find_mean_std':
        mean_std_finder.find(train_data_loader)

    elif args.task == 'find_lr':
        lr_finder.find(args.model_name, args.pretrained, args.train_only_last_layer,
                       train_data_loader, args.min_learning_rate, args.max_learning_rate,
                       args.num_iter, args.step_mode)
    elif args.task == 'train':
        training.train_and_validate(args.model_name, args.pretrained, train_data_loader, validation_data_loader,
                                    args.epochs, args.path_to_saved_model, args.batch_size, args.train_only_last_layer,
                                    args.base_learning_rate, args.max_learning_rate, args.path_tensorboard)
