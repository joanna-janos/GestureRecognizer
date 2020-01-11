from torch.utils.tensorboard import SummaryWriter

from model.utils.directory import create_not_existing_directory


def setup_tensorboard(path_tensorboard, model_name, train_only_last_layer):
    if train_only_last_layer:
        path_to_results_wth_filename = path_tensorboard + model_name + '_trained_only_last_layer'
    else:
        path_to_results_wth_filename = path_tensorboard + model_name + '_trained_everything'

    create_not_existing_directory(path_tensorboard)
    return SummaryWriter(path_to_results_wth_filename)


def log_results(writer, accuracy_train, loss_train, accuracy_validation, loss_validation, epoch):
    writer.add_scalar('Train/Accuracy', accuracy_train, epoch)
    writer.add_scalar('Train/Loss', loss_train, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy_validation, epoch)
    writer.add_scalar('Validation/Loss', loss_validation, epoch)
