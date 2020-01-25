import torch
from tqdm import tqdm


def find(data_loader: torch.utils.data.DataLoader):
    """ Find mean and std values which should be used to data normalization.
    Based on solution provided on PyTorch forum: https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/2

    Arguments
    ----------
    data_loader : torch.utils.data.DataLoader
        Iterated over data whose mean and std will be computed
    """
    mean = 0.
    std = 0.
    for x, _ in tqdm(data_loader):
        batch_samples = x.size(0)
        images = x.view(batch_samples, x.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    print(f'MEAN: {mean / len(data_loader.dataset)}')
    print(f'STD: {std / len(data_loader.dataset)}')
