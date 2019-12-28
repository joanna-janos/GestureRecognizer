import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, ColorJitter


class GestureDataset(Dataset):
    def __init__(self, paths, gestures, transform=None):
        self.paths = paths
        self.gestures = gestures
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(self.gestures[idx])


def _get_img_transformations():
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    return Compose([
        ToTensor(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)  # will be applied RANDOMLY
        # TODO: consider other transformations like horizontal/vertical flip where gestures will be choosen
    ])


def _create_dataloader(X, y, batch_size=64, shuffle=True):
    transforms = _get_img_transformations()
    dataset = GestureDataset(X, y, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def _split_into_train_and_validation(data, split_rate=0.2, seed=1024):
    X = [img for (img, gesture) in data]
    y = [gesture for (img, gesture) in data]
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=split_rate, random_state=seed)
    return X_train, X_validation, y_train, y_validation


def prepare_data(data):
    X_train, X_validation, y_train, y_validation = _split_into_train_and_validation(data)
    train_dataloader = _create_dataloader(X_train, y_train)
    validation_dataloader = _create_dataloader(X_validation, y_validation, False)
    return train_dataloader, validation_dataloader
