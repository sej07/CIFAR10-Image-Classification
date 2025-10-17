import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from src import config

#training transformations with augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding= 4),
    transforms.ToTensor(),
    transforms.Normalize(config.CIFAR_MEAN, config.CIFAR_STD)
])

#validation transformations
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(config.CIFAR_MEAN, config.CIFAR_STD)
])

#loading dataset
'''Returns train_loader: DataLoader for training set, 
                        val_loader: DataLoader for validation set,
                        test_loader: DataLoader for test set'''
def get_data_loaders():
    train_dataset= datasets.CIFAR10(
        root = config.DATA_DIR,
        train = True,
        download=True,
        transform= train_transform
    )
    test_dataset= datasets.CIFAR10(
        root= config.DATA_DIR,
        train = False,
        download=True, 
        transform=valid_transform
    )
    train_size = int(config.TRAIN_SPLIT *len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size])
    val_dataset.dataset.transform = valid_transform
    train_loader = DataLoader(
        train_dataset, 
        batch_size= config.BATCH_SIZE,
        shuffle=True,
        num_workers= config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    return train_loader, val_loader, test_loader