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
    """
    Download CIFAR-10, split into train/val, and return DataLoaders.
    
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
    """
    # Download CIFAR-10 training data WITH AUGMENTATION
    full_train_dataset = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Download CIFAR-10 test data
    test_dataset = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=False,
        download=True,
        transform=valid_transform
    )
    
    # Split training data into train and validation
    train_size = int(config.TRAIN_SPLIT * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # Create indices for split
    indices = list(range(len(full_train_dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create separate datasets with different transforms
    train_dataset = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=valid_transform  # No augmentation for validation!
    )
    
    # Use Subset with specific indices
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_subset,
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