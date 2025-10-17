from src.dataset import get_data_loaders
from src import config

if __name__ =='__main__':

    print("loading CIFAR 10 dataset")
    train_loader , val_loader, test_loader = get_data_loaders()
    print("\n Dataset Statistics \n")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images , labels = next(iter(train_loader))
    print(f"\n Batch shape: {images.shape}")
    print(f"\n Labels shape: {labels.shape}")
    print(f"\n Device: {config.DEVICE}")