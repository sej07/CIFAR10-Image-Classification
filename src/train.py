import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from src.model import CNN
from src.dataset import get_data_loaders
from src.utils import set_seed, save_checkpoint, plot_training_history
from src import config

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct =0 
    total = 0
    pbar = tqdm(train_loader, desc= 'Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postifx({'loss': loss.item()})
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='validation')
        for images, labels in pbar:
            images.labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    set_seed()
    os.makedirs(config.MODEL_DIR, exist_ok = True)
    os.makedirs(config.RESULTS_DIR, exist_ok = True)

    print('Loading data')
    train_loader, val_loader, test_loader = get_data_loaders()
    print('Creating model')
    model = CNN().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE, weight_decay= config.WEIGHT_DECAY)
    print(f"\nStarting training on {config.DEVICE}")
    print(f"Total epochs: {config.NUM_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}\n")
    
    train_losses = []
    val_losses= []
    train_accs =[]
    val_accs =[]
    best_val_acc = 0.0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n Epoch {epoch} / {config.NUM_EPOCHS}")
        print('-' * 50)
        train_loss, train_acc = train_one_epoch(model , val_loader, criterion, optimizer, config.DEVICE)
        val_loss , val_acc = validate(model, val_loader, criterion, config.DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f'\n Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}')
        print(f'\n Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, val_loss, val_acc, 
                os.path.join(config.MODEL_DIR, 'best_model.pth')
                )
            print(f'New best model saved (Val Acc: {val_acc:.2f}%)')
    print("\n Training complete")
    print('Best Validation accuracy: {best_val_acc:.2f}%')
    plot_path = os.path.join(config.RESULTS_DIR, 'training_curves.png')
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    print("\n Training curves saved to {plot_path}")
    print(f"Best model saved to {os.path.join(config.MODEL_DIR, 'best_model.pth')}")

if __name__ == '__main__':
    main()