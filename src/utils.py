import torch 
import random
import numpy as np
import matplotlib.pyplot as plt
from src import config

#set random seed
def set_seed(seed = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

#save checkpoint
def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    checkpoint = {
        'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    print(f'Checkpoint saved to {filepath}')

#load checkpoint
def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath, map_location = config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    print(f'Checkpoint loaded from {filepath} (Epoch {epoch})')
    return epoch, loss, accuracy

#plot training history
def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    print(f"DEBUG: train_losses length: {len(train_losses)}")
    print(f"DEBUG: val_losses length: {len(val_losses)}")
    print(f"DEBUG: train_accs length: {len(train_accs)}")
    print(f"DEBUG: val_accs length: {len(val_accs)}")
    
    epochs = range(1, len(train_losses) + 1)
    print(f"DEBUG: epochs: {list(epochs)}")
    fig , (ax1, ax2) = plt.subplots(1,2,figsize = (12,4))

    #plot losses
    ax1.plot(epochs, train_losses, 'b-', label ='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label ='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    #plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label ='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label ='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy(%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")
    