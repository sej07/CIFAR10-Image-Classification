import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from src.model import CNN
from src.dataset import get_data_loaders
from src.utils import load_checkpoint
from src import config

def test_model(model, test_loader, device):
    model.eval()
    all_preds =[]
    all_labels =[]
    correct = 0
    total = 0
    print("Testing Model")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    return np.array(all_preds), np.array(all_labels), test_acc

def plot_confusion_matrix(all_labels, all_preds, class_names, save_path):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm ,annot = True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_per_class_accuracy(all_labels, all_preds, class_names, save_path):
    accuracies = []
    for i in range(len(class_names)):
        class_mask = all_labels == i
        class_correct = (all_preds[class_mask] == all_labels[class_mask]).sum()
        class_total = class_mask.sum()
        class_acc = 100 * class_correct / class_total if class_total > 0 else 0
        accuracies.append(class_acc)
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, accuracies, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Per-class accuracy plot saved to {save_path}")

def visualize_predictions(model, test_loader, class_names, device, save_path, num_images=20):
    model.eval()
    images_shown = 0
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.ravel()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                # Move image to CPU and denormalize
                img = images[i].cpu()
                img = img * torch.tensor(config.CIFAR_STD).view(3, 1, 1) + torch.tensor(config.CIFAR_MEAN).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                # Plot
                axes[images_shown].imshow(img)
                axes[images_shown].axis('off')
                true_label = class_names[labels[i]]
                pred_label = class_names[predicted[i]]
                color = 'green' if labels[i] == predicted[i] else 'red'
                axes[images_shown].set_title(f'True: {true_label}\nPred: {pred_label}', 
                                            color=color, fontsize=10)
                images_shown += 1
            if images_shown >= num_images:
                break
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Sample predictions saved to {save_path}")

def main():
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    print("Loading data...")
    _, _, test_loader = get_data_loaders()
    print("Loading model...")
    model = CNN().to(config.DEVICE)
    # Load best checkpoint
    checkpoint_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    epoch, loss, accuracy = load_checkpoint(model, None, checkpoint_path)
    print(f"Loaded model from epoch {epoch} with validation accuracy: {accuracy:.2f}%\n")
    # Test the model
    all_preds, all_labels, test_acc = test_model(model, test_loader, config.DEVICE)
    print(f"Test Accuracy: {test_acc:.2f}%")
    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    # Plot confusion matrix
    cm_path = os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(all_labels, all_preds, class_names, cm_path)
    # Plot per-class accuracy
    per_class_path = os.path.join(config.RESULTS_DIR, 'per_class_accuracy.png')
    plot_per_class_accuracy(all_labels, all_preds, class_names, per_class_path)
    # Visualize sample predictions
    samples_path = os.path.join(config.RESULTS_DIR, 'sample_predictions.png')
    visualize_predictions(model, test_loader, class_names, config.DEVICE, samples_path)
    print(f"\nAll evaluation results saved to {config.RESULTS_DIR}/")

if __name__ == '__main__':
    main()