import torch
import os
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import seaborn as sns
from tqdm import tqdm
from configs.A_config import Config
from torch.utils.tensorboard import SummaryWriter


def save_checkpoint(model, optimizer, epoch, train_losses, train_accuracies, val_losses, val_accuracies, best_val_acc, checkpoint_dir, epoch_interval=5):
    """Helper function to save the model checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")


def save_final_model(model, final_model_name, checkpoint_dir):
    """Save the final model with necessary configuration metadata for fusion"""
    final_model_path = os.path.join(checkpoint_dir, final_model_name.split('.')[0] + '_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': Config.CLASS_TO_IDX,
        'input_channels': Config.INPUT_CHANNELS,
        'spectrogram_size': Config.SPECTROGRAM_SIZE,
        'num_classes': Config.NUM_CLASSES
    }, final_model_path)
    print(f"Final model saved for fusion at: {final_model_path}")


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, output_dir):
    """Helper function to plot loss and accuracy graphs"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))


def plot_confusion_matrix(all_labels, all_predictions, output_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(all_labels, all_predictions)
    if cm.size > 0:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=Config.CLASSES, yticklabels=Config.CLASSES, cmap='Blues')
        plt.xlabel('Prediction', fontsize=13)
        plt.ylabel('Actual', fontsize=13)
        plt.title(f'Confusion Matrix', fontsize=17)
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))


def plot_roc_curve(all_labels, all_predictions, output_dir):
    """Plot ROC Curve"""
    one_hot_labels = label_binarize(all_labels, classes=np.arange(len(Config.CLASSES)))
    probabilities = torch.zeros(len(all_predictions), len(Config.CLASSES))
    probabilities.scatter_(1, torch.unsqueeze(torch.tensor(all_predictions), 1), 1)
    probabilities = nn.Softmax(dim=1)(probabilities).detach().numpy()

    plt.figure(figsize=(10, 8))
    for i in range(len(Config.CLASSES)):
        fpr, tpr, _ = roc_curve(one_hot_labels[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve for class {Config.CLASSES[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Multiclass')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'auroc_curve.png'))
    plt.show()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, final_model_name='best_model.pth', checkpoint_dir='output/checkpoints/'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Tensorboard Setup (optional)
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

    best_val_acc = 0.0
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    all_predictions, all_labels = [], []

    step_size = 5
    gamma = 0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", position=0, leave=True, colour='green'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_acc = correct / total
        train_accuracies.append(train_acc)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                

                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_acc = correct / total
        val_accuracies.append(val_acc)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        # Adjust Learning Rate
        scheduler.step()

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, train_losses, train_accuracies, val_losses, val_accuracies, best_val_acc, checkpoint_dir)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, final_model_name))
            print(f"Best model saved with accuracy: {best_val_acc:.4f}")

    # Final Model and Metrics
    save_final_model(model, final_model_name, checkpoint_dir)
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, checkpoint_dir)
    plot_confusion_matrix(all_labels, all_predictions, checkpoint_dir)
    plot_roc_curve(all_labels, all_predictions, checkpoint_dir)

    writer.close()  # Close TensorBoard writer
