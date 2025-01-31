import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from configs.V_config import Config

class TrainingPipeline:
    def __init__(self, model, train_loader, test_loader, num_classes, experiment_name=None):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model and data
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.experiment_name = experiment_name  # Store experiment name here

        # Outputs
        self.output_dir = Config.OUTPUT_DIR
        self.checkpoint_dir = Config.CHECKPOINT_DIR
        self.results_dir = Config.RESULTS_DIR
        
        # Training parameters
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        
        # Tracking variables
        self.best_test_loss = float('inf')
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training', unit='batch')
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Accuracy': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        progress_bar = tqdm(self.test_loader, desc='Validation', unit='batch')
        with torch.no_grad():
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Accuracy': f'{100. * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy, all_preds, all_targets

    def run_training(self, epochs=Config.NUM_EPOCHS):
        print(f"Starting training for {epochs} epochs")
        for epoch in tqdm(range(epochs), desc='Epochs', unit='epoch'):
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc, preds, targets = self.validate()
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            
            self.scheduler.step(test_loss)
            
            # Save best model
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self._save_best_model(epoch)
            
            # Checkpoint
            self._save_checkpoint(epoch)
        
        # Final evaluations
        self._plot_metrics()
        self._plot_confusion_matrix(preds, targets)
        self._save_model_fusion_info()

    def _save_best_model(self, epoch):
        model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_test_loss
        }, model_path)

    def _save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }, checkpoint_path)

    def _plot_metrics(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss Curves')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'metrics.png'))
        plt.close()

    def _plot_confusion_matrix(self, preds, targets):
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=Config.CLASSES, 
                    yticklabels=Config.CLASSES)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()

    def _save_model_fusion_info(self):
        fusion_info = {
            'model_architecture': str(self.model),
            'input_shape': next(iter(self.train_loader))[0].shape,
            'num_classes': self.num_classes,
            'class_mapping': Config.CLASS_TO_IDX,
            'best_test_loss': self.best_test_loss,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }
        
        torch.save(fusion_info, os.path.join(self.results_dir, 'video_model_f.pth'))