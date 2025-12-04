"""
Training module for CBIS-DDSM Breast Cancer Classification.

This module provides a complete training pipeline with:
- Train/Validation split
- Class-weighted loss for imbalanced data
- Metrics: Accuracy, Precision, Recall, F1, AUC
- Learning curve visualization
- Model checkpointing
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from collections import defaultdict
import logging

from .config import Config
from .model import BreastCancerModel
from .preprocessing import CBISDDSMDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for breast cancer classification.
    
    Args:
        backbone: Model backbone name ('resnet18', 'resnet34', 'efficientnet_b0')
        use_clahe: Whether to use CLAHE preprocessing
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save checkpoints and plots
    """
    
    def __init__(
        self,
        backbone: str = None,
        use_clahe: bool = True,
        batch_size: int = None,
        learning_rate: float = None,
        num_epochs: int = None,
        device: str = None,
        save_dir: str = None
    ):
        self.backbone = backbone or Config.BACKBONE
        self.use_clahe = use_clahe
        self.batch_size = batch_size or Config.BATCH_SIZE
        self.learning_rate = learning_rate or Config.LEARNING_RATE
        self.num_epochs = num_epochs or Config.NUM_EPOCHS
        self.save_dir = save_dir or Config.SAVE_DIR
        
        # Determine device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize tracking
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Backbone: {self.backbone}")
        logger.info(f"  CLAHE: {self.use_clahe}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Epochs: {self.num_epochs}")
    
    def prepare_data(self):
        """Load dataset and create train/val split."""
        logger.info("Loading dataset...")
        
        # Load full dataset
        full_dataset = CBISDDSMDataset(
            use_clahe=self.use_clahe,
            phase='train'  # We'll handle augmentation separately
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(Config.TRAIN_VAL_SPLIT * total_size)
        val_size = total_size - train_size
        
        # Random split
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Compute class weights for imbalanced data
        labels = full_dataset.get_labels()
        class_counts = np.bincount(labels)
        total = sum(class_counts)
        self.class_weights = torch.tensor(
            [total / (len(class_counts) * count) for count in class_counts],
            dtype=torch.float32
        ).to(self.device)
        
        logger.info(f"Dataset split: {train_size} train, {val_size} val")
        logger.info(f"Class distribution: {dict(zip(Config.CLASSES, class_counts.tolist()))}")
        logger.info(f"Class weights: {self.class_weights.tolist()}")
        
    def build_model(self):
        """Initialize model, optimizer, and loss function."""
        logger.info(f"Building model with backbone: {self.backbone}")
        
        # Create model
        self.model = BreastCancerModel(backbone_name=self.backbone)
        self.model = self.model.to(self.device)
        
        logger.info(f"Model parameters: {self.model.get_num_params():,}")
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of malignant
        
        # Calculate metrics
        val_loss = running_loss / len(self.val_loader)
        metrics = self.compute_metrics(all_labels, all_preds, all_probs)
        
        return val_loss, metrics
    
    def compute_metrics(self, labels, preds, probs=None):
        """Compute classification metrics."""
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
        }
        
        if probs is not None:
            try:
                metrics['auc'] = roc_auc_score(labels, probs)
            except ValueError:
                metrics['auc'] = 0.0
        
        return metrics
    
    def train(self):
        """Full training loop."""
        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics.get('auc', 0))
            
            epoch_time = time.time() - epoch_start
            
            # Log epoch results
            logger.info(f"\nEpoch {epoch+1}/{self.num_epochs} ({epoch_time:.1f}s)")
            logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
            logger.info(f"  F1: {val_metrics['f1']:.4f} | AUC: {val_metrics.get('auc', 0):.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"  ✓ New best model saved!")
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best Val Loss: {self.best_val_loss:.4f}")
        logger.info(f"Best Val Accuracy: {self.best_val_acc:.4f}")
        logger.info("=" * 60)
        
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'backbone': self.backbone,
            'use_clahe': self.use_clahe,
            'history': dict(self.history)
        }
        
        # Save latest
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            
            # Also save just the model weights for inference
            model_only_path = os.path.join(self.save_dir, 'model.pth')
            torch.save(self.model.state_dict(), model_only_path)
    
    def plot_learning_curves(self, save_path: str = None):
        """Plot training and validation curves."""
        if not self.history['train_loss']:
            logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(epochs, self.history['val_f1'], 'g-', label='Validation F1')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUC
        axes[1, 1].plot(epochs, self.history['val_auc'], 'm-', label='Validation AUC')
        axes[1, 1].set_title('AUC-ROC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.suptitle(f'{self.backbone} | CLAHE={self.use_clahe}', fontsize=14)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, f'learning_curves_{self.backbone}_clahe{self.use_clahe}.png')
        
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Learning curves saved to {save_path}")
    
    def evaluate(self, test_loader=None):
        """Final evaluation with detailed metrics."""
        if test_loader is None:
            test_loader = self.val_loader
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # Classification report
        print("\n" + "=" * 60)
        print("Classification Report:")
        print("=" * 60)
        print(classification_report(all_labels, all_preds, target_names=Config.CLASSES))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm)
        
        return all_labels, all_preds, all_probs
    
    def plot_confusion_matrix(self, cm, save_path: str = None):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        classes = Config.CLASSES
        ax.set(
            xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes,
            yticklabels=classes,
            ylabel='True Label',
            xlabel='Predicted Label',
            title=f'Confusion Matrix\n{self.backbone} | CLAHE={self.use_clahe}'
        )
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, f'confusion_matrix_{self.backbone}_clahe{self.use_clahe}.png')
        
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix saved to {save_path}")


def run_experiment(
    backbone: str = 'resnet34',
    use_clahe: bool = True,
    num_epochs: int = 10,
    batch_size: int = 32
):
    """
    Run a training experiment.
    
    Args:
        backbone: Model backbone
        use_clahe: Whether to use CLAHE
        num_epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        Trainer object with results
    """
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {backbone} | CLAHE={use_clahe}")
    print("=" * 60 + "\n")
    
    # Create trainer
    trainer = Trainer(
        backbone=backbone,
        use_clahe=use_clahe,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    # Prepare data
    trainer.prepare_data()
    
    # Build model
    trainer.build_model()
    
    # Train
    history = trainer.train()
    
    # Plot learning curves
    trainer.plot_learning_curves()
    
    # Final evaluation
    trainer.evaluate()
    
    return trainer


if __name__ == "__main__":
    # Run default experiment
    trainer = run_experiment(
        backbone='resnet34',
        use_clahe=True,
        num_epochs=10
    )
