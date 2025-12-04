#!/usr/bin/env python3
"""
Training script for CBIS-DDSM Breast Cancer Classification.

This script allows you to train models with different configurations and compare results.

Usage:
    # Train ResNet34 with CLAHE (recommended)
    python train_script.py --backbone resnet34 --use-clahe --epochs 10
    
    # Train ResNet34 without CLAHE for comparison
    python train_script.py --backbone resnet34 --no-clahe --epochs 10
    
    # Train ResNet18 (smaller, faster)
    python train_script.py --backbone resnet18 --use-clahe --epochs 10
"""

import argparse
import sys
from core.train import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description='Train breast cancer classification model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet34',
        choices=['resnet18', 'resnet34', 'efficientnet_b0'],
        help='Model backbone architecture'
    )
    
    parser.add_argument(
        '--use-clahe',
        dest='use_clahe',
        action='store_true',
        default=True,
        help='Use CLAHE preprocessing (default)'
    )
    
    parser.add_argument(
        '--no-clahe',
        dest='use_clahe',
        action='store_false',
        help='Do not use CLAHE preprocessing'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Backbone:     {args.backbone}")
    print(f"CLAHE:        {args.use_clahe}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch Size:   {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 70 + "\n")
    
    # Run experiment
    try:
        trainer = run_experiment(
            backbone=args.backbone,
            use_clahe=args.use_clahe,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Best Validation Accuracy: {trainer.best_val_acc:.4f}")
        print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
        print(f"\nCheckpoints saved in: {trainer.save_dir}/")
        print(f"  - best_model.pth: Complete checkpoint")
        print(f"  - model.pth: Model weights only (for inference)")
        print(f"  - learning_curves_*.png: Training curves")
        print(f"  - confusion_matrix_*.png: Confusion matrix")
        print("=" * 70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
