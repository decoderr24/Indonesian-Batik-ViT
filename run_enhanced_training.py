#!/usr/bin/env python3
"""
Enhanced Anti-Overfitting Training Script for Batik Vision Project

This script implements state-of-the-art anti-overfitting techniques including:
- Advanced data augmentation (Mixup, CutMix, TrivialAugmentWide)
- Label smoothing and Focal Loss for class imbalance
- Enhanced dropout and regularization
- Test Time Augmentation (TTA)
- Advanced learning rate scheduling
- Gradient clipping and early stopping

Usage:
    python run_enhanced_training.py

Requirements:
    - PyTorch with CUDA support
    - torchvision
    - scikit-learn
    - matplotlib
    - seaborn
    - tensorboard
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def main():
    """
    Main function to run enhanced anti-overfitting training.
    """
    print("="*80)
    print("BATIK VISION - ENHANCED ANTI-OVERFITTING TRAINING")
    print("="*80)
    print()
    print("This script implements state-of-the-art techniques to combat overfitting:")
    print("✓ Advanced Data Augmentation (Mixup, CutMix, TrivialAugmentWide)")
    print("✓ Label Smoothing + Focal Loss for class imbalance")
    print("✓ Enhanced Dropout (0.7) + Weight Decay (2e-3)")
    print("✓ Test Time Augmentation (TTA)")
    print("✓ Cosine Annealing with Warm Restarts")
    print("✓ Gradient Clipping + Early Stopping")
    print("✓ Advanced Regularization Techniques")
    print()
    
    try:
        # Import and run the enhanced training
        from src.train_enhanced_anti_overfitting import main as run_enhanced_training
        run_enhanced_training()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all required dependencies are installed:")
        print("pip install torch torchvision scikit-learn matplotlib seaborn tensorboard")
        return 1
        
    except Exception as e:
        print(f"❌ Training Error: {e}")
        print("Please check your data path and model configuration.")
        return 1
    
    print()
    print("="*80)
    print("ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
