"""
Google Colab Setup Script untuk Enhanced Anti-Overfitting Training
================================================================

Script ini untuk setup dan menjalankan enhanced training di Google Colab.

Urutan eksekusi:
1. Jalankan cell ini terlebih dahulu untuk setup environment
2. Upload dataset ke Colab
3. Jalankan training script
"""

# Import required libraries
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR, CosineAnnealingWarmRestarts
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

# Check if running in Colab
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Setup for Colab
if is_colab():
    print("🚀 Running in Google Colab environment")
    
    # Mount Google Drive (optional)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except Exception as e:
        print(f"⚠️ Google Drive mount failed: {e}")
    
    # Set working directory
    os.chdir('/content')
    print("📁 Working directory set to /content")
    
else:
    print("🖥️ Running in local environment")

print(f"🔧 PyTorch version: {torch.__version__}")
print(f"💻 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# Create necessary directories
def create_directories():
    """Create necessary directories for the project"""
    directories = [
        'src',
        'outputs',
        'data',
        'models'
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📂 Created directory: {dir_name}")

create_directories()
