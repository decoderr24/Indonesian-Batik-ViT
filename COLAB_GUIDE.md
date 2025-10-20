# Panduan Enhanced Training di Google Colab

## 🚀 File yang Perlu Di-run di Google Colab

Untuk menjalankan enhanced anti-overfitting training di Google Colab, Anda hanya perlu menjalankan **2 file**:

### 1. **Setup Script** (Cell Pertama)
```python
# Jalankan file: colab_setup.py
# Atau copy-paste isi file ini ke cell pertama
```

### 2. **Enhanced Training Script** (Cell Kedua)
```python
# Jalankan file: colab_enhanced_training.py
# Atau copy-paste isi file ini ke cell kedua
```

## 📋 Langkah-langkah Detail

### **Langkah 1: Setup Environment**
Jalankan cell pertama yang berisi setup untuk Colab:

```python
# Cell 1 - Setup Colab Environment
exec(open('colab_setup.py').read())
```

### **Langkah 2: Upload Dataset**
1. Upload dataset batik Anda ke Colab
2. Extract jika dalam format .zip
3. Update path dataset jika diperlukan

### **Langkah 3: Run Enhanced Training**
Jalankan cell kedua yang berisi training script:

```python
# Cell 2 - Enhanced Training
exec(open('colab_enhanced_training.py').read())
```

## 🔧 Alternatif: Copy-Paste Manual

Jika Anda tidak ingin menggunakan `exec()`, copy-paste langsung ke Colab:

### **Cell 1: Setup**
```python
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from pathlib import Path
import warnings
import random
warnings.filterwarnings('ignore')

# Check Colab
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

# Setup
if is_colab():
    print("🚀 Running in Google Colab")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except:
        pass
    
    # Install timm
    try:
        import timm
    except ImportError:
        os.system("pip install timm")
        import timm
    
    os.chdir('/content')
    os.makedirs('src', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data', exist_ok=True)

print(f"🔧 PyTorch: {torch.__version__}")
print(f"💻 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
```

### **Cell 2: Configuration & Training**
```python
# Configuration
class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    EPOCHS = 60
    IMAGE_SIZE = 224
    LEARNING_RATE = 3e-5
    TEST_SPLIT_SIZE = 0.2
    RANDOM_SEED = 42
    MODEL_LIST = ["convnext_tiny"]
    DROPOUT_RATE = 0.7
    WEIGHT_DECAY = 2e-3
    EARLY_STOPPING_PATIENCE = 7
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    LABEL_SMOOTHING = 0.1
    DATA_PATH = "/content/data"  # Update sesuai path dataset Anda

config = Config()

# Update data path jika diperlukan
user_data_path = input("📁 Masukkan path dataset Anda (atau Enter untuk default /content/data): ").strip()
if user_data_path:
    config.DATA_PATH = user_data_path

print(f"📂 Using data path: {config.DATA_PATH}")
```

### **Cell 3: Advanced Augmentation Functions**
```python
# Copy semua fungsi dari colab_enhanced_training.py mulai dari:
# - cutmix_data()
# - mixup_data()
# - LabelSmoothingCrossEntropy class
# - FocalLoss class
# - get_enhanced_transforms()
# - create_dataloaders_colab()
# - dll...
```

### **Cell 4: Training Execution**
```python
# Training execution
def main_colab():
    print("🚀 ENHANCED ANTI-OVERFITTING TRAINING")
    
    # Create data loaders
    train_loader, val_loader, class_names = create_dataloaders_colab()
    
    if train_loader is None:
        return None
    
    # Train model
    results = train_enhanced_model_colab(
        model_name_key="convnext_tiny",
        model_name="convnext_tiny",
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=class_names
    )
    
    return results

# Run training
results = main_colab()
```

## 📁 Struktur File untuk Colab

```
/content/
├── colab_setup.py              # Setup script
├── colab_enhanced_training.py  # Training script (all-in-one)
├── data/                       # Dataset folder
│   └── Batik-Indonesia/        # Your dataset
│       ├── Aceh/
│       ├── Bali/
│       └── ...
└── outputs/                    # Training results
```

## ⚙️ Konfigurasi Penting

### **Update Data Path**
Pastikan update path dataset sesuai struktur Colab:

```python
# Jika dataset di Google Drive
config.DATA_PATH = "/content/drive/MyDrive/Batik-Indonesia"

# Jika dataset diupload langsung ke Colab
config.DATA_PATH = "/content/data/Batik-Indonesia"

# Jika dataset di extract dari zip
config.DATA_PATH = "/content/extracted_dataset/Batik-Indonesia"
```

### **GPU Usage**
Colab akan otomatis menggunakan GPU jika tersedia:

```python
print(f"Device: {config.DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 🎯 Expected Output

Setelah training selesai, Anda akan mendapat:

```
🚀 BATIK VISION - ENHANCED ANTI-OVERFITTING TRAINING (COLAB)
======================================================================

📂 Data path saat ini: /content/data
[Data] Memuat dataset dari: /content/data
[Data] Ditemukan 39 kelas: ['Aceh', 'Bali', 'Bali_Barong', ...]
[Data] Train: 3120 | Val: 780

🚀 TRAINING ENHANCED MODEL: CONVNEXT_TINY
   Model: convnext_tiny
   Classes: 39
   Device: cuda
--------------------------------------------------
📊 Memulai enhanced training 60 epochs...
   Early Stopping: 7 epochs patience
   Learning Rate Scheduler: CosineAnnealingWarmRestarts
   Weight Decay: 0.002 (AdamW)
   Dropout Rate: 0.7
   Loss Function: Combined Label Smoothing + Focal Loss
   Augmentation: Mixup + CutMix + Advanced Transforms

📈 Epoch 1/60
   📊 Train: Loss=2.1234, Acc=0.4567
   📊 Val:   Loss=1.9876, Acc=0.5234
   🏆 Best:  0.5234 (Epoch 1)
   📉 LR:    3.00e-05
   ⏳ No Improve: 0/7
💾 Model terbaik disimpan: /content/convnext_tiny_best_enhanced.pth

...

✅ Enhanced training selesai!
   ⏱️ Waktu: 1847.3 detik
   🏆 Best Accuracy: 0.8756
   📈 Epochs trained: 43

🏆 RINGKASAN HASIL ENHANCED TRAINING
==================================================
convnext_tiny   | Best: 0.8756 | Epochs: 43 | Time: 1847.3s

🥇 Model terbaik: convnext_tiny (0.8756)

✅ Training completed!
```

## 🔧 Troubleshooting

### **Error Import**
```python
# Jika ada error import, install packages:
!pip install timm
!pip install torchvision
!pip install scikit-learn
```

### **CUDA Out of Memory**
```python
# Reduce batch size
config.BATCH_SIZE = 16  # atau 8
```

### **Dataset Not Found**
```python
# Check dataset path
import os
print("Contents of /content/data:", os.listdir("/content/data"))
```

### **Permission Error**
```python
# Fix permissions
!chmod -R 755 /content/data
```

## 📊 Monitoring Progress

Training akan menampilkan progress real-time:
- ✅ Train/Validation loss dan accuracy
- 🏆 Best model tracking
- 📉 Learning rate changes
- ⏳ Early stopping counter
- 💾 Model checkpointing

## 🎯 Expected Improvements

Dibanding training sebelumnya (82.54% val accuracy), enhanced training ini diharapkan mencapai:

- **Validation Accuracy**: 85-90%
- **Training-Validation Gap**: <5%
- **Better Class Balance**: Konsisten di semua kelas
- **Stable Training**: Kurva yang lebih smooth

Selamat training! 🚀
