"""
Google Colab Enhanced Training Script
===================================

Script terpadu untuk menjalankan enhanced anti-overfitting training di Google Colab.
Menggabungkan semua modul yang diperlukan dalam satu file untuk kemudahan penggunaan.
"""

import os
import sys
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
import warnings
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from pathlib import Path
import random
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for enhanced training"""
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 60
    IMAGE_SIZE = 224
    LEARNING_RATE = 3e-5
    
    # Data split
    TEST_SPLIT_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Model
    MODEL_LIST = ["convnext_tiny"]
    
    # Enhanced parameters
    DROPOUT_RATE = 0.7
    WEIGHT_DECAY = 2e-3
    EARLY_STOPPING_PATIENCE = 7
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    LABEL_SMOOTHING = 0.1
    
    # Data path - akan disesuaikan di Colab
    DATA_PATH = "/content/data"  # Default untuk Colab

config = Config()

# ============================================================================
# ADVANCED AUGMENTATION MODULES
# ============================================================================

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    """CutMix data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device == 'cuda' and torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    W = x.size(2)
    H = x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """CutMix loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if device == 'cuda' and torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ============================================================================
# DATA LOADING
# ============================================================================

def get_enhanced_transforms():
    """Get enhanced augmentation transforms"""
    # Training transforms
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomCrop((config.IMAGE_SIZE, config.IMAGE_SIZE), padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class TransformedDataset(Dataset):
    """Wrapper Dataset untuk menerapkan transformasi ke Subset"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        try:
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        except Exception as e:
            print(f"[Warning] Error pada index {index}: {e}")
            next_index = (index + 1) % len(self.subset)
            return self.__getitem__(next_index)
    
    def __len__(self):
        return len(self.subset)

def create_dataloaders_colab():
    """Create data loaders for Colab"""
    print(f"[Data] Memuat dataset dari: {config.DATA_PATH}")
    
    # Check if data path exists
    if not os.path.exists(config.DATA_PATH):
        print(f"❌ Data path tidak ditemukan: {config.DATA_PATH}")
        print("📋 Pastikan Anda telah:")
        print("   1. Upload dataset ke Colab")
        print("   2. Extract dataset jika dalam format zip")
        print("   3. Update config.DATA_PATH dengan path yang benar")
        return None, None, None
    
    # Load dataset
    full_dataset = datasets.ImageFolder(config.DATA_PATH)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"[Data] Ditemukan {num_classes} kelas: {class_names}")
    
    # Split dataset
    total_size = len(full_dataset)
    val_size = int(total_size * config.TEST_SPLIT_SIZE)
    train_size = total_size - val_size
    
    train_dataset_raw, val_dataset_raw = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    print(f"[Data] Train: {len(train_dataset_raw)} | Val: {len(val_dataset_raw)}")
    
    # Apply transforms
    train_transform, val_transform = get_enhanced_transforms()
    train_dataset = TransformedDataset(train_dataset_raw, transform=train_transform)
    val_dataset = TransformedDataset(val_dataset_raw, transform=val_transform)
    
    # Handle class imbalance
    train_targets = [full_dataset.targets[i] for i in train_dataset_raw.indices]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_targets]
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )
    
    print("[Data] Data loaders berhasil dibuat")
    return train_loader, val_loader, class_names

# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model_colab(model_name, num_classes, pretrained=True):
    """Create model for Colab"""
    try:
        import timm
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        print(f"✅ Model {model_name} berhasil dibuat")
        return model
    except Exception as e:
        print(f"❌ Error creating model {model_name}: {e}")
        return None

def add_enhanced_dropout_to_model(model, dropout_rate=0.7):
    """Add enhanced dropout to model"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'head' in name:
            # Find parent module
            parts = name.split('.')
            current = model
            for part in parts[:-1]:
                current = getattr(current, part)
            old_linear = getattr(current, parts[-1])
            new_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(old_linear.in_features, old_linear.out_features)
            )
            setattr(current, parts[-1], new_head)
    return model

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def apply_mixup_cutmix_probability():
    """Randomly choose between Mixup and CutMix"""
    return random.choice(['mixup', 'cutmix', 'none'])

def enhanced_train_step(model, dataloader, loss_fn, optimizer, device, 
                       use_mixup=True, use_cutmix=True, mixup_alpha=0.2, cutmix_alpha=1.0):
    """Enhanced training step dengan Mixup dan CutMix"""
    model.train()
    train_loss, train_acc = 0, 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        augmentation_type = apply_mixup_cutmix_probability()
        
        if augmentation_type == 'mixup' and use_mixup:
            mixed_x, y_a, y_b, lam = mixup_data(X, y, mixup_alpha, device)
            y_pred_logits = model(mixed_x)
            loss = mixup_criterion(loss_fn, y_pred_logits, y_a, y_b, lam)
            
            _, predicted = torch.max(y_pred_logits, 1)
            train_acc += (lam * (predicted == y_a).float() + 
                         (1 - lam) * (predicted == y_b).float()).mean().item()
            
        elif augmentation_type == 'cutmix' and use_cutmix:
            mixed_x, y_a, y_b, lam = cutmix_data(X, y, cutmix_alpha, device)
            y_pred_logits = model(mixed_x)
            loss = cutmix_criterion(loss_fn, y_pred_logits, y_a, y_b, lam)
            
            _, predicted = torch.max(y_pred_logits, 1)
            train_acc += (lam * (predicted == y_a).float() + 
                         (1 - lam) * (predicted == y_b).float()).mean().item()
            
        else:
            y_pred_logits = model(X)
            loss = loss_fn(y_pred_logits, y)
            
            y_pred_class = torch.argmax(y_pred_logits, dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred_logits)
        
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

def val_step_colab(model, dataloader, loss_fn, device):
    """Validation step"""
    model.eval()
    val_loss, val_acc = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            y_pred_logits = model(X)
            loss = loss_fn(y_pred_logits, y)
            
            y_pred_class = torch.argmax(y_pred_logits, dim=1)
            val_acc += (y_pred_class == y).sum().item() / len(y_pred_logits)
            
            val_loss += loss.item()
    
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    
    return val_loss, val_acc

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_enhanced_model_colab(model_name_key="convnext_tiny", model_name="convnext_tiny", 
                              train_loader=None, val_loader=None, class_names=None):
    """Main training function for Colab"""
    
    if train_loader is None or val_loader is None:
        print("❌ Data loaders tidak tersedia")
        return None
    
    num_classes = len(class_names)
    print(f"\n🚀 TRAINING ENHANCED MODEL: {model_name_key.upper()}")
    print(f"   Model: {model_name}")
    print(f"   Classes: {num_classes}")
    print(f"   Device: {config.DEVICE}")
    print("-" * 50)
    
    # Create model
    model = create_model_colab(model_name, num_classes, pretrained=True)
    if model is None:
        return None
    
    # Add enhanced dropout
    model = add_enhanced_dropout_to_model(model, dropout_rate=config.DROPOUT_RATE)
    model = model.to(config.DEVICE)
    
    # Setup loss function
    label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=config.LABEL_SMOOTHING)
    focal_loss = FocalLoss(alpha=1, gamma=2)
    
    def combined_loss(pred, target):
        return 0.7 * label_smooth_loss(pred, target) + 0.3 * focal_loss(pred, target)
    
    loss_fn = combined_loss
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Setup scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    
    # Tracking variables
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    
    print(f"📊 Memulai enhanced training {config.EPOCHS} epochs...")
    print(f"   Early Stopping: {config.EARLY_STOPPING_PATIENCE} epochs patience")
    print(f"   Learning Rate Scheduler: CosineAnnealingWarmRestarts")
    print(f"   Weight Decay: {config.WEIGHT_DECAY} (AdamW)")
    print(f"   Dropout Rate: {config.DROPOUT_RATE}")
    print(f"   Loss Function: Combined Label Smoothing + Focal Loss")
    print(f"   Augmentation: Mixup + CutMix + Advanced Transforms")
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        print(f"\n📈 Epoch {epoch+1}/{config.EPOCHS}")
        
        # Training
        train_loss, train_acc = enhanced_train_step(
            model=model, dataloader=train_loader, loss_fn=loss_fn,
            optimizer=optimizer, device=config.DEVICE,
            use_mixup=True, use_cutmix=True, 
            mixup_alpha=config.MIXUP_ALPHA, cutmix_alpha=config.CUTMIX_ALPHA
        )
        
        # Validation
        val_loss, val_acc = val_step_colab(
            model=model, dataloader=val_loader, loss_fn=loss_fn,
            device=config.DEVICE
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Check best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            
            # Save best model
            model_path = f"/content/convnext_tiny_best_enhanced.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'val_accuracy': val_acc,
                'model_name': model_name,
                'num_classes': num_classes
            }, model_path)
            print(f"💾 Model terbaik disimpan: {model_path}")
        else:
            epochs_no_improve += 1
        
        # Progress
        print(f"   📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"   📊 Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"   🏆 Best:  {best_val_acc:.4f} (Epoch {best_epoch})")
        print(f"   📉 LR:    {optimizer.param_groups[0]['lr']:.2e}")
        print(f"   ⏳ No Improve: {epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE}")
        
        # Early stopping
        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n⏹️ Early stopping! Tidak ada kemajuan selama {config.EARLY_STOPPING_PATIENCE} epoch.")
            print(f"🏆 Model terbaik: Epoch {best_epoch} dengan Val Acc: {best_val_acc:.4f}")
            break
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ Enhanced training selesai!")
    print(f"   ⏱️ Waktu: {training_time:.1f} detik")
    print(f"   🏆 Best Accuracy: {best_val_acc:.4f}")
    print(f"   📈 Epochs trained: {epoch + 1}")
    
    return {
        'model': model,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'epochs_trained': epoch + 1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main_colab():
    """Main function untuk Colab"""
    print("🚀 BATIK VISION - ENHANCED ANTI-OVERFITTING TRAINING (COLAB)")
    print("="*70)
    
    # Setup data path - user harus update ini
    print(f"📂 Data path saat ini: {config.DATA_PATH}")
    print("💡 Jika data path salah, update config.DATA_PATH dengan path yang benar")
    
    # Create data loaders
    print("\n📊 Membuat data loaders...")
    train_loader, val_loader, class_names = create_dataloaders_colab()
    
    if train_loader is None:
        print("❌ Gagal membuat data loaders. Pastikan dataset sudah diupload dengan benar.")
        return None
    
    num_classes = len(class_names)
    print(f"✅ Data siap! {num_classes} kelas ditemukan.")
    print(f"   Kelas: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")
    
    # Model mapping
    model_mapping = {
        "convnext_tiny": "convnext_tiny",
        "vit": "vit_base_patch16_224",
        "swin_transformer": "swin_base_patch4_window7_224"
    }
    
    # Train models
    results = []
    
    for model_name_key in config.MODEL_LIST:
        if model_name_key not in model_mapping:
            print(f"⚠️ Model '{model_name_key}' tidak dikenali. Dilewati.")
            continue
        
        model_name = model_mapping[model_name_key]
        
        try:
            result = train_enhanced_model_colab(
                model_name_key=model_name_key,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=class_names
            )
            
            if result:
                results.append({
                    'model_name': model_name_key,
                    'best_val_acc': result['best_val_acc'],
                    'best_epoch': result['best_epoch'],
                    'training_time': result['training_time'],
                    'epochs_trained': result['epochs_trained']
                })
                
        except Exception as e:
            print(f"❌ ERROR training {model_name_key}: {e}")
            continue
    
    # Summary
    if results:
        print(f"\n🏆 RINGKASAN HASIL ENHANCED TRAINING")
        print("="*50)
        
        for result in results:
            print(f"{result['model_name']:15} | "
                  f"Best: {result['best_val_acc']:.4f} | "
                  f"Epochs: {result['epochs_trained']} | "
                  f"Time: {result['training_time']:.1f}s")
        
        best_model = max(results, key=lambda x: x['best_val_acc'])
        print(f"\n🥇 Model terbaik: {best_model['model_name']} "
              f"({best_model['best_val_acc']:.4f})")
    
    print(f"\n✅ Training completed!")
    return results

# ============================================================================
# COLAB EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Install required packages untuk Colab
    try:
        import timm
    except ImportError:
        print("📦 Installing timm...")
        os.system("pip install timm")
        import timm
    
    try:
        import google.colab
        print("🔧 Running in Google Colab")
        
        # Update data path sesuai kebutuhan user
        user_data_path = input("📁 Masukkan path dataset Anda (atau tekan Enter untuk default): ").strip()
        if user_data_path:
            config.DATA_PATH = user_data_path
        
        results = main_colab()
        
    except ImportError:
        print("🖥️ Running locally")
        results = main_colab()
