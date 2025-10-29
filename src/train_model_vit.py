import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings('ignore')

# Import modul yang sudah dibuat
from src import config
from src.data_loader import create_dataloaders
from src.model import create_model
from src.engine import train_step, val_step
from src.advanced_augmentation import (
    calculate_class_weights, get_advanced_scheduler, TestTimeAugmentation
)

# --- Label Smoothing Loss ---
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = pred.log_softmax(dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# --- Enhanced Dropout (lebih ringan) ---
def add_enhanced_dropout_to_model(model, dropout_rate=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'head' in name:
            new_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(module.in_features, module.out_features)
            )
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, name.split('.')[-1], new_head)
            else:
                setattr(model, name.split('.')[-1], new_head)
    return model

# --- Training Step Sederhana (tanpa Mixup/CutMix) ---
def simple_train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred_logits = model(X)
        loss = loss_fn(y_pred_logits, y)
        y_pred_class = torch.argmax(y_pred_logits, dim=1)
        train_acc += (y_pred_class == y).float().mean().item()
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return train_loss / len(dataloader), train_acc / len(dataloader)

# --- Training dengan Fine-Tuning Bertahap ---
def train_enhanced_anti_overfitting_model(model_name_key: str, model_name: str, num_classes: int, 
                                       train_loader, val_loader, writer, model_dir: Path, class_names):
    print(f"\nTRAINING ENHANCED MODEL: {model_name_key.upper()}")
    print(f"   Model: {model_name}")
    print(f"   Classes: {num_classes}")
    print("-" * 50)
    
    model = create_model(model_name, num_classes, pretrained=True)
    if model is None:
        print(f"ERROR: Gagal membuat model {model_name}")
        return None

    model = add_enhanced_dropout_to_model(model, dropout_rate=0.3)
    model = model.to(config.DEVICE)

    # === TAHAP 1: Freeze backbone, latih head saja (5 epoch) ===
    print("   [Tahap 1] Freeze backbone, latih head selama 5 epoch...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    PATIENCE = 8

    for epoch in range(5):
        train_loss, train_acc = simple_train_step(model, train_loader, loss_fn, optimizer, config.DEVICE)
        val_loss, val_acc = val_step(model, val_loader, loss_fn, config.DEVICE)
        scheduler.step()
        print(f"   [Tahap1] Epoch {epoch+1}/5 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # === TAHAP 2: Unfreeze semua, fine-tune penuh ===
    print("   [Tahap 2] Unfreeze semua layer, fine-tune penuh...")
    for param in model.parameters():
        param.requires_grad = True

    # Reset optimizer dengan LR lebih kecil
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()
    for epoch in range(config.EPOCHS - 5):
        train_loss, train_acc = simple_train_step(model, train_loader, loss_fn, optimizer, config.DEVICE)
        val_loss, val_acc = val_step(model, val_loader, loss_fn, config.DEVICE)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        writer.add_scalar(f'{model_name_key}/Train/Loss', train_loss, epoch + 5)
        writer.add_scalar(f'{model_name_key}/Train/Accuracy', train_acc, epoch + 5)
        writer.add_scalar(f'{model_name_key}/Val/Loss', val_loss, epoch + 5)
        writer.add_scalar(f'{model_name_key}/Val/Accuracy', val_acc, epoch + 5)
        writer.add_scalar(f'{model_name_key}/Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 5)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 6
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 6,
                'val_accuracy': val_acc,
            }, model_dir / f"{model_name_key}_best.pth")
            print(f"   Model terbaik disimpan (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1

        print(f"   Epoch {epoch+6}/{config.EPOCHS} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= PATIENCE:
            print(f"   Early stopping di epoch {epoch+6}")
            break

    end_time = time.time()
    print(f"\nTraining selesai! Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")

    # Generate confusion matrix dengan TTA (gunakan mean=0.5, std=0.5)
    generate_enhanced_confusion_matrix(model, val_loader, class_names, model_dir, model_name_key)

    return {
        'model_name': model_name_key,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'training_time': end_time - start_time,
        'epochs_trained': best_epoch,
    }

# --- TTA dengan Preprocessing ViT yang Benar ---
def generate_enhanced_confusion_matrix(model, val_loader, class_names, model_dir, model_name_key):
    model.eval()
    all_preds = []
    all_labels = []
    tta = TestTimeAugmentation(model, config.DEVICE, num_augmentations=3)  # kurangi jadi 3 agar lebih cepat

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            for i in range(X.size(0)):
                img_tensor = X[i]
                # Denormalisasi sesuai ViT: (x - 0.5)/0.5 → x = img * 0.5 + 0.5
                img_tensor = img_tensor * 0.5 + 0.5
                img_tensor = torch.clamp(img_tensor, 0, 1)
                from torchvision.transforms import ToPILImage
                img_pil = ToPILImage()(img_tensor.cpu())
                tta_pred = tta.predict(img_pil)
                _, pred = torch.max(tta_pred, 1)
                all_preds.append(pred.item())
                all_labels.append(y[i].cpu().item())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name_key.upper()} (TTA)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(model_dir / f"{model_name_key}_enhanced_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open(model_dir / f"{model_name_key}_enhanced_classification_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"   Confusion matrix & report disimpan.")

# --- Setup & Main ---
def setup_enhanced_anti_overfitting_training():
    config.BATCH_SIZE = 32
    config.EPOCHS = 50
    config.IMAGE_SIZE = 224
    config.LEARNING_RATE = 3e-5
    print("SETUP ENHANCED TRAINING (Optimized for High Accuracy)")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("outputs") / f"high_acc_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "models").mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=experiment_dir / "logs")
    return writer, experiment_dir, experiment_dir / "models"

def main():
    print("BATIK VISION - HIGH-ACCURACY TRAINING MODE")
    print("="*60)
    
    writer, experiment_dir, model_dir = setup_enhanced_anti_overfitting_training()
    
    print("\nMembuat data loaders...")
    train_loader, val_loader, class_names = create_dataloaders()
    num_classes = len(class_names)
    print(f"Data siap! {num_classes} kelas.")

    model_mapping = {"vit": "vit_base_patch16_224"}
    all_results = []

    for model_name_key in config.MODEL_LIST:
        if model_name_key not in model_mapping:
            continue
        result = train_enhanced_anti_overfitting_model(
            model_name_key, model_mapping[model_name_key], num_classes,
            train_loader, val_loader, writer, model_dir, class_names
        )
        if result:
            all_results.append(result)

    if all_results:
        best = max(all_results, key=lambda x: x['best_val_acc'])
        print(f"\n✅ Model terbaik: {best['model_name']} | Akurasi: {best['best_val_acc']:.4f}")

    writer.close()
    print(f"\nHasil disimpan di: {experiment_dir}")

if __name__ == "__main__":
    main()