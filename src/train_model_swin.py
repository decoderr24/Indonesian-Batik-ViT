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

# Import modul lokal
from src import config
from src.data_loader import create_dataloaders
from src.model import create_model
from src.engine import val_step
from src.advanced_augmentation import TestTimeAugmentation

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

# --- Dropout Ringan ---
def add_enhanced_dropout_to_model(model, dropout_rate=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ('head' in name or 'classifier' in name):
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

# --- Training Step Sederhana ---
def simple_train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    num_batches = len(dataloader)
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred_logits = model(X)
        loss = loss_fn(y_pred_logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Akurasi
        y_pred_class = torch.argmax(y_pred_logits, dim=1)
        acc = (y_pred_class == y).float().mean().item()

        train_loss += loss.item()
        train_acc += acc

        if batch_idx % 10 == 0:
            print(f"    Batch {batch_idx}/{num_batches} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

    return train_loss / num_batches, train_acc / num_batches

# --- Training Utama ---
def train_enhanced_anti_overfitting_model(model_name_key: str, model_name: str, num_classes: int,
                                       train_loader, val_loader, writer, model_dir: Path, class_names):
    print(f"\n🚀 TRAINING MODEL: {model_name_key.upper()}")
    print(f"   Model: {model_name}")
    print(f"   Classes: {num_classes}")
    print("-" * 50)

    model = create_model(model_name, num_classes, pretrained=True)
    if model is None:
        print(f"❌ ERROR: Gagal membuat model {model_name}")
        return None

    # Sesuaikan dropout & LR berdasarkan model
    if "swin" in model_name_key:
        model = add_enhanced_dropout_to_model(model, dropout_rate=0.2)
        LEARNING_RATE = 5e-5
        print("   [Info] Model Swin: dropout=0.2, LR=5e-5")
    else:
        model = add_enhanced_dropout_to_model(model, dropout_rate=0.3)
        LEARNING_RATE = 3e-5
        print("   [Info] Model ViT: dropout=0.3, LR=3e-5")

    model = model.to(config.DEVICE)
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)

    # === TAHAP 1: Freeze backbone (5 epoch) ===
    print("\n[🔄 Tahap 1] Freeze backbone, latih head saja (5 epoch)...")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

    for epoch in range(5):
        train_loss, train_acc = simple_train_step(model, train_loader, loss_fn, optimizer, config.DEVICE)
        val_loss, val_acc = val_step(model, val_loader, loss_fn, config.DEVICE)
        scheduler.step()
        print(f"   [Tahap1] Epoch {epoch+1}/5 | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # === TAHAP 2: Unfreeze semua ===
    print("\n[🔄 Tahap 2] Unfreeze semua layer, fine-tune penuh...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    PATIENCE = 8
    total_epochs = config.EPOCHS - 5

    start_time = time.time()
    for epoch in range(total_epochs):
        print(f"\n[Epoch {epoch+6}/{config.EPOCHS}]")
        train_loss, train_acc = simple_train_step(model, train_loader, loss_fn, optimizer, config.DEVICE)
        val_loss, val_acc = val_step(model, val_loader, loss_fn, config.DEVICE)
        scheduler.step()

        # Logging TensorBoard
        writer.add_scalar(f'{model_name_key}/Train/Loss', train_loss, epoch + 5)
        writer.add_scalar(f'{model_name_key}/Train/Acc', train_acc, epoch + 5)
        writer.add_scalar(f'{model_name_key}/Val/Loss', val_loss, epoch + 5)
        writer.add_scalar(f'{model_name_key}/Val/Acc', val_acc, epoch + 5)
        writer.add_scalar(f'{model_name_key}/LR', optimizer.param_groups[0]['lr'], epoch + 5)

        # Simpan model terbaik
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 6
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch,
                'val_accuracy': val_acc,
            }, model_dir / f"{model_name_key}_best.pth")
            print(f"   🏆 Model terbaik disimpan! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1

        # Print lengkap per epoch
        print(f"   ➤ Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   ➤ Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"   ➤ LR: {optimizer.param_groups[0]['lr']:.2e} | Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")

        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early stopping di epoch {epoch+6} (tidak ada peningkatan selama {PATIENCE} epoch)")
            break

    end_time = time.time()
    print(f"\n✅ Training selesai! Waktu: {end_time - start_time:.1f} detik")
    print(f"   🥇 Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")

    # Generate confusion matrix
    generate_enhanced_confusion_matrix(model, val_loader, class_names, model_dir, model_name_key)

    return {
        'model_name': model_name_key,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'training_time': end_time - start_time,
    }

# --- TTA dengan Preprocessing Benar ---
def generate_enhanced_confusion_matrix(model, val_loader, class_names, model_dir, model_name_key):
    model.eval()
    all_preds, all_labels = [], []
    tta = TestTimeAugmentation(model, config.DEVICE, num_augmentations=3)

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            for i in range(X.size(0)):
                img = X[i] * 0.5 + 0.5  # denormalisasi ViT/Swin
                img = torch.clamp(img, 0, 1)
                from torchvision.transforms import ToPILImage
                pil_img = ToPILImage()(img.cpu())
                pred = tta.predict(pil_img)
                _, cls = torch.max(pred, 1)
                all_preds.append(cls.item())
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
    plt.savefig(model_dir / f"{model_name_key}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open(model_dir / f"{model_name_key}_classification_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"   📊 Confusion matrix & classification report disimpan.")

# --- Setup & Main ---
def setup_enhanced_anti_overfitting_training():
    config.EPOCHS = 50
    config.BATCH_SIZE = 32
    config.IMAGE_SIZE = 224
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path("outputs") / f"swin_vit_comparison_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=exp_dir / "logs")
    return writer, exp_dir, exp_dir / "models"

def main():
    print("🎯 BATIK VISION - SWIN vs ViT TRAINING")
    print("="*60)

    writer, exp_dir, model_dir = setup_enhanced_anti_overfitting_training()
    print("\n📂 Memuat data...")
    train_loader, val_loader, class_names = create_dataloaders()
    num_classes = len(class_names)
    print(f"✅ Data siap! {num_classes} kelas, {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")

    # Mapping model
    model_mapping = {
        "vit": "vit_base_patch16_224",
        "swin": "swin_base_patch4_window7_224"
    }

    results = []
    for model_key in config.MODEL_LIST:
        if model_key not in model_mapping:
            print(f"⚠️ Model '{model_key}' tidak didukung. Lewati.")
            continue
        result = train_enhanced_anti_overfitting_model(
            model_key, model_mapping[model_key], num_classes,
            train_loader, val_loader, writer, model_dir, class_names
        )
        if result:
            results.append(result)

    # Ringkasan
    if results:
        print("\n" + "="*60)
        print("📊 RINGKASAN AKHIR")
        for r in results:
            print(f"{r['model_name']:6} | Best Val Acc: {r['best_val_acc']:.4f} | Epoch: {r['best_epoch']}")
        best = max(results, key=lambda x: x['best_val_acc'])
        print(f"\n🏆 Model terbaik: {best['model_name']} ({best['best_val_acc']:.4f})")
    else:
        print("❌ Tidak ada model yang berhasil dilatih.")

    writer.close()
    print(f"\n📁 Hasil disimpan di: {exp_dir}")

if __name__ == "__main__":
    main()