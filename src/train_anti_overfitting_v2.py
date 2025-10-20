"""
train_anti_overfitting_v2.py

Versi upgrade dari `train_anti_overfitting.py`:
- MixUp & CutMix augmentation (opsional, diaktifkan via flag)
- Label smoothing pada CrossEntropyLoss
- Dropout ditambahkan ke classifier head dan block terakhir (jika tersedia)
- Gradient clipping
- CosineAnnealingWarmRestarts scheduler (default) + optional ReduceLROnPlateau
- Class-weighting support (opsional, dihitung dari train labels jika tersedia)
- Freeze backbone untuk N epoch pertama (fine-tune strategy)
- Menyimpan plot loss/accuracy otomatis dan classification report + confusion matrix

Catatan: script ini mengasumsikan struktur proyek yang sama (src.config, src.data_loader, src.model, src.engine).
Jalankan dari root project (sama seperti script lama).
"""

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
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

from src import config
from src.data_loader import create_dataloaders
from src.model import create_model
from src.engine import train_step, val_step

# --------------------------- Augmentation utilities ---------------------------

def mixup_data(x, y, alpha=0.4, device='cpu'):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0, device='cpu'):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(device)

    # sample bounding box
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]
    # adjust lambda to actual area
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y_a, y_b, lam

# --------------------------- Model modification utilities ---------------------------

def add_dropout_to_head(model, dropout_rate=0.5):
    """Tambahkan dropout tepat sebelum classifier head (Linear) dengan pendekatan aman."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'head' in name:
            parent_name = '.'.join(name.split('.')[:-1])
            attr = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            linear = getattr(parent, attr)
            seq = nn.Sequential(nn.Dropout(dropout_rate), linear)
            setattr(parent, attr, seq)
    return model


def add_dropout_to_last_block(model, dropout_rate=0.3):
    """Coba tambahkan dropout ke block akhir dari backbone jika attribute dikenali.
    Implementasi ini aman-check untuk beberapa arsitektur (convnext, timm models).
    """
    # ConvNeXt-like: stages / blocks
    try:
        if hasattr(model, 'stages'):
            last_stage = model.stages[-1]
            # Jika last_stage adalah Sequential of blocks
            if isinstance(last_stage, (nn.Sequential, list, tuple)):
                for i, block in enumerate(last_stage):
                    # tambahkan dropout ke dalam block jika memungkinkan
                    if isinstance(block, nn.Module):
                        block.add_module('drop_extra', nn.Dropout(p=dropout_rate))
                        break  # tambahkan hanya ke block pertama di last stage agar aman
        # Swin/ViT style: add dropout before head
        if hasattr(model, 'patch_embed') and hasattr(model, 'norm'):
            # tambahkan dropout setelah norm
            model.add_module('backbone_dropout', nn.Dropout(p=dropout_rate))
    except Exception:
        # Jika gagal, jangan crash
        pass
    return model

# --------------------------- Training utilities ---------------------------

def apply_gradient_clipping(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def save_plots(train_losses, val_losses, train_accs, val_accs, out_dir, model_name_key):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name_key}_loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name_key}_acc_curve.png", dpi=300)
    plt.close()

# --------------------------- Main training function ---------------------------

def train_anti_overfitting_model_v2(model_name_key: str, model_name: str, num_classes: int,
                                    train_loader, val_loader, writer, model_dir: Path, class_names,
                                    config_overrides=None):
    """Versi v2: integrasikan MixUp/CutMix, label smoothing, gradient clipping, scheduler CosineWarm.
    config_overrides: dict optional keys:
        - mixup_alpha, cutmix_alpha, use_mixup, use_cutmix
        - dropout_head, dropout_backbone
        - label_smoothing
        - freeze_backbone_epochs
        - use_reduce_on_plateau (bool)
        - max_grad_norm
    """
    co = config_overrides or {}
    use_mixup = co.get('use_mixup', True)
    use_cutmix = co.get('use_cutmix', False)
    mixup_alpha = co.get('mixup_alpha', 0.4)
    cutmix_alpha = co.get('cutmix_alpha', 1.0)
    dropout_head = co.get('dropout_head', 0.6)
    dropout_backbone = co.get('dropout_backbone', 0.3)
    label_smoothing = co.get('label_smoothing', 0.1)
    freeze_backbone_epochs = co.get('freeze_backbone_epochs', 5)
    use_reduce_on_plateau = co.get('use_reduce_on_plateau', False)
    max_grad_norm = co.get('max_grad_norm', 1.0)

    print(f"\nTRAINING MODEL (v2): {model_name_key.upper()}")
    print(f"   Model: {model_name}")
    print(f"   Classes: {num_classes}")
    print("-"*50)

    # 1) create model
    model = create_model(model_name, num_classes, pretrained=True)
    if model is None:
        print(f"ERROR: Gagal membuat model {model_name}")
        return None

    # 2) add dropout to head + last block
    model = add_dropout_to_head(model, dropout_head)
    model = add_dropout_to_last_block(model, dropout_backbone)

    # 3) move to device
    model = model.to(config.DEVICE)

    # 4) optionally freeze backbone for few epochs
    backbone_params = [p for n, p in model.named_parameters() if 'head' not in n and p.requires_grad]
    def set_backbone_requires_grad(flag):
        for n, p in model.named_parameters():
            if 'head' not in n:
                p.requires_grad = flag

    # 5) Loss function with label smoothing
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 6) Optional: compute class weights from train_loader labels
    try:
        y_train = []
        for _, y in train_loader.dataset:  # assumes dataset returns (x, y)
            y_train.append(int(y))
        class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train)
        weights = torch.FloatTensor(class_weights).to(config.DEVICE)
        weighted_loss = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        loss_fn = weighted_loss
        print("   Class weights applied to loss function.")
    except Exception:
        # jika gagal hitung, lanjut tanpa class weights
        pass

    # 7) Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=1e-3)

    # 8) Scheduler: CosineAnnealingWarmRestarts (default) + optional ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)
    if use_reduce_on_plateau:
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3, min_lr=1e-7)
    else:
        plateau = None

    # tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    patience = 10  # sedikit lebih longgar pada v2
    epochs_no_improve = 0

    print(f"Memulai training {config.EPOCHS} epochs...")
    print(f"   Freeze backbone epochs: {freeze_backbone_epochs}")
    print(f"   MixUp: {use_mixup}, CutMix: {use_cutmix}")
    print(f"   Label smoothing: {label_smoothing}")

    start_time = time.time()
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

        # unfreeze if passed freeze_backbone_epochs
        if epoch == freeze_backbone_epochs:
            set_backbone_requires_grad(True)
            # re-init optimizer to include newly trainable params
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=1e-3)
            # reattach scheduler state if needed (simple approach: recreate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)
            if plateau is not None:
                plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=3, min_lr=1e-7)
            print("   Backbone unfrozen and optimizer reinitialized.")

        # TRAIN LOOP (with MixUp/CutMix applied per-batch inside train_step wrapper)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs, targets = batch
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            # Apply MixUp or CutMix randomly
            applied_mix = False
            if use_mixup and np.random.rand() < 0.5:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha, device=config.DEVICE)
                applied_mix = 'mixup'
            elif use_cutmix and np.random.rand() < 0.5:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, cutmix_alpha, device=config.DEVICE)
                applied_mix = 'cutmix'

            optimizer.zero_grad()
            outputs = model(inputs)

            if applied_mix:
                loss = lam * loss_fn(outputs, targets_a) + (1 - lam) * loss_fn(outputs, targets_b)
            else:
                loss = loss_fn(outputs, targets)

            loss.backward()
            # gradient clipping
            if max_grad_norm:
                apply_gradient_clipping(model, max_grad_norm)
            optimizer.step()

            # stats (for accuracy, if mixup applied we approximate by taking max against targets_a)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            if applied_mix:
                # count prediction correct if matches either target (loose estimation)
                correct += (predicted.eq(targets_a).sum().item() + predicted.eq(targets_b).sum().item()) / 2.0
            else:
                correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # VALIDATION
        val_loss, val_acc = val_step(model=model, dataloader=val_loader, loss_fn=loss_fn, device=config.DEVICE)

        # scheduler step
        # CosineWarm uses epoch-based step via scheduler.step(epoch + epoch_fraction) using optimizer state
        scheduler.step()
        if plateau is not None:
            plateau.step(val_acc)

        # store
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # TensorBoard
        writer.add_scalar(f'{model_name_key}/Train/Loss', train_loss, epoch)
        writer.add_scalar(f'{model_name_key}/Train/Accuracy', train_acc, epoch)
        writer.add_scalar(f'{model_name_key}/Val/Loss', val_loss, epoch)
        writer.add_scalar(f'{model_name_key}/Val/Accuracy', val_acc, epoch)
        writer.add_scalar(f'{model_name_key}/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # best model check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            model_path = model_dir / f"{model_name_key}_best.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'val_accuracy': val_acc,
                'model_name': model_name,
                'num_classes': num_classes
            }, model_path)
            print(f"Model terbaik disimpan: {model_path}")
        else:
            epochs_no_improve += 1

        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"   Best:  {best_val_acc:.4f} (Epoch {best_epoch})")
        print(f"   LR:    {optimizer.param_groups[0]['lr']:.2e}")
        print(f"   No Improve: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping! Tidak ada kemajuan selama {patience} epoch.")
            print(f"Model terbaik: Epoch {best_epoch} dengan Val Acc: {best_val_acc:.4f}")
            break

    end_time = time.time()
    training_time = end_time - start_time

    print(f"\nTraining selesai!")
    print(f"   Waktu: {training_time:.1f} detik")
    print(f"   Best Accuracy: {best_val_acc:.4f}")
    print(f"   Epochs trained: {epoch + 1}")

    # save plots
    save_plots(train_losses, val_losses, train_accs, val_accs, model_dir, model_name_key)

    # generate confusion matrix + classification report
    print(f"\nGenerating Confusion Matrix dan Classification Report...")
    generate_confusion_matrix(model, val_loader, class_names, model_dir, model_name_key)

    return {
        'model_name': model_name_key,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_val_acc': val_acc,
        'training_time': training_time,
        'epochs_trained': epoch + 1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

# reuse generate_confusion_matrix dari versi awal (disalin untuk independensi)

def generate_confusion_matrix(model, val_loader, class_names, model_dir, model_name_key):
    model.eval()
    all_preds = []
    all_labels = []

    print("   Mengumpulkan prediksi untuk confusion matrix...")
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name_key.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    cm_path = model_dir / f"{model_name_key}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    report = classification_report(all_labels, all_preds,
                                 target_names=class_names,
                                 output_dict=True)

    report_path = model_dir / f"{model_name_key}_classification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"   Confusion Matrix disimpan: {cm_path}")
    print(f"   Classification Report disimpan: {report_path}")

    print(f"\n   Per-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        if class_name in report:
            acc = report[class_name]['f1-score']
            print(f"   {class_name:25}: {acc:.4f}")

# --------------------------- main ---------------------------

def setup_anti_overfitting_training_v2():
    print("SETUP TRAINING ANTI-OVERFITTING - AGGRESSIVE (v2)")
    print("="*60)

    # override minimal config
    config.BATCH_SIZE = getattr(config, 'BATCH_SIZE', 32)
    config.EPOCHS = getattr(config, 'EPOCHS', 50)
    config.IMAGE_SIZE = getattr(config, 'IMAGE_SIZE', 224)
    config.LEARNING_RATE = getattr(config, 'LEARNING_RATE', 5e-5)

    print(f"Konfigurasi (v2): BATCH={config.BATCH_SIZE}, EPOCHS={config.EPOCHS}, IMG={config.IMAGE_SIZE}, LR={config.LEARNING_RATE}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("outputs") / f"anti_overfitting_v2_{timestamp}"
    model_dir = experiment_dir / "models"
    log_dir = experiment_dir / "logs"

    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    return writer, experiment_dir, model_dir


def main():
    print("BATIK VISION - ANTI-OVERFITTING TRAINING MODE (v2)")
    print("="*60)

    writer, experiment_dir, model_dir = setup_anti_overfitting_training_v2()

    print("\nMembuat data loaders...")
    try:
        train_loader, val_loader, class_names = create_dataloaders()
        num_classes = len(class_names)
        print(f"Data siap! {num_classes} kelas ditemukan.")
    except Exception as e:
        print(f"ERROR data loader: {e}")
        return

    model_mapping = {
        "vit": "vit_base_patch16_224",
        "swin_transformer": "swin_base_patch4_window7_224",
        "convnext_tiny": "convnext_tiny"
    }

    all_results = []

    # Default overrides (kamu bisa ubah sesuai kebutuhan)
    overrides = {
        'use_mixup': True,
        'use_cutmix': False,
        'mixup_alpha': 0.4,
        'cutmix_alpha': 1.0,
        'dropout_head': 0.6,
        'dropout_backbone': 0.3,
        'label_smoothing': 0.1,
        'freeze_backbone_epochs': 5,
        'use_reduce_on_plateau': False,
        'max_grad_norm': 1.0
    }

    for model_name_key in config.MODEL_LIST:
        if model_name_key not in model_mapping:
            print(f"WARNING: Model '{model_name_key}' tidak dikenali. Dilewati.")
            continue
        model_name = model_mapping[model_name_key]
        try:
            result = train_anti_overfitting_model_v2(
                model_name_key=model_name_key,
                model_name=model_name,
                num_classes=num_classes,
                train_loader=train_loader,
                val_loader=val_loader,
                writer=writer,
                model_dir=model_dir,
                class_names=class_names,
                config_overrides=overrides
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"ERROR training {model_name_key}: {e}")
            continue

    if all_results:
        print(f"\nRINGKASAN HASIL")
        print("="*40)
        for result in all_results:
            print(f"{result['model_name']:15} | Best: {result['best_val_acc']:.4f} | Epochs: {result['epochs_trained']} | Time: {result['training_time']:.1f}s")
        best_model = max(all_results, key=lambda x: x['best_val_acc'])
        print(f"\nModel terbaik: {best_model['model_name']} ({best_model['best_val_acc']:.4f})")

    writer.close()
    print(f"\nHasil disimpan di: {experiment_dir}")

if __name__ == '__main__':
    main()
