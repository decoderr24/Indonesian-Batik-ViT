import sys
from pathlib import Path
# Tambahkan parent project ke sys.path sehingga 'src' dapat diimport saat menjalankan skrip langsung
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

# Import modul yang sudah dibuat
from src import config
from src.data_loader import create_dataloaders
from src.model import create_model
from src.engine import train_step, val_step

def setup_anti_overfitting_training():
    """
    Setup untuk training anti-overfitting yang sangat agresif.
    """
    print("SETUP TRAINING ANTI-OVERFITTING - AGGRESSIVE")
    print("="*60)
    
    # Override config untuk training anti-overfitting
    config.BATCH_SIZE = 32  # Batch size lebih besar untuk stabilisasi
    config.EPOCHS = 50      # Lebih banyak epoch dengan early stopping
    config.IMAGE_SIZE = 224  # Resolusi standar
    config.LEARNING_RATE = 5e-5  # Learning rate lebih kecil
    
    print(f"Konfigurasi Anti-Overfitting:")
    print(f"   - Batch Size: {config.BATCH_SIZE}")
    print(f"   - Epochs: {config.EPOCHS}")
    print(f"   - Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"   - Learning Rate: {config.LEARNING_RATE}")
    print(f"   - Device: {config.DEVICE}")
    print(f"   - Model: {config.MODEL_LIST[0] if config.MODEL_LIST else 'None'}")
    
    # Buat direktori untuk hasil
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("outputs") / f"anti_overfitting_{timestamp}"
    model_dir = experiment_dir / "models"
    log_dir = experiment_dir / "logs"
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_dir))
    
    return writer, experiment_dir, model_dir

def add_dropout_to_model(model, dropout_rate=0.5):
    """
    Menambahkan dropout layers ke model untuk mengurangi overfitting.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'head' in name:
            # Tambahkan dropout sebelum classifier head
            new_head = nn.Sequential(
                nn.Dropout(dropout_rate),
                module
            )
            # Ganti head dengan dropout
            parent_name = '.'.join(name.split('.')[:-1])
            if parent_name:
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, name.split('.')[-1], new_head)
            else:
                setattr(model, name.split('.')[-1], new_head)
    
    return model

def train_anti_overfitting_model(model_name_key: str, model_name: str, num_classes: int, 
                                train_loader, val_loader, writer, model_dir: Path, class_names):
    """
    Training model dengan teknik anti-overfitting yang sangat agresif.
    """
    print(f"\nTRAINING MODEL: {model_name_key.upper()}")
    print(f"   Model: {model_name}")
    print(f"   Classes: {num_classes}")
    print("-" * 50)
    
    # Buat model
    model = create_model(model_name, num_classes, pretrained=True)
    if model is None:
        print(f"ERROR: Gagal membuat model {model_name}")
        return None
    
    # Tambahkan dropout untuk mengurangi overfitting
    model = add_dropout_to_model(model, dropout_rate=0.6)
    
    model = model.to(config.DEVICE)
    
    # Setup optimizer dengan weight decay yang lebih besar
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-3)
    
    # Setup learning rate scheduler yang lebih agresif
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=2, min_lr=1e-7)
    
    # Tracking variables
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    
    # Early stopping yang lebih ketat
    patience = 5  # Stop jika tidak ada improvement selama 5 epoch
    epochs_no_improve = 0
    
    print(f"Memulai training {config.EPOCHS} epochs...")
    print(f"   Early Stopping: {patience} epochs patience")
    print(f"   Learning Rate Scheduler: ReduceLROnPlateau (factor=0.3)")
    print(f"   Weight Decay: 1e-3 (AdamW)")
    print(f"   Dropout Rate: 0.6")
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        # Training
        train_loss, train_acc = train_step(
            model=model, dataloader=train_loader, loss_fn=loss_fn,
            optimizer=optimizer, device=config.DEVICE
        )
        
        # Validation
        val_loss, val_acc = val_step(
            model=model, dataloader=val_loader, loss_fn=loss_fn,
            device=config.DEVICE
        )
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        
        # Simpan metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Log ke TensorBoard
        writer.add_scalar(f'{model_name_key}/Train/Loss', train_loss, epoch)
        writer.add_scalar(f'{model_name_key}/Train/Accuracy', train_acc, epoch)
        writer.add_scalar(f'{model_name_key}/Val/Loss', val_loss, epoch)
        writer.add_scalar(f'{model_name_key}/Val/Accuracy', val_acc, epoch)
        writer.add_scalar(f'{model_name_key}/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Cek model terbaik
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0  # Reset counter
            
            # Simpan model terbaik
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
        
        # Progress
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"   Best:  {best_val_acc:.4f} (Epoch {best_epoch})")
        print(f"   LR:    {optimizer.param_groups[0]['lr']:.2e}")
        print(f"   No Improve: {epochs_no_improve}/{patience}")
        
        # Early stopping check
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
    
    # Generate confusion matrix dan classification report
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

def generate_confusion_matrix(model, val_loader, class_names, model_dir, model_name_key):
    """
    Generate confusion matrix dan classification report.
    """
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
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name_key.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Simpan confusion matrix
    cm_path = model_dir / f"{model_name_key}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Simpan classification report
    report_path = model_dir / f"{model_name_key}_classification_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"   Confusion Matrix disimpan: {cm_path}")
    print(f"   Classification Report disimpan: {report_path}")
    
    # Print per-class accuracy
    print(f"\n   Per-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        if i < len(report) - 3:  # Exclude 'accuracy', 'macro avg', 'weighted avg'
            acc = report[class_name]['f1-score']
            print(f"   {class_name:25}: {acc:.4f}")

def main():
    """
    Training anti-overfitting dengan teknik yang sangat agresif.
    """
    print("BATIK VISION - ANTI-OVERFITTING TRAINING MODE")
    print("="*60)
    
    # 1. Setup training anti-overfitting
    writer, experiment_dir, model_dir = setup_anti_overfitting_training()
    
    # 2. Buat data loaders
    print("\nMembuat data loaders...")
    try:
        train_loader, val_loader, class_names = create_dataloaders()
        num_classes = len(class_names)
        print(f"Data siap! {num_classes} kelas ditemukan.")
        print(f"   Kelas: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")
    except Exception as e:
        print(f"ERROR data loader: {e}")
        return
    
    # 3. Model mapping
    model_mapping = {
        "vit": "vit_base_patch16_224",
        "swin_transformer": "swin_base_patch4_window7_224", 
        "convnext_tiny": "convnext_tiny"
    }
    
    # 4. Training
    all_results = []
    
    for model_name_key in config.MODEL_LIST:
        if model_name_key not in model_mapping:
            print(f"WARNING: Model '{model_name_key}' tidak dikenali. Dilewati.")
            continue
            
        model_name = model_mapping[model_name_key]
        
        try:
            result = train_anti_overfitting_model(
                model_name_key=model_name_key,
                model_name=model_name,
                num_classes=num_classes,
                train_loader=train_loader,
                val_loader=val_loader,
                writer=writer,
                model_dir=model_dir,
                class_names=class_names
            )
            
            if result:
                all_results.append(result)
                
        except Exception as e:
            print(f"ERROR training {model_name_key}: {e}")
            continue
    
    # 5. Ringkasan
    if all_results:
        print(f"\nRINGKASAN HASIL")
        print("="*40)
        
        for result in all_results:
            print(f"{result['model_name']:15} | "
                  f"Best: {result['best_val_acc']:.4f} | "
                  f"Epochs: {result['epochs_trained']} | "
                  f"Time: {result['training_time']:.1f}s")
        
        best_model = max(all_results, key=lambda x: x['best_val_acc'])
        print(f"\nModel terbaik: {best_model['model_name']} "
              f"({best_model['best_val_acc']:.4f})")
    
    writer.close()
    print(f"\nHasil disimpan di: {experiment_dir}")

if __name__ == "__main__":
    main()
