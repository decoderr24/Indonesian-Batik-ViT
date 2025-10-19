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

# Import modul yang sudah dibuat
from src import config
from src.data_loader import create_dataloaders
from src.model import create_model
from src.engine import train_step, val_step

def setup_fast_training():
    """
    Setup untuk training yang lebih cepat di laptop.
    """
    print("SETUP TRAINING CEPAT UNTUK LAPTOP")
    print("="*50)
    
    # Override config untuk training cepat
    config.BATCH_SIZE = 4  # Sangat kecil untuk laptop
    config.EPOCHS = 3      # Hanya 3 epoch untuk testing
    config.IMAGE_SIZE = 128  # Resolusi lebih kecil
    
    print(f"Konfigurasi Training Cepat:")
    print(f"   - Batch Size: {config.BATCH_SIZE}")
    print(f"   - Epochs: {config.EPOCHS}")
    print(f"   - Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"   - Device: {config.DEVICE}")
    print(f"   - Model: {config.MODEL_LIST[0] if config.MODEL_LIST else 'None'}")
    
    # Buat direktori untuk hasil
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("outputs") / f"fast_training_{timestamp}"
    model_dir = experiment_dir / "models"
    log_dir = experiment_dir / "logs"
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_dir))
    
    return writer, experiment_dir, model_dir

def train_fast_model(model_name_key: str, model_name: str, num_classes: int, 
                    train_loader, val_loader, writer, model_dir: Path):
    """
    Training model dengan optimasi untuk laptop.
    """
    print(f"\n🔥 TRAINING MODEL: {model_name_key.upper()}")
    print(f"   Model: {model_name}")
    print(f"   Classes: {num_classes}")
    print("-" * 40)
    
    # Buat model
    model = create_model(model_name, num_classes, pretrained=True)
    if model is None:
        print(f"❌ Gagal membuat model {model_name}")
        return None
    
    model = model.to(config.DEVICE)
    
    # Setup optimizer dengan learning rate yang lebih tinggi untuk konvergensi cepat
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE * 2)  # 2x lebih cepat
    
    # Tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"⏱️  Memulai training {config.EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        print(f"\n📈 Epoch {epoch+1}/{config.EPOCHS}")
        
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
        
        # Cek model terbaik
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            # Simpan model terbaik
            model_path = model_dir / f"{model_name_key}_best.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_accuracy': val_acc,
                'model_name': model_name,
                'num_classes': num_classes
            }, model_path)
            print(f"💾 Model terbaik disimpan: {model_path}")
        
        # Progress
        print(f"   📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"   📊 Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"   🏆 Best:  {best_val_acc:.4f} (Epoch {best_epoch})")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ Training selesai!")
    print(f"   ⏱️  Waktu: {training_time:.1f} detik")
    print(f"   🏆 Best Accuracy: {best_val_acc:.4f}")
    
    return {
        'model_name': model_name_key,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_val_acc': val_acc,
        'training_time': training_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def main():
    """
    Training cepat untuk laptop.
    """
    print("🎯 BATIK VISION - FAST TRAINING MODE")
    print("="*50)
    
    # 1. Setup training cepat
    writer, experiment_dir, model_dir = setup_fast_training()
    
    # 2. Buat data loaders
    print("\n📁 Membuat data loaders...")
    try:
        train_loader, val_loader, class_names = create_dataloaders()
        num_classes = len(class_names)
        print(f"✅ Data siap! {num_classes} kelas ditemukan.")
        print(f"   Kelas: {class_names[:5]}{'...' if len(class_names) > 5 else ''}")
    except Exception as e:
        print(f"❌ Error data loader: {e}")
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
            print(f"⚠️  Model '{model_name_key}' tidak dikenali. Dilewati.")
            continue
            
        model_name = model_mapping[model_name_key]
        
        try:
            result = train_fast_model(
                model_name_key=model_name_key,
                model_name=model_name,
                num_classes=num_classes,
                train_loader=train_loader,
                val_loader=val_loader,
                writer=writer,
                model_dir=model_dir
            )
            
            if result:
                all_results.append(result)
                
        except Exception as e:
            print(f"❌ Error training {model_name_key}: {e}")
            continue
    
    # 5. Ringkasan
    if all_results:
        print(f"\n🎉 RINGKASAN HASIL")
        print("="*30)
        
        for result in all_results:
            print(f"📊 {result['model_name']:15} | "
                  f"Best: {result['best_val_acc']:.4f} | "
                  f"Time: {result['training_time']:.1f}s")
        
        best_model = max(all_results, key=lambda x: x['best_val_acc'])
        print(f"\n🏆 Model terbaik: {best_model['model_name']} "
              f"({best_model['best_val_acc']:.4f})")
    
    writer.close()
    print(f"\n💾 Hasil disimpan di: {experiment_dir}")

if __name__ == "__main__":
    main()
