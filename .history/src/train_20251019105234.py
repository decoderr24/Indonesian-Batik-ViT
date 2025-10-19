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

def setup_experiment_logging(experiment_name: str):
    """
    Setup logging dan direktori untuk eksperimen.
    
    Args:
        experiment_name (str): Nama eksperimen
        
    Returns:
        tuple: (writer, experiment_dir, model_dir)
    """
    # Buat direktori untuk eksperimen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("outputs") / f"{experiment_name}_{timestamp}"
    model_dir = experiment_dir / "models"
    log_dir = experiment_dir / "logs"
    
    # Buat direktori jika belum ada
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir))
    
    print(f"[Setup] Eksperimen: {experiment_name}")
    print(f"[Setup] Direktori: {experiment_dir}")
    print(f"[Setup] Model akan disimpan di: {model_dir}")
    print(f"[Setup] Logs akan disimpan di: {log_dir}")
    
    return writer, experiment_dir, model_dir

def save_training_results(experiment_dir: Path, model_name: str, 
                         train_losses: list, val_losses: list, 
                         train_accs: list, val_accs: list, 
                         best_val_acc: float, best_epoch: int):
    """
    Simpan hasil training dalam format JSON dan plot.
    
    Args:
        experiment_dir (Path): Direktori eksperimen
        model_name (str): Nama model
        train_losses (list): List loss training per epoch
        val_losses (list): List loss validasi per epoch
        train_accs (list): List akurasi training per epoch
        val_accs (list): List akurasi validasi per epoch
        best_val_acc (float): Akurasi validasi terbaik
        best_epoch (int): Epoch dengan akurasi terbaik
    """
    # Simpan hasil dalam format JSON
    results = {
        "model_name": model_name,
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "total_epochs": len(train_losses),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accs,
        "val_accuracies": val_accs,
        "config": {
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "image_size": config.IMAGE_SIZE,
            "epochs": config.EPOCHS,
            "device": config.DEVICE
        }
    }
    
    # Simpan JSON
    results_file = experiment_dir / f"{model_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Buat plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Loss
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title(f'{model_name} - Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title(f'{model_name} - Training & Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Simpan plot
    plot_file = experiment_dir / f"{model_name}_training_curves.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Save] Hasil training disimpan di: {results_file}")
    print(f"[Save] Plot training disimpan di: {plot_file}")

def train_model(model_name_key: str, model_name: str, num_classes: int, 
                train_loader, val_loader, writer, model_dir: Path):
    """
    Melatih satu model dan menyimpan hasilnya.
    
    Args:
        model_name_key (str): Kunci model dari config (misal: 'vit')
        model_name (str): Nama model timm (misal: 'vit_base_patch16_224')
        num_classes (int): Jumlah kelas
        train_loader: DataLoader untuk training
        val_loader: DataLoader untuk validasi
        writer: TensorBoard writer
        model_dir (Path): Direktori untuk menyimpan model
        
    Returns:
        dict: Hasil training (best accuracy, best epoch, dll)
    """
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL: {model_name_key.upper()} ({model_name})")
    print(f"{'='*60}")
    
    # 1. Buat model
    model = create_model(model_name, num_classes, pretrained=True)
    if model is None:
        print(f"[Error] Gagal membuat model {model_name}")
        return None
    
    model = model.to(config.DEVICE)
    
    # 2. Setup loss function dan optimizer
    loss_fn = nn.CrossEntropyLoss()
    # 1e-2 atau 1e-3 adalah nilai awal yang bagus untuk dicoba
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-3)
    
    # 3. Setup tracking variables
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_epoch = 0
    #eraly stopping
    patience = 10  # <-- TAMBAHKAN INI (artinya: berhenti jika 10 epoch tidak ada kemajuan)
    epochs_no_improve = 0 # <-- TAMBAHKAN INI (artinya: berhenti jika 10 epoch tidak ada kemajuan)
    
    # 4. Training loop
    print(f"[Training] Memulai training untuk {config.EPOCHS} epochs...")
    print(f"[Training] Device: {config.DEVICE}")
    print(f"[Training] Learning Rate: {config.LEARNING_RATE}")
    print(f"[Training] Batch Size: {config.BATCH_SIZE}")
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        print(f"\n[Epoch {epoch+1}/{config.EPOCHS}]")
        
        # Training step
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=config.DEVICE
        )
        
        # Validation step
        val_loss, val_acc = val_step(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
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
        
        # Cek apakah ini model terbaik
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
            print(f"[Save] Model terbaik disimpan di: {model_path}")
        else:
            epochs_no_improve += 1
        # Print progress
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
    if epochs_no_improve >= patience:
        print(f"[Early Stopping] Training berhenti karena tidak ada kemajuan setelah {patience} epoch")
        break
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n[Training] Selesai dalam {training_time:.2f} detik")
    print(f"[Training] Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    # Simpan model final
    final_model_path = model_dir / f"{model_name_key}_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': config.EPOCHS,
        'val_accuracy': val_acc,
        'model_name': model_name,
        'num_classes': num_classes
    }, final_model_path)
    
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
    Fungsi utama untuk menjalankan training semua model.
    """
    print("="*80)
    print("BATIK VISION PROJECT - TRAINING SCRIPT")
    print("="*80)
    
    # 1. Setup eksperimen
    experiment_name = "batik_classification"
    writer, experiment_dir, model_dir = setup_experiment_logging(experiment_name)
    
    # 2. Buat data loaders
    print("\n[Data] Membuat data loaders...")
    try:
        train_loader, val_loader, class_names = create_dataloaders()
        num_classes = len(class_names)
        print(f"[Data] Berhasil! {num_classes} kelas ditemukan.")
        print(f"[Data] Kelas: {class_names}")
    except Exception as e:
        print(f"[Error] Gagal membuat data loaders: {e}")
        return
    
    # 3. Mapping model names dari config ke timm
    model_mapping = {
        "vit": "vit_base_patch16_224",
        "swin_transformer": "swin_base_patch4_window7_224", 
        "convnext_tiny": "convnext_tiny"
    }
    
    # 4. Training loop untuk setiap model
    all_results = []
    
    for model_name_key in config.MODEL_LIST:
        if model_name_key not in model_mapping:
            print(f"[Warning] Model '{model_name_key}' tidak dikenali. Dilewati.")
            continue
            
        model_name = model_mapping[model_name_key]
        
        try:
            # Train model
            result = train_model(
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
                
                # Simpan hasil individual
                save_training_results(
                    experiment_dir=experiment_dir,
                    model_name=model_name_key,
                    train_losses=result['train_losses'],
                    val_losses=result['val_losses'],
                    train_accs=result['train_accs'],
                    val_accs=result['val_accs'],
                    best_val_acc=result['best_val_acc'],
                    best_epoch=result['best_epoch']
                )
                
        except Exception as e:
            print(f"[Error] Gagal training model {model_name_key}: {e}")
            continue
    
    # 5. Simpan ringkasan hasil
    if all_results:
        summary = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_models": len(all_results),
            "results": all_results,
            "best_model": max(all_results, key=lambda x: x['best_val_acc'])
        }
        
        summary_file = experiment_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("RINGKASAN HASIL TRAINING")
        print(f"{'='*60}")
        
        for result in all_results:
            print(f"{result['model_name']:15} | Best Val Acc: {result['best_val_acc']:.4f} | "
                  f"Final Val Acc: {result['final_val_acc']:.4f} | "
                  f"Time: {result['training_time']:.1f}s")
        
        best_model = summary['best_model']
        print(f"\nModel terbaik: {best_model['model_name']} dengan akurasi {best_model['best_val_acc']:.4f}")
        print(f"Ringkasan lengkap disimpan di: {summary_file}")
    
    # 6. Tutup TensorBoard writer
    writer.close()
    print(f"\n[Complete] Training selesai! Hasil disimpan di: {experiment_dir}")

if __name__ == "__main__":
    main()
