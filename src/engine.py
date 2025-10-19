import torch
import torch.nn as nn
from tqdm.auto import tqdm # Untuk progress bar yang bagus

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """
    Melakukan satu epoch training.
    
    Mengatur model ke mode training, melakukan forward pass,
    menghitung loss, melakukan backpropagation, dan update weights.
    """
    # 1. Set model ke mode training
    # Ini penting untuk mengaktifkan lapisan seperti Dropout dan BatchNorm
    model.train()
    
    # 2. Setup variabel pelacak loss dan akurasi
    train_loss, train_acc = 0, 0
    
    # 3. Loop melalui data loader
    # Gunakan tqdm untuk progress bar
    for X, y in tqdm(dataloader, desc="Training"):
        # Pindahkan data ke device (GPU jika ada)
        X, y = X.to(device), y.to(device)
        
        # 4. Forward pass
        y_pred_logits = model(X)
        
        # 5. Hitung loss
        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item() 
        
        # 6. Nol-kan gradien optimizer
        optimizer.zero_grad()
        
        # 7. Backpropagation
        loss.backward()
        
        # 8. Update weights
        optimizer.step()
        
        # 9. Hitung akurasi
        # Ambil kelas dengan probabilitas tertinggi
        y_pred_class = torch.argmax(y_pred_logits, dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred_logits)
        
    # 10. Hitung rata-rata loss dan akurasi per epoch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

def val_step(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             loss_fn: torch.nn.Module,
             device: torch.device):
    """
    Melakukan satu epoch validasi.
    
    Mengatur model ke mode evaluasi, melakukan forward pass,
    dan menghitung loss/akurasi. Tidak ada backpropagation.
    """
    # 1. Set model ke mode evaluasi
    # Ini penting untuk menonaktifkan Dropout dan BatchNorm
    model.eval() 
    
    # 2. Setup variabel pelacak loss dan akurasi
    val_loss, val_acc = 0, 0
    
    # 3. Matikan perhitungan gradien
    # Ini menghemat memori dan komputasi
    with torch.no_grad():
        # 4. Loop melalui data loader
        for X, y in tqdm(dataloader, desc="Validasi"):
            # Pindahkan data ke device
            X, y = X.to(device), y.to(device)
            
            # 5. Forward pass
            y_pred_logits = model(X)
            
            # 6. Hitung loss
            loss = loss_fn(y_pred_logits, y)
            val_loss += loss.item()
            
            # 7. Hitung akurasi
            y_pred_class = torch.argmax(y_pred_logits, dim=1)
            val_acc += (y_pred_class == y).sum().item() / len(y_pred_logits)
            
    # 8. Hitung rata-rata loss dan akurasi per epoch
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    
    return val_loss, val_acc