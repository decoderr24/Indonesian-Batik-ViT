import sys
from pathlib import Path
# tambahkan parent project ke sys.path sehingga 'src' dapat diimport saat menjalankan skrip langsung
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from src import config  # Mengimpor dari file config.py Anda
import matplotlib.pyplot as plt
import warnings

# --- 1. Mendefinisikan Transformasi (Augmentasi) ---

# Statistik ImageNet untuk normalisasi (penting untuk model pre-trained)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Transformasi untuk data TRAINING
# Tujuannya: "menyiksa" data agar model bisa generalisasi
train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), # Ukuran seragam
    transforms.RandomHorizontalFlip(p=0.5), # Balik gambar horizontal
    transforms.RandomRotation(degrees=15),  # Putar gambar sedikit
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # Ubah warna
    transforms.ToTensor(), # Konversi ke tensor PyTorch
    transforms.Normalize(mean=MEAN, std=STD) # Normalisasi
])

# Transformasi untuk data VALIDASI
# Tujuannya: Hanya membersihkan data untuk evaluasi, TANPA augmentasi acak
val_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), # Ukuran seragam
    transforms.ToTensor(), # Konversi ke tensor PyTorch
    transforms.Normalize(mean=MEAN, std=STD) # Normalisasi
])


# --- 2. Helper Class untuk Menerapkan Transformasi Berbeda ---
# INI PENTING:
# Kita perlu membagi dataset (split) SEBELUM menerapkan augmentasi.
# Helper class ini memungkinkan kita menerapkan transform yang berbeda (train/val)
# pada dataset subset yang sudah dibagi.

class TransformedDataset(Dataset):
    """Wrapper Dataset untuk menerapkan transformasi ke Subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        # Ambil data asli (gambar, label) dari subset
        x, y = self.subset[index]
        
        # Terapkan transformasi jika ada
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.subset)

# --- 3. Fungsi Utama Pembuat DataLoader ---

def create_dataloaders():
    """
    Fungsi utama untuk membuat dan mengembalikan data loader
    untuk training dan validasi.
    """
    
    # --- LANGKAH A: Muat Dataset Induk ---
    # Muat semua 2599 gambar menggunakan ImageFolder, TANPA transform
    print(f"[Data] Memuat dataset induk dari: {config.DATA_PATH}")
    full_dataset = datasets.ImageFolder(config.DATA_PATH)
    
    # Simpan nama kelas
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"[Data] Ditemukan {num_classes} kelas: {class_names}")

    # --- LANGKAH B: Bagi Dataset 80:20 (Secara Hati-hati) ---
    print(f"[Data] Membagi dataset 80:20 (seed: {config.RANDOM_SEED})...")
    total_size = len(full_dataset)
    val_size = int(total_size * config.TEST_SPLIT_SIZE)
    train_size = total_size - val_size
    
    # Bagi dataset menggunakan random_split dengan SEED yang tetap
    # Ini memastikan pembagian data SELALU SAMA setiap kali skrip dijalankan
    train_dataset_raw, val_dataset_raw = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    print(f"[Data] Ukuran Train: {len(train_dataset_raw)} | Ukuran Validasi: {len(val_dataset_raw)}")

    # --- LANGKAH C: Terapkan Transformasi yang Berbeda ---
    train_dataset = TransformedDataset(train_dataset_raw, transform=train_transform)
    val_dataset = TransformedDataset(val_dataset_raw, transform=val_transform)

    # --- LANGKAH D: Mengatasi Ketidakseimbangan Kelas (Wajib!) ---
    print("[Data] Menghitung bobot untuk mengatasi ketidakseimbangan kelas...")
    
    # 1. Ambil semua label (target) HANYA dari set training
    train_targets = [full_dataset.targets[i] for i in train_dataset_raw.indices]
    
    # 2. Hitung jumlah gambar per kelas
    #    Kita gunakan bincount untuk efisiensi
    class_counts = np.bincount(train_targets)
    
    # 3. Hitung bobot kebalikan (inverse weight) untuk setiap kelas
    #    Kelas langka -> bobot tinggi
    #    Kelas umum -> bobot rendah
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    
    # 4. Buat daftar bobot untuk SETIAP sampel di set training
    #    Setiap sampel akan memiliki bobot sesuai kelasnya
    sample_weights = class_weights[train_targets]
    
    # 5. Buat Sampler
    #    WeightedRandomSampler akan mengambil data berdasarkan bobot ini
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True # Izinkan pengambilan sampel berulang (oversampling)
    )
    
    print("[Data] WeightedRandomSampler berhasil dibuat.")

    # --- LANGKAH E: Buat DataLoaders ---
    
    # DataLoader untuk Training
    # PENTING: Jika menggunakan 'sampler', 'shuffle' HARUS False.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2, # Gunakan 2 proses untuk memuat data
        pin_memory=True, # Percepat transfer ke GPU
        shuffle=False 
    )
    
    # DataLoader untuk Validasi
    # Tidak perlu sampler, tidak perlu shuffle (evaluasi harus konsisten)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        shuffle=False 
    )
    
    print("[Data] Data loader untuk Train dan Validasi siap.")
    
    return train_loader, val_loader, class_names


# --- 5. Blok Pengujian (Opsional tapi Sangat Direkomendasikan) ---
# Kode ini HANYA akan berjalan jika Anda menjalankan file ini secara langsung
# (misal: `python src/data_loader.py`)
# Ini sangat berguna untuk memverifikasi bahwa loader Anda berfungsi.

if __name__ == "__main__":
    print("Menjalankan pengujian data_loader.py...")
    
    # Coba buat data loader
    train_loader, val_loader, class_names = create_dataloaders()
    
    print(f"\nTotal kelas: {len(class_names)}")
    
    # Ambil satu batch dari train_loader
    print("\nMengambil 1 batch dari train_loader (untuk tes)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Abaikan peringatan UserWarning dari matplotlib
        
        try:
            images, labels = next(iter(train_loader))
            
            print(f"  > Ukuran batch gambar: {images.shape}") # [Batch, Channel, H, W]
            print(f"  > Ukuran batch label: {labels.shape}")
            print(f"  > Contoh 5 label di batch ini: {labels[:5]}")
            
            # Coba visualisasikan 1 gambar (untuk cek normalisasi)
            img_to_show = images[0].permute(1, 2, 0).numpy() # Ubah (C, H, W) -> (H, W, C)
            # Denormalisasi (penting untuk visualisasi)
            img_to_show = STD * img_to_show + MEAN 
            img_to_show = np.clip(img_to_show, 0, 1) # Pastikan nilai antara 0 dan 1
            
            plt.imshow(img_to_show)
            plt.title(f"Contoh Gambar (Label: {class_names[labels[0]]})")
            plt.axis('off')
            plt.show()
            
            print("\n[Sukses] data_loader.py berfungsi dengan baik!")
            
        except Exception as e:
            print(f"\n[Error] Gagal menguji data loader: {e}")