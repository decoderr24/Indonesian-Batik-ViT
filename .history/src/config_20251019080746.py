# src/config.py

import torch
from pathlib import Path  # <-- 1. Tambahkan ini

# Definisikan ROOT path proyek (folder batik_vision_project)
ROOT_PATH = Path(__file__).resolve().parent.parent 
# __file__ -> .../src/config.py
# .parent -> .../src/
# .parent -> .../ (ROOT PROYEK)
# Path ke data
DATA_PATH = "BatikIndonesia" 

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMAGE_SIZE = 224 # Ukuran input untuk ViT/Swin
LEARNING_RATE = 1e-4
EPOCHS = 30

# Pengaturan split
TEST_SPLIT_SIZE = 0.2 # 20% untuk validasi
RANDOM_SEED = 42 # Agar hasil split selalu sama

# Daftar model yang akan diuji
MODEL_LIST = ["vit", "swin_transformer", "convnext_tiny"]