# src/config.py

import torch

# Path ke data
DATA_PATH = "Batik_Indonesia_JPG" 

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMAGE_SIZE = 224 # Ukuran input untuk ViT/Swin
LEARNING_RATE = 1e-4
EPOCHS = 50

# Pengaturan split
TEST_SPLIT_SIZE = 0.2 # 20% untuk validasi
RANDOM_SEED = 42 # Agar hasil split selalu sama

# Daftar model yang akan diuji
MODEL_LIST = ["vit", "swin_transformer", "convnext_tiny"]