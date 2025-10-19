import torch
from pathlib import Path

# Definisikan ROOT path proyek (folder batik_vision_project)
ROOT_PATH = Path(__file__).resolve().parent.parent 

# Path ke data
DATA_PATH = ROOT_PATH / "Batik-Indonesia" # <-- GANTI BARIS INI

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32  # Dikurangi untuk laptop
IMAGE_SIZE = 224 # Ukuran input untuk ViT/Swin
LEARNING_RATE = 1e-4
EPOCHS = 5  # Dikurangi untuk testing awal

# Pengaturan split
TEST_SPLIT_SIZE = 0.2 # 20% untuk validasi
RANDOM_SEED = 42 # Agar hasil split selalu sama

# Daftar model yang akan diuji
# Mulai dengan model terkecil dulu untuk testing
MODEL_LIST = ["convnext_tiny"]  # Model terkecil untuk testing awal