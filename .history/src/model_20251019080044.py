import timm
import torch
from src import config # Kita import config untuk daftar model dan device

def create_model(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Membuat model Computer Vision dari library timm.
    
    Args:
        model_name (str): Nama model yang akan dibuat (misal: 'vit_base_patch16_224').
        num_classes (int): Jumlah kelas output (misal: 38 untuk batik).
        pretrained (bool): Apakah akan menggunakan bobot pre-trained ImageNet.
    
    Returns:
        torch.nn.Module: Model yang sudah dibuat.
    """
    print(f"[Model] Membuat model: {model_name}...")
    
    try:
        # timm.create_model adalah fungsi ajaib:
        # 1. 'pretrained=True' akan otomatis men-download bobot ImageNet.
        # 2. 'num_classes=num_classes' akan otomatis MENGGANTI
        #    layer klasifikasi terakhir (misal: 1000 kelas ImageNet)
        #    dengan layer baru yang sesuai jumlah kelas kita (38 kelas).
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        return model
    
    except Exception as e:
        print(f"[Error] Gagal membuat model {model_name}: {e}")
        return None

# --- Blok Pengujian (Sangat Direkomendasikan) ---
# Kode ini HANYA akan berjalan jika Anda menjalankan file ini secara langsung
# (misal: `python src/models.py`)

if __name__ == "__main__":
    print("Menjalankan pengujian models.py...")
    
    # Kita butuh jumlah kelas untuk pengujian
    # Cara cepat: hitung folder di DATA_PATH dari config
    import os
    try:
        NUM_CLASSES = len(os.listdir(config.DATA_PATH))
        print(f"  > Ditemukan {NUM_CLASSES} kelas dari {config.DATA_PATH}")
    except FileNotFoundError:
        print(f"  > Error: Folder data di {config.DATA_PATH} tidak ditemukan.")
        print("  > Menggunakan 38 sebagai jumlah kelas default untuk tes.")
        NUM_CLASSES = 38 # Default jika data path salah

    # Buat data input palsu (dummy input) untuk tes
    # Ukuran: [Batch, Channel, Height, Width]
    dummy_input = torch.randn(
        2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE
    ).to(config.DEVICE)
    
    print(f"  > Membuat data input palsu ukuran: {dummy_input.shape}")
    print("-" * 30)

    # Loop dan uji setiap model dalam daftar di config.py
    for model_name_key in config.MODEL_LIST:
        
        # Ini adalah nama-nama model yang sebenarnya di library 'timm'
        model_arch_names = {
            "vit": "vit_base_patch16_224",
            "swin_transformer": "swin_base_patch4_window7_224",
            "convnext_tiny": "convnext_tiny"
        }
        
        model_name = model_arch_names.get(model_name_key)
        
        if model_name:
            model = create_model(model_name=model_name, num_classes=NUM_CLASSES)
            
            if model:
                model = model.to(config.DEVICE)
                model.eval() # Set ke mode evaluasi untuk tes
                
                # Coba lewatkan data palsu ke model
                with torch.no_grad():
                    output = model(dummy_input)
                
                print(f"  > Tes Forward Pass... SUKSES")
                print(f"  > Ukuran Output: {output.shape}") # Harusnya [2, 38]
                print(f"  > Tes {model_name_key} selesai.")
                print("-" * 30)
        else:
            print(f"[Warning] Kunci model '{model_name_key}' di config.py tidak dikenali.")

    print("\n[Sukses] models.py berfungsi dengan baik!")