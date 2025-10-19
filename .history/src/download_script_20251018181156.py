import os
from datasets import load_dataset
# Kita tambahkan UnidentifiedImageError agar bisa ditangkap
from PIL import Image, UnidentifiedImageError

# Nama dataset dari Hugging Face Hub
dataset_name = "muhammadsalmanalfaridzi/Batik-Indonesia"
# Folder utama untuk menyimpan gambar JPG
output_dir = "Batik_Indonesia_JPG"

# Buat folder utama jika belum ada
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Muat dataset (akan menggunakan cache jika sudah diunduh)
print("Memuat dataset...")
dataset = load_dataset(dataset_name)
print("Dataset dimuat.")

# Ambil informasi nama kelas/label dari dataset 'train'
labels = dataset['train'].features['label'].names

# Proses dan simpan setiap gambar dari split 'train'
print("Memulai proses ekstraksi gambar...")
skipped_files = 0
for item in dataset['train']:
    # ---- MULAI BLOK TRY ----
    # Kita "coba" lakukan semua proses ini
    try:
        # Ambil gambar dan labelnya
        gambar: Image.Image = item['image']
        label_index = item['label']
        
        # Dapatkan nama label (contoh: "Batik Parang")
        label_name = labels[label_index]
        
        # Buat folder untuk kelas ini jika belum ada
        class_dir = os.path.join(output_dir, label_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
        # Ambil nama file asli (jika tersedia) atau buat nama file unik
        num_existing_files = len(os.listdir(class_dir))
        file_name = f"{label_name.replace(' ', '_')}_{num_existing_files + 1}.jpg"
        
        # Gabungkan path untuk menyimpan
        save_path = os.path.join(class_dir, file_name)
        
        # Simpan gambar
        if gambar.mode != 'RGB':
            gambar = gambar.convert('RGB')
        gambar.save(save_path)
        
    # ---- BLOK EXCEPT ----
    # Jika terjadi error "UnidentifiedImageError" (file rusak),
    # jalankan kode di bawah ini alih-alih crash.
    except UnidentifiedImageError:
        skipped_files += 1
        print(f"WARNING: 1 file gambar terdeteksi rusak atau tidak valid. Melewati...")
    # ----------------------

print(f"Ekstraksi selesai!")
print(f"Total file yang dilewati (rusak): {skipped_files}")