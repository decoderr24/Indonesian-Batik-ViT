import os
from datasets import load_dataset
from PIL import Image

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
# Contoh: ['Batik Cendrawasih', 'Batik Dayak', ...]
labels = dataset['train'].features['label'].names

# Proses dan simpan setiap gambar dari split 'train'
print("Memulai proses ekstraksi gambar...")
for item in dataset['train']:
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
    # Karena dataset ini tidak menyediakan nama file, kita buat sendiri
    # Kita bisa gunakan hash dari gambar atau nomor acak, tapi untuk simpelnya kita hitung saja
    num_existing_files = len(os.listdir(class_dir))
    file_name = f"{label_name.replace(' ', '_')}_{num_existing_files + 1}.jpg"
    
    # Gabungkan path untuk menyimpan
    save_path = os.path.join(class_dir, file_name)
    
    # Simpan gambar
    # Pastikan gambar dalam mode RGB sebelum menyimpan sebagai JPG
    if gambar.mode != 'RGB':
        gambar = gambar.convert('RGB')
    gambar.save(save_path)

print(f"Ekstraksi selesai! Gambar tersimpan di folder '{output_dir}'")