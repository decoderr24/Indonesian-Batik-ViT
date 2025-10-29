# src/data_distribution.py
import sys
from pathlib import Path

# Tambahkan root proyek ke sys.path agar 'src.config' bisa diimpor
ROOT_DIR = Path(__file__).resolve().parent.parent  # naik ke batik_vision_project/
sys.path.insert(0, str(ROOT_DIR))

from src import config
from collections import Counter
import os

def analyze_class_distribution():
    data_path = config.DATA_PATH
    if not data_path.exists():
        print(f"ERROR: Path dataset tidak ditemukan: {data_path}")
        return

    class_counts = {}
    total_images = 0

    for class_dir in sorted(data_path.iterdir()):
        if class_dir.is_dir():
            image_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ]
            count = len(image_files)
            class_counts[class_dir.name] = count
            total_images += count

    # Urutkan berdasarkan jumlah gambar (ascending)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])

    print(f"📊 ANALISIS DISTRIBUSI KELAS")
    print("=" * 50)
    print(f"Total Kelas      : {len(class_counts)}")
    print(f"Total Gambar     : {total_images}")
    print(f"Rata-rata/Gambar : {total_images / len(class_counts):.1f}")
    print("-" * 50)
    print(f"{'Kelas':<30} | {'Jumlah'}")
    print("-" * 50)
    for cls, count in sorted_classes:
        status = "⚠️  SANGAT SEDIKIT" if count < 5 else ""
        print(f"{cls:<30} | {count:>6} {status}")

    # Identifikasi kelas kritis
    critical_classes = [cls for cls, cnt in class_counts.items() if cnt < 5]
    if critical_classes:
        print("\n❗ KELAS KRITIS (< 5 gambar):")
        for cls in critical_classes:
            print(f"  - {cls} ({class_counts[cls]} gambar)")
        print("\n💡 Rekomendasi: Pertimbangkan menggabungkan atau menghapus kelas ini.")
    else:
        print("\n✅ Semua kelas memiliki ≥5 gambar. Distribusi relatif seimbang.")

if __name__ == "__main__":
    analyze_class_distribution()