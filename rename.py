import os
from pathlib import Path

# Ganti dengan path folder Anda jika berbeda
folder_path = Path("test_f1")

# Ambil semua file .mp4 di dalam folder
files = sorted(folder_path.glob("*.mp4"))

# Kelompokkan file berdasarkan prefix (misal: AKU, APA, BAGAIMANA)
groups = {}
for file in files:
  # Memisahkan nama file berdasarkan underscore terakhir
  parts = file.stem.rsplit("_", 1)
  if len(parts) == 2:
    prefix = parts[0]
    if prefix not in groups:
      groups[prefix] = []
    groups[prefix].append(file)

# Proses rename untuk setiap kelompok dengan format dua digit (01, 02, dst.)
for prefix, group_files in groups.items():
  group_files.sort()  # Mengurutkan dari 21 ke 30
  for index, file in enumerate(group_files, start=1):
    # :02d berfungsi untuk memaksa angka menjadi 2 digit (01, 02, 03, ...)
    new_name = f"{prefix}_{index:02d}{file.suffix}"
    new_path = folder_path / new_name

    file.rename(new_path)
    print(f"Berhasil: {file.name} -> {new_name}")

print("\nSemua file berhasil di-rename menjadi format 01-10!")