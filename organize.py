import os
import shutil

CLASSES_LIST = [
    'AKU', 'APA', 'BAGAIMANA', 'BELAJAR', 'BERANGKAT', 'DATANG', 
    'DIA', 'DIMANA', 'INGAT', 'KAMU', 'KAPAN', 'KITA', 
    'MAKAN', 'MENGAPA', 'MEREKA', 'SIAPA'
]

SOURCE_DIR = "dataset"
TARGET_DIR = "dataset_per_kata"

def reorganize_dataset():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    for cls in CLASSES_LIST:
        os.makedirs(os.path.join(TARGET_DIR, cls), exist_ok=True)

    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Folder '{SOURCE_DIR}' tidak ditemukan!")
        return

    peraga_folders = [f for f in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, f))]
    
    print(f"Ditemukan folder peraga: {peraga_folders}\n")

    # 1. Iterasi tiap folder peraga
    for peraga in peraga_folders:
        peraga_path = os.path.join(SOURCE_DIR, peraga)
        files = os.listdir(peraga_path)

        # RESET counter ke 1 setiap kali masuk ke folder peraga baru
        counters = {cls: 1 for cls in CLASSES_LIST}

        print(f"Memproses folder: {peraga}...")

        # Urutkan nama file agar diproses dari urutan terkecil (01, 02, dst.)
        for file_name in sorted(files):
            if not file_name.endswith(".mp4"):
                continue

            extracted_label = file_name.split("_")[0].upper()

            if extracted_label in CLASSES_LIST:
                # Sekarang nomor urut akan reset kembali ke 01 untuk setiap peraga
                new_filename = f"{peraga}_{extracted_label}_{counters[extracted_label]:02d}.mp4"
                
                src_path = os.path.join(peraga_path, file_name)
                dst_path = os.path.join(TARGET_DIR, extracted_label, new_filename)

                shutil.copy(src_path, dst_path)
                5
                # Increment counter khusus untuk peraga ini
                counters[extracted_label] += 1
            else:
                print(f"  [Peringatan] File '{file_name}' memiliki label '{extracted_label}' yang tidak terdaftar.")

    print("\n================================================================")
    print("Reorganisasi dataset selesai!")
    print("================================================================")

if __name__ == "__main__":
    reorganize_dataset()