import cv2
import os
import glob
import re

DATASET_DIR = "dataset"
OFFSET = 20  # Nomor urut dasar video asli (misal: AKU_01 -> AKU_21)

def mirror_dataset():
    if not os.path.exists(DATASET_DIR):
        print(f"Folder '{DATASET_DIR}' tidak ditemukan!")
        return

    # Ambil hanya file video asli (menghindari memproses video yang sudah di-flip)
    # Mencari pola misal: AKU_01.mp4 s/d AKU_20.mp4
    video_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.mp4")))
    
    processed_count = 0

    for video_path in video_files:
        filename = os.path.basename(video_path)
        
        # Regex untuk memisahkan Nama Kata dan Nomor Urut (contoh: 'AKU' dan '01')
        match = re.match(r"^(.+)_(\d+)\.mp4$", filename)
        if not match:
            continue

        label_name = match.group(1)      # Contoh: AKU
        num_str = match.group(2)         # Contoh: 01
        num_int = int(num_str)

        # Abaikan file yang nomornya sudah > OFFSET (supaya tidak men-flip hasil flip)
        if num_int > OFFSET:
            continue

        new_num = num_int + OFFSET      # 1 + 20 = 21, dst.
        new_filename = f"{label_name}_{new_num:02d}.mp4"
        save_path = os.path.join(DATASET_DIR, new_filename)

        # Cek jika file flipped sudah pernah dibuat sebelumnya
        if os.path.exists(save_path):
            print(f"Melewati (sudah ada): {new_filename}")
            continue

        # Proses Mirroring Video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Jika fps invalid, berikan default 30.0
        if fps <= 0 or fps is None:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip horizontal (sumbu Y)
            flipped_frame = cv2.flip(frame, 1)
            out.write(flipped_frame)

        cap.release()
        out.release()

        print(f"Berhasil dibuat: {filename} -> {new_filename}")
        processed_count += 1

    print(f"\nSelesai! Total {processed_count} video flipped berhasil ditambahkan ke '{DATASET_DIR}'.")

if __name__ == "__main__":
    mirror_dataset()