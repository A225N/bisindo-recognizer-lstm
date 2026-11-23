def get_classes_list(dataset_dir = "dataset"):
    label_map = load_labels(dataset_dir)
    return sorted(set(label_map.values()))

def evaluate_peraga_with_metrics(dataset_dir, test_peraga, model_path):
    classes_list = get_classes_list(dataset_dir)
    print(f"Classes list ({len(classes_list)} kelas): {classes_list}")
    
    model = load_model(model_path)
    
    # Mengumpulkan semua video yang hanya berasal dari peraga test yang diinginkan
    video_files = []
    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in sorted(os.listdir(class_path)):
            if filename.endswith(".mp4"):
                # Cek apakah nama file diawali dengan peraga yang diuji (misal: peraga_3_...)
                if filename.startswith(test_peraga):
                    video_files.append(os.path.join(class_path, filename))
    
    if not video_files:
        print(f"Tidak ada file video ditemukan untuk peraga: {test_peraga}")
        return

    total_videos = len(video_files)
    metrics = {cls: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for cls in classes_list}
    
    print(f"\nMemulai pengujian untuk {total_videos} video dari '{test_peraga}'...\n")
    print(" ================================================================ ")

    for video_path in sorted(video_files):
        filename = os.path.basename(video_path)
        parts = filename.split('_')
        true_label = parts[2].upper() # Menyesuaikan format peraga_X_KATA_01.mp4 -> Indeks ke-2 adalah label
        
        if true_label not in classes_list:
            continue
        
        try:
            _, landmarks = landmarks_extraction(video_path)
            if landmarks is None:
                continue
                
            input_tensor = np.expand_dims(landmarks, axis=0)
            
            probability = model.predict(input_tensor, verbose=0)[0]
            predicted_index = int(np.argmax(probability))
            predicted_class = classes_list[predicted_index]
            confidence = float(probability[predicted_index])
            
            # Update matriks One-vs-All
            for cls in classes_list:
                is_actual = (true_label == cls)
                is_predicted = (predicted_class == cls)
                
                if is_actual and is_predicted:
                    metrics[cls]["TP"] += 1
                elif not is_actual and is_predicted:
                    metrics[cls]["FP"] += 1
                elif is_actual and not is_predicted:
                    metrics[cls]["FN"] += 1
                else:
                    metrics[cls]["TN"] += 1
            
            status = "BENAR" if predicted_class == true_label else "SALAH"
            print(f"Video          : {filename}")
            print(f"True Label     : {true_label}")
            print(f"Predicted Class: {predicted_class} ({confidence * 100:.2f}%) -> [{status}]")
            print("-" * 64)
            
        except Exception as e:
            print(f"Error saat memproses {filename}: {str(e)}")

    # Kalkulasi F1-Score, Precision, Recall (Sama seperti sebelumnya)
    print("\n ================================================================ ")
    print("                HASIL METRIK PER KELAS & F1-SCORE                 ")
    print(" ================================================================ ")
    
    f1_scores = []
    for cls in classes_list:
        tp, fp, fn, tn = metrics[cls]["TP"], metrics[cls]["FP"], metrics[cls]["FN"], metrics[cls]["TN"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        
        print(f"Kelas: {cls} | Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}% | F1-Score: {f1:.4f}")

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    print(" ================================================================ ")
    print(f" Macro-Average F1-Score Keseluruhan ({test_peraga}) : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print(" ================================================================ ")

if __name__ == "__main__":
    DATASET_DIR = "dataset_per_kata" # Sesuaikan dengan folder dataset Anda
    
    # PERMUTASI PERAGA TESTING: 
    # Ubah ini sesuai peraga yang ingin Anda uji (misal: "peraga_3", "peraga_2", atau "peraga_1")
    target_test_peraga = "peraga_3" 
    
    # Path model hasil training yang ingin diuji
    MODEL_PATH = "peraga_1_2_train_prominence_78.h5" # Ganti dengan nama file model .h5 terbaru Anda
    
    # Membuat path direktori testing secara dinamis berdasarkan peraga yang dipilih
    # Asumsi struktur folder per kata: dataset_per_kata/[KATA]/peraga_X_[KATA]_[no].mp4
    # Kita bisa buat fungsi helper atau langsung ambil sampel path foldernya:
    # Atau jika ingin mengecek video berdasarkan nama file yang mengandung peraga tertentu:
    
    print(f"Menjalankan evaluasi F1-Score khusus untuk: {target_test_peraga}")
    evaluate_peraga_with_metrics(DATASET_DIR, target_test_peraga, MODEL_PATH)