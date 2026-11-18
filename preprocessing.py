import os
import cv2
import csv
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Global Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20

# BISINDO dataset layout (see PROJECT_CONTEXT.md §6): videos live directly
# inside DATASET_DIR - there are NO per-class subfolders like UCF50 had.
# Two supported naming schemes:
#   1. "WORD_NN.mp4"            (e.g. BELAJAR_01.mp4)  -> label = "BELAJAR"
#      This is the default convention for the main training dataset.
#   2. A "labels.csv" (columns: filename,label) sitting inside DATASET_DIR.
#      Required for folders that DON'T follow the WORD_NN convention, e.g.
#      the test/ folder which uses word-LAST names like
#      "SIGNER_0_BELAJAR.mp4" - splitting on "_" there would wrongly give
#      "SIGNER" as the label (flagged as a known gap in PROJECT_CONTEXT.md §5.4).
DATASET_DIR = "dataset"
VIDEO_EXTENSIONS = (".mp4")


def load_labels(dataset_dir):
    """
    Builds a {filename: label} map for every video in dataset_dir.

    Priority:
      1. labels.csv (filename,label), if present.
      2. Fallback: label = filename.split("_")[0]
         (works for WORD_NN.mp4; breaks on word-LAST filenames such as
         "SIGNER_0_BELAJAR.mp4" - provide labels.csv for those folders).
    """
    labels_csv_path = os.path.join(dataset_dir, "labels.csv")
    label_map = {}

    if os.path.isfile(labels_csv_path):
        with open(labels_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_map[row["filename"]] = row["label"]
        return label_map

    for file_name in sorted(os.listdir(dataset_dir)):
        if file_name.lower().endswith(VIDEO_EXTENSIONS):
            label_map[file_name] = file_name.split("_")[0]

    return label_map


def frames_extraction(video_path):
    """
    Extracts SEQUENCE_LENGTH frames, resized and normalized to [0, 1].

    Short clips (2-3s) make CAP_PROP_FRAME_COUNT / CAP_PROP_POS_FRAMES-based
    seeking unreliable (frame count is often mis-reported, and seeking can
    silently fail to land on a readable frame) - that's what was causing
    "Skipped - only 14/16 frames read". To avoid that, this reads every
    frame sequentially (no seeking) into a buffer first, then:
      - if the clip actually has >= SEQUENCE_LENGTH frames, samples
        SEQUENCE_LENGTH indices evenly spaced across the frames that were
        genuinely read;
      - if it has fewer (very short clip), uses all of them and forward-fills
        (repeats the last frame) up to SEQUENCE_LENGTH - mirroring the
        forward-fill used for gaps in the active-span sampling in
        dataset.py, so a short clip is padded rather than dropped.

    This always returns exactly SEQUENCE_LENGTH frames unless the video
    couldn't be read at all (returns an empty list in that case).
    """
    all_frames = []
    video_reader = cv2.VideoCapture(video_path)

    while True:
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        all_frames.append(normalized_frame)

    video_reader.release()

    total_read = len(all_frames)
    if total_read == 0:
        return []

    if total_read >= SEQUENCE_LENGTH:
        indices = np.linspace(0, total_read - 1, SEQUENCE_LENGTH).astype(int)
        frames_list = [all_frames[i] for i in indices]
    else:
        # Forward-fill: pad with the last frame instead of dropping the clip.
        frames_list = list(all_frames) + [all_frames[-1]] * (SEQUENCE_LENGTH - total_read)

    return frames_list


def create_dataset(dataset_dir=DATASET_DIR):
    """
    Reads every video directly inside dataset_dir (flat structure - no
    class subfolders), extracts a fixed-length frame sequence per video,
    and returns feature/label arrays plus the class list discovered from
    the labels (sorted, so class order is deterministic across runs).
    """
    label_map = load_labels(dataset_dir)
    if not label_map:
        raise ValueError(
            f"Tidak ada video ditemukan di '{dataset_dir}' "
            f"(diharapkan file WORD_NN.mp4 atau labels.csv)."
        )

    classes_list = sorted(set(label_map.values()))
    class_to_index = {name: idx for idx, name in enumerate(classes_list)}

    features = []
    labels = []

    for file_name, label in label_map.items():
        print(f"Extracting data: {file_name} (label: {label})")
        video_file_path = os.path.join(dataset_dir, file_name)
        frames = frames_extraction(video_file_path)

        if len(frames) == SEQUENCE_LENGTH:
            features.append(frames)
            labels.append(class_to_index[label])
        else:
            # Only happens if the video couldn't be opened/read at all.
            print(f"  Skipped - could not read any frames from this file")

    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels, classes_list


if __name__ == "__main__":
    features, labels, classes_list = create_dataset(DATASET_DIR)
    print(f"Classes found ({len(classes_list)}): {classes_list}")

    one_hot_encoded_labels = to_categorical(labels, num_classes=len(classes_list))

    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        one_hot_encoded_labels,
        test_size=0.25,
        shuffle=True,
        stratify=labels,
        random_state=seed_constant,
    )

    print(f"Train: {features_train.shape}, Test: {features_test.shape}")