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

# BISINDO dataset layout: videos live directly inside DATASET_DIR (no per-class
# subfolders). Two supported naming schemes:
#   1. "WORD_NN.mp4"  (e.g. BELAJAR_01.mp4) -> label = "BELAJAR"
#   2. A "labels.csv" (columns: filename,label) inside DATASET_DIR, needed for
#      folders that don't follow WORD_NN (e.g. "SIGNER_0_BELAJAR.mp4").
DATASET_DIR = "dataset"
VIDEO_EXTENSIONS = (".mp4")


def load_labels(dataset_dir):
    """
    Builds a {filename: label} map for every video in dataset_dir.

    Priority:
      1. labels.csv (filename,label), if present.
      2. Fallback: label = filename.split("_")[0]
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
    (ORIGINAL, whole-frame only.) Extracts SEQUENCE_LENGTH frames, resized and
    normalized to [0, 1], sampled uniformly across the whole clip.
    Reads sequentially (no seeking) and forward-fills very short clips.
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
        frames_list = list(all_frames) + [all_frames[-1]] * (SEQUENCE_LENGTH - total_read)

    return frames_list


def create_dataset(dataset_dir=DATASET_DIR):
    """
    (ORIGINAL, whole-frame only.) Returns (features, labels, classes_list).
    Kept unchanged so existing notebooks keep working. For pose/hand landmarks
    use create_dataset_pose() below.
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
            print(f"  Skipped - could not read any frames from this file")

    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels, classes_list


# =====================================================================
#  POSE + HAND LANDMARK PIPELINE (MediaPipe Holistic)
# =====================================================================
# Per-frame feature vector (LANDMARK_DIM = 184):
#
#   [ pose (14) | left hand (85) | right hand (85) ]
#
#   pose  : 7 upper-body landmarks (nose, L/R shoulder, L/R elbow, L/R wrist),
#           (x, y) each, centred on the shoulder midpoint and divided by the
#           shoulder width -> invariant to camera distance / framing.
#   hand  : 21 landmarks x (x, y), given TWICE:
#             (a) body-normalized  (same centre/scale as the pose) -> WHERE the
#                 hand is relative to the body (chest, face, side, ...)
#             (b) wrist-relative, scaled by hand extent -> the hand SHAPE
#                 independent of position
#           + 1 detection flag (1.0 = hand detected, 0.0 = missing / zeros)
#
# This is what the skeleton overlay in the screenshot represents: body pose
# (arms/shoulders) plus both hand skeletons.

POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # nose, Lsh, Rsh, Lelb, Relb, Lwr, Rwr
POSE_DIM = len(POSE_IDX) * 2            # 14
HAND_DIM = 21 * 2 * 2 + 1               # 85
LANDMARK_DIM = POSE_DIM + 2 * HAND_DIM  # 184
_LEFT_FLAG_POS = POSE_DIM + HAND_DIM - 1
_RIGHT_FLAG_POS = LANDMARK_DIM - 1

SHOULDER_VIS_THRESHOLD = 0.3
MAX_PROCESS_WIDTH = 640  # downscale before MediaPipe for speed


def _new_holistic():
    import mediapipe as mp  # lazy import: only needed for the pose pipeline
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _lm_xy(landmark_list, idx=None):
    pts = landmark_list.landmark
    if idx is not None:
        pts = [pts[i] for i in idx]
    return np.array([[p.x, p.y] for p in pts], dtype=np.float32)


def _hand_vector(hand_landmarks, center, scale):
    if hand_landmarks is None:
        return np.zeros(HAND_DIM, dtype=np.float32)
    xy = _lm_xy(hand_landmarks)                      # [21, 2]
    body = (xy - center) / scale                     # position in body frame
    rel = xy - xy[0]                                 # wrist-relative shape
    extent = np.abs(rel).max()
    if extent > 1e-6:
        rel = rel / extent
    return np.concatenate([body.flatten(), rel.flatten(), [1.0]]).astype(np.float32)


def _results_to_vector(results, last_state):
    """
    Converts one MediaPipe Holistic result to a LANDMARK_DIM vector.
    last_state = (center, scale, pose_vec) of the last frame with a reliable
    pose; it is reused when the pose drops out, so hands can still be
    expressed in the body frame. Returns (vector or None, new_last_state).
    """
    pose_ok = False
    if results.pose_landmarks is not None:
        lm = results.pose_landmarks.landmark
        if lm[11].visibility >= SHOULDER_VIS_THRESHOLD and lm[12].visibility >= SHOULDER_VIS_THRESHOLD:
            pts = _lm_xy(results.pose_landmarks, POSE_IDX)  # [7, 2]
            l_sh, r_sh = pts[1], pts[2]
            scale = float(np.linalg.norm(l_sh - r_sh))
            if scale > 1e-3:
                center = (l_sh + r_sh) / 2.0
                pose_vec = ((pts - center) / scale).flatten().astype(np.float32)
                last_state = (center, scale, pose_vec)
                pose_ok = True

    if last_state is None:
        return None, last_state  # no body reference yet

    center, scale, pose_vec = last_state
    left = _hand_vector(results.left_hand_landmarks, center, scale)
    right = _hand_vector(results.right_hand_landmarks, center, scale)
    return np.concatenate([pose_vec, left, right]).astype(np.float32), last_state


def _select_indices(vectors, sequence_length=SEQUENCE_LENGTH):
    """
    Active-span sampling: find first/last frame where a hand is detected and
    sample `sequence_length` indices uniformly INSIDE that span (so idle
    lead-in / trail-out frames do not waste sequence slots). Falls back to the
    span of valid frames, then the whole clip.
    """
    n = len(vectors)
    valid = [i for i, v in enumerate(vectors) if v is not None]
    if not valid:
        return None
    active = [i for i in valid
              if vectors[i][_LEFT_FLAG_POS] == 1.0 or vectors[i][_RIGHT_FLAG_POS] == 1.0]
    span = active if active else valid
    start, end = span[0], span[-1]
    return np.linspace(start, end, sequence_length).round().astype(int).tolist()


def _forward_fill(vectors, indices):
    """Fills selected frames whose vector is None from the nearest earlier
    valid frame (or the nearest later one for leading gaps)."""
    picked = []
    for i in indices:
        v = vectors[i]
        if v is None:
            j = i
            while j >= 0 and vectors[j] is None:
                j -= 1
            if j < 0:
                j = i
                while j < len(vectors) and vectors[j] is None:
                    j += 1
            v = vectors[j]
        picked.append(v)
    return np.stack(picked)


def _read_and_detect(video_path, keep_full_frames=False):
    """Runs Holistic sequentially over every frame (tracking works because
    frames are processed in order). Returns small frames, vectors, and
    optionally the raw MediaPipe results + full frames (debug only)."""
    small_frames, vectors, raw = [], [], []
    full_frames = []
    cap = cv2.VideoCapture(video_path)
    last_state = None

    with _new_holistic() as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            proc = frame
            if w > MAX_PROCESS_WIDTH:
                proc = cv2.resize(frame, (MAX_PROCESS_WIDTH, int(h * MAX_PROCESS_WIDTH / w)))
            results = holistic.process(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))

            vec, last_state = _results_to_vector(results, last_state)
            vectors.append(vec)
            small_frames.append(cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT)) / 255.0)
            if keep_full_frames:
                full_frames.append(proc)
                raw.append(results)
    cap.release()
    return small_frames, vectors, full_frames, raw


def landmarks_extraction(video_path, sequence_length=SEQUENCE_LENGTH):
    """
    Returns (frames [T,H,W,3] in [0,1], landmarks [T,LANDMARK_DIM]) sampled
    with active-span sampling, or (None, None) if no body pose was ever found.
    """
    small_frames, vectors, _, _ = _read_and_detect(video_path)
    if not small_frames:
        return None, None
    indices = _select_indices(vectors, sequence_length)
    if indices is None:
        return None, None
    lm_seq = _forward_fill(vectors, indices)
    frame_seq = np.stack([small_frames[i] for i in indices]).astype(np.float32)
    return frame_seq, lm_seq


def create_dataset_pose(dataset_dir=DATASET_DIR, cache_path=None, force_recompute=False):
    """
    Pose + hand version of create_dataset().
    Returns (frames [N,T,H,W,3], landmarks [N,T,LANDMARK_DIM], labels [N], classes_list).
    Results are cached to an .npz because MediaPipe over 240 clips is slow.
    """
    if cache_path is None:
        cache_path = f"pose_cache_T{SEQUENCE_LENGTH}.npz"

    label_map = load_labels(dataset_dir)
    if not label_map:
        raise ValueError(f"Tidak ada video ditemukan di '{dataset_dir}'.")
    classes_list = sorted(set(label_map.values()))

    if os.path.isfile(cache_path) and not force_recompute:
        data = np.load(cache_path, allow_pickle=True)
        if list(data["classes"]) == classes_list and data["landmarks"].shape[-1] == LANDMARK_DIM:
            print(f"Loaded cache: {cache_path}")
            return data["frames"], data["landmarks"], data["labels"], classes_list
        print("Cache mismatch (classes/feature dim changed) -> recomputing.")

    class_to_index = {name: idx for idx, name in enumerate(classes_list)}
    frames_all, lm_all, labels = [], [], []

    for file_name, label in label_map.items():
        print(f"Extracting pose+hands: {file_name} (label: {label})")
        frames, lms = landmarks_extraction(os.path.join(dataset_dir, file_name))
        if frames is None:
            print("  Skipped - unreadable video or no body pose detected")
            continue
        frames_all.append(frames)
        lm_all.append(lms)
        labels.append(class_to_index[label])

    frames_all = np.asarray(frames_all, dtype=np.float32)
    lm_all = np.asarray(lm_all, dtype=np.float32)
    labels = np.array(labels)

    np.savez_compressed(cache_path, frames=frames_all, landmarks=lm_all,
                        labels=labels, classes=np.array(classes_list))
    print(f"Saved cache: {cache_path}")
    return frames_all, lm_all, labels, classes_list


def visualize_video_pose(video_path, out_path="pose_debug.png", sequence_length=SEQUENCE_LENGTH, cols=5):
    """
    Debug helper: draws the pose + hand skeleton (like the reference
    screenshot) on the exact frames that the model will receive, as a montage.
    Use it to verify detection before training.
    """
    import mediapipe as mp
    mp_draw = mp.solutions.drawing_utils
    mp_hol = mp.solutions.holistic

    _, vectors, full_frames, raw = _read_and_detect(video_path, keep_full_frames=True)
    indices = _select_indices(vectors, sequence_length)
    if indices is None:
        print("No body pose detected in this video.")
        return

    tiles = []
    for t, i in enumerate(indices):
        img = full_frames[i].copy()
        r = raw[i]
        mp_draw.draw_landmarks(img, r.pose_landmarks, mp_hol.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(img, r.left_hand_landmarks, mp_hol.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(img, r.right_hand_landmarks, mp_hol.HAND_CONNECTIONS)
        img = cv2.resize(img, (240, 180))
        det = ("L" if r.left_hand_landmarks else "-") + ("R" if r.right_hand_landmarks else "-")
        cv2.putText(img, f"t={t} f={i} {det}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        tiles.append(img)

    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
    cv2.imwrite(out_path, np.vstack(rows))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    frames, landmarks, labels, classes_list = create_dataset_pose(DATASET_DIR)
    print(f"Classes found ({len(classes_list)}): {classes_list}")
    print(f"frames: {frames.shape}, landmarks: {landmarks.shape}")

    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.25, shuffle=True, stratify=labels, random_state=seed_constant
    )
    print(f"Train: {len(idx_train)}, Test: {len(idx_test)}")