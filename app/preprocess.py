import os
import cv2
import glob
import random
import numpy as np
import datetime as dt
import tensorflow as tf
import mediapipe as mp
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input


seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

image_height = 112
image_width = 112
sequence_length = 20
dataset_dir = "dataset"


def load_labels(dataset_dir):
    label_map = {}

    for file_name in sorted(os.listdir(dataset_dir)):
        if file_name.endswith(".mp4"):
            label_map[file_name] = file_name.split("_")[0]

    return label_map


def frames_extraction(video_path):
    all_frames = []
    video_reader = cv2.VideoCapture(video_path)

    while True:
        success, frame = video_reader.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (image_width, image_height))
        normalized_frame = resized_frame / 255.0
        all_frames.append(normalized_frame)

    video_reader.release()
    total_read = len(all_frames)

    if total_read >= sequence_length:
        indicies = np.linspace(0, total_read - 1, sequence_length).astype(int)
        frames_list = [all_frames[i] for i in indicies]
    else:
        frame_list = list(all_frames) + [all_frames[-1]] * (sequence_length - total_read)
        return frames_list  # this one


def create_dataset(dir = dataset_dir):
    label_map = load_labels(dir)
    classes_list = sorted(set(label_map.values()))

    classes_to_index = {}

    for i in range(len(classes_list)):
        name = classes_list[i]
        classes_to_index[name] = i

    features = []
    labels = []

    for file_name, label in label_map.items():
        print(f"extracting data: {file_name} (label: {label})")
        video_file_path = os.path.join(dir, file_name)
        frames = frames_extraction(video_file_path)

        if len(frames) == sequence_length:
            features.append(frames)
            labels.append(classes_to_index[label])
        else:
            print(f"skipped {file_name}")

    features = np.asarray(features)
    labels = np.array(labels)

    return features, labels, classes_list

pose_idx = [0, 11, 12, 13, 14, 15, 16] # face, left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist
pose_dim = len(pose_idx) * 2
hand_dim = 21 * 2 * 2 + 1

landmark_dim = pose_dim + 2 * hand_dim
left_flag_pos = pose_dim + hand_dim - 1
right_flag_pos = landmark_dim - 1

shoulder_vis_threshold = 0.3
max_process_width = 640


def new_holistic():
    return mp.solutions.holistic.Holistic(static_image_mode = False, model_complexity = 1, smooth_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)


def lm_xy(landmark_list, idx = None):
    pts = landmark_list.landmark

    if idx is not None:
        pts_baru = []

        for i in idx:
            item = pts[i]
            pts_baru.append(item)

        pts = pts_baru

    return np.array([[p.x, p.y] for p in pts], dtype = np.float32)


def hand_vector(hand_landmarks, center, scale):
    if hand_landmarks is None:
        return np.zeros(hand_dim, dtype = np.float32)

    xy = lm_xy(hand_landmarks)
    body = (xy - center) / scale
    rel = xy - xy[0]
    extent = np.abs(rel).max()

    if extent > 1e-6:
        rel = rel / extent

    return np.concatenate([body.flatten(), rel.flatten(), [1.0]]).astype(np.float32)


def results_to_vector(results, last_state):
    pose_ok = False
    if results.pose_landmarks is not None:
        lm = results.pose_landmarks.landmark

        if lm[11].visibility > shoulder_vis_threshold and lm[12].visibility > shoulder_vis_threshold:
            pts = lm_xy(results.pose_landmarks, pose_idx)
            l_shoulder, r_shoulder = pts[1], pts[2]
            scale = float(np.linalg.norm(l_shoulder - r_shoulder))

            if scale > 1e-3:
                center = (l_shoulder + r_shoulder) / 2.0
                pose_vec = ((pts - center) / scale).flatten().astype(np.float32)
                last_state = (center, scale, pose_vec)
                pose_ok = True

    if last_state is None:
        return None, last_state

    center, scale, pose_vec = last_state
    left = hand_vector(results.left_hand_landmarks, center, scale)
    right = hand_vector(results.right_hand_landmarks, center, scale)

    return np.concatenate([pose_vec, left, right]).astype(np.float32), last_state


def select_indices(vectors, sequence = sequence_length):
    n = len(vectors)
    valid = []

    for i in range(len(vectors)):
        v = vectors[i]
        if v is not None:
            valid.append(i)

    if not valid:
        return None
    
    active = []

    for i in valid:
        vector_now = vectors[i]
        
        # Ambil nilai penanda kiri dan kanan
        left_flag = vector_now[left_flag_pos]
        right_flag = vector_now[right_flag_pos]
        
        if left_flag == 1.0 or right_flag == 1.0:
            active.append(i)

    span = active if active else valid
    span_start, span_end = span[0], span[-1]

    return np.linspace(span_start, span_end, sequence).round().astype(int).tolist()
    

def forward_fill(vectors, indices):
    picked = []

    for i in indices:
        v = vectors[i]

        if v is None:
            j = i

            while j >= 0 and vectors[j] is None:
                j -= 1

            if j < 0:
                j = 1

                while j < len(vectors) and vectors[j] is None:
                    j += 1

            v = vectors[j]
        picked.append(v)
    return np.stack(picked)


def process_frame_live(frame, holistic, last_state):
    height, width = frame.shape[:2]
    proc = frame

    if width > max_process_width:
        proc = cv2.resize(frame, (max_process_width, int(height * max_process_width / width)))

    results = holistic.process(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    vector, last_state = results_to_vector(results, last_state)
    small = cv2.resize(frame, (image_width, image_height)) / 255.0

    return small.astype(np.float32), vector, last_state, results


def read_and_detect(video_path, keep_full_frames = False):
    small_frames, vectors, raw = [], [], []
    full_frames = []
    cap = cv2.VideoCapture(video_path)
    last_state = None

    with new_holistic() as holistic:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            small, vector, last_state, results = process_frame_live(frame, holistic, last_state)
            vectors.append(vector)
            small_frames.append(small)

            if keep_full_frames: # kemungkinan bisa dihapus
                height, width = frame.shape[:2]
                proc = frame

                if width > max_process_width:
                    proc = cv2.resize(frame, (max_process_width, int(height * max_process_width / width)))

                full_frames.append(proc)
                raw.append(results)

    cap.release()
    return small_frames, vectors, full_frames, raw


def landmarks_extraction(video_path, sequence = sequence_length):
    small_frames, vectors, _, _ = read_and_detect(video_path)

    if not small_frames:
        return None, None

    indices = select_indices(vectors, sequence)

    if indices is None:
        return None, None

    landmark_seq = forward_fill(vectors, indices)
    frame_seq = np.stack([small_frames[i] for i in indices]).astype(np.float32)

    return frame_seq, landmark_seq


def create_dataset_pose(dir = dataset_dir, cache_path = "pose_cache_T20.npz", force_recompute = False):
    if cache_path is None:
        cache_path = "pose_cache_T20.npz"

    label_map = load_labels(dir)

    classes_list = sorted(set(label_map.values()))

    if os.path.exists(cache_path) and not force_recompute:
        data = np.load(cache_path, allow_pickle = True)

        if list(data["classes"]) == classes_list and data["landmarks"].shape[-1] == landmark_dim:
            print(f"using cache file: {cache_path}")

            return data["frames"], data["landmarks"], data["labels"], classes_list

    classes_to_index = {}

    for i in range(len(classes_list)):
        name = classes_list[i]
        classes_to_index[name] = i

    frames_all, landmarks_all, labels = [], [], []

    for file_name, label in label_map.items():
        print(f"pose and hands extraction: {file_name} (label: {label})")
        frames, landmarks = landmarks_extraction(os.path.join(dir, file_name))

        if frames is None:
            print("skipped (no pose / hands detected)")
            continue

        frames_all.append(frames)
        landmarks_all.append(landmarks)
        labels.append(classes_to_index[label])

    frames_all = np.asarray(frames_all, dtype = np.float32)
    landmarks_all = np.asarray(landmarks_all, dtype = np.float32)
    labels = np.array(labels)

    np.savez_compressed(cache_path, frames = frames_all, landmarks = landmarks_all, labels = labels, classes = np.array(classes_list))
    print(f"saved cache file: {cache_path}")

    return frames_all, landmarks_all, labels, classes_list


def visualize_video_pose(video_path, out_path = "pose_debug.png", sequence = sequence_length, cols = 5):

    mediapipe_draw = mp.solutions.drawing_utils
    mediapipe_holistic = mp.solutions.holistic

    _, vectors, full_frames, raw = read_and_detect(video_path, keep_full_frames = True)
    indices = select_indices(vectors, sequence)

    if indices is None:
        print("no pose / hands detected")
        return

    titles = []

    for t, i in enumerate(indices):
        image = full_frames[i].copy()
        
        r = raw[i]

        mediapipe_draw.draw_landmarks(image, r.pose_landmarks, mediapipe_holistic.POSE_CONNECTIONS)
        mediapipe_draw.draw_landmarks(image, r.left_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS)
        mediapipe_draw.draw_landmarks(image, r.right_hand_landmarks, mediapipe_holistic.HAND_CONNECTIONS)

        image = cv2.resize(image, (240, 180))

        if r.left_hand_landmarks:
            left_status = "L"
        else:
            left_status = "-"

        if r.right_hand_landmarks:
            right_status = "R"
        else:
            right_status = "-"

        det = left_status + right_status

        text_info = f"t={t} f={i} {det}" #time, frames, detected hands
        cv2.putText(image, text_info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        titles.append(image)

    while len(titles) % cols:
        titles.append(np.zeros_like(titles[0]))

    rows = [np.hstack(titles[k:k + cols]) for k in range(0, len(titles), cols)]
    cv2.imwrite(out_path, np.vstack(rows))

    print(f"saved debug image: {out_path}")


def vgg_preprocessing(x):
    return preprocess_input(x * 255.0)


if __name__ == "__main__":
    frames, landmarks, labels, classes_list = create_dataset_pose(dataset_dir)

    print(f"number of classes: {len(classes_list)}")
    print(f"list of classes: {classes_list}")
    print(f"frames shape: {frames.shape}")
    print(f"landmarks shape: {landmarks.shape}")

    print(" ================================================================ ")

    idx = np.arange(len(labels))
    idx_train, idx_test = train_test_split(idx, test_size = 0.25, shuffle = True, stratify = labels, random_state = seed_constant)

    print(f"NUMBER OF TRAIN: {len(idx_train)}")
    print(f"NUMBER OF TEST: {len(idx_test)}")