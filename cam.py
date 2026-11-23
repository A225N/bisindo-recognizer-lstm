from tensorflow.keras.models import load_model
import mediapipe as mp
import numpy as np
import time
import cv2

camera_index = 0
model_path = "model_09_10_2026(13-26-44).h5"
dataset_dir = "dataset"

countdown = 3.0
capture_window = 2.0
T = sequence_length

result_display = 2.5
confidence_threshold = 0.0
skeleton_overlay = True


def predict(model, frame_seq, landmark_seq):
    frame, landmark = frame_seq[None], landmark_seq[None]
    return model.predict(landmark, verbose = 0)[0]


def reload_model():
    custom = {}
    try:
        # from cnn_lstm import _vgg16_preprocess_from_unit_range
        custom["vgg_preprocessing"] = vgg_preprocessing
    except Exception:
        pass
    try:
        return load_model(model_path, compile=False, custom_objects=custom)
    except Exception:
        # Keras 3 menolak layer Lambda kecuali safe_mode dimatikan.
        return load_model(model_path, compile=False, custom_objects=custom, safe_mode=False)

    # return load_model(model_path, compile = False, custom_objects = custom, safe_mode = False)


def put_text(frame, text, y, scale = 1.0, color = (255, 255, 255), thick = 2):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_skeleton(display, results):
    draw, hol = mp.solutions.drawing_utils, mp.solutions.holistic

    draw.draw_landmarks(display, results.pose_landmarks, hol.POSE_CONNECTIONS)
    draw.draw_landmarks(display, results.left_hand_landmarks, hol.HAND_CONNECTIONS)
    draw.draw_landmarks(display, results.right_hand_landmarks, hol.HAND_CONNECTIONS)


if __name__ == "__main__":
    model = reload_model()
    num_out = model.output_shape[-1]
    classes = None
    
    classes = sorted(set(load_labels(dataset_dir).values()))

    if not classes or len(classes) != num_out:
        classes = [f"class_{i}" for i in range(num_out)]

    print(f"classes ({num_out}): {classes}")

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise SystemExit(f"camera index {camera_index} not found")

    state_idle, state_countdown, state_recording, state_result = range(4)
    state = state_idle
    state_start = 0.0
    result_text = []

    holistic = new_holistic()
    post_state = None
    rec_frames, rec_vectors = [], []

    print("tekan SPACE untuk merekam, 'Q' untuk keluar")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1) # mirror-ing
            now = time.time()
            display = frame.copy()

            if state == state_idle:
                put_text(display, "tekan SPASI untuk mulai", 40, 0.8, (0, 255, 0))

            elif state == state_countdown:
                if skeleton_overlay:
                    _, _, _, results = process_frame_live(frame, holistic, post_state)
                    draw_skeleton(display, results)

                remaining = countdown - (now - state_start)

                if remaining <= 0:
                    state, state_start = state_recording, now
                    rec_frames, rec_vectors = [], []
                    post_state = None
                else:
                    put_text(display, f"get ready... {remaining:0.1f}s", 60, 1.4, (0, 255, 255), 3)

            elif state == state_recording:
                small, vec, post_state, results = process_frame_live(frame, holistic, post_state)

                rec_frames.append(small)
                rec_vectors.append(vec)

                if skeleton_overlay:
                    draw_skeleton(display, results)

                elapsed = now - state_start
                remaining = capture_window - elapsed

                cv2.circle(display, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)
                put_text(display, f"recording {max(0.0, remaining):0.1f}s", 40, 0.9, (0, 0, 255))

                if remaining <= 0:
                    indices = select_indices(rec_vectors, T)

                    landmark_seq = landmark_sequence(rec_vectors, indices)
                    frame_seq = np.stack([rec_frames[i] for i in indices]).astype(np.float32)
                    probs = predict(model, frame_seq, landmark_seq)
                    order = np.argsort(-probs)[:3]
                    result_text = [f"{classes[i]}: {probs[i]:.1f}%" for i in order]

                    if probs[order[0]] < confidence_threshold:
                        result_text.insert(0, "(low confidence)")

                    state, state_start = state_result, now

            elif state == state_result:
                k = 0

                for line in result_text:
                    y_pos = 40 + (k * 35)

                    if k == 0:
                        scale = 1.1
                        color = (0, 255, 0)
                    else:
                        scale = 0.85
                        color = (255, 255, 255)

                    put_text(display, line, y_pos, scale, color)
                    
                    k = k + 1

                if now - state_start >= result_display:
                    state = state_idle

            cv2.imshow("cam test", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord(' ') and state == state_idle:
                state, state_start = state_countdown, now
    finally:
        holistic.close()
        cap.release()
        cv2.destroyAllWindows()