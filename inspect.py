"""
inspect_model_input.py - shows what the model ACTUALLY sees for one video.

Usage:
  python inspect_model_input.py --video dataset/BELAJAR_01.mp4
  python inspect_model_input.py --video dataset/BELAJAR_01.mp4 \
        --model pose_landmark_model_XXXX.h5 --dataset_dir dataset

Outputs (prefix = --out, default "inspect_<videoname>"):
  <prefix>_1_overlay.png    raw frames + MediaPipe skeleton, on the exact frames sampled
  <prefix>_2_frames64.png   the 64x64 frames that go into VGG16 (fusion / LRCN models)
  <prefix>_3_skeleton.png   the landmark vectors redrawn as stick figures = the ONLY thing
                            the landmark model sees (no pixels, no background)
  <prefix>_4_heatmap.png    the raw [T x 184] matrix fed to the LSTM
  <prefix>_5_importance.png (only with --model) prediction + which frames mattered
  + a console report with detection stats and a per-frame table.
"""
import argparse
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import (
    _read_and_detect, _select_indices, _forward_fill, load_labels,
    LANDMARK_DIM, POSE_DIM, HAND_DIM, _LEFT_FLAG_POS, _RIGHT_FLAG_POS,
    SEQUENCE_LENGTH, DATASET_DIR,
)

# ============================== CONFIG ==============================
VIDEO_PATH = "dataset/BERANGKAT_01.mp4"   
MODEL_PATH = None                        
DATASET_DIR_FOR_CLASSES = DATASET_DIR    
T = SEQUENCE_LENGTH                      
OUT_PREFIX = None                       
# ====================================================================
 
# Standard MediaPipe hand topology (21 points)
HAND_CONN = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
             (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
             (18, 19), (19, 20), (0, 17)]
# pose vector order: 0 nose, 1 Lshoulder, 2 Rshoulder, 3 Lelbow, 4 Relbow, 5 Lwrist, 6 Rwrist
POSE_CONN = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]
 
C_POSE, C_LEFT, C_RIGHT = (255, 200, 0), (0, 220, 0), (0, 140, 255)  # BGR
 
 
def _montage(tiles, cols):
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
        
    return np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
 
 
# ---------------------------------------------------------------- 1. overlay
def save_overlay(full_frames, raw, indices, filled, path, cols=5):    
    draw, hol = mp.solutions.drawing_utils, mp.solutions.holistic
    tiles = []

    for t, i in enumerate(indices):
        img, r = full_frames[i].copy(), raw[i]

        draw.draw_landmarks(img, r.pose_landmarks, hol.POSE_CONNECTIONS)
        draw.draw_landmarks(img, r.left_hand_landmarks, hol.HAND_CONNECTIONS)
        draw.draw_landmarks(img, r.right_hand_landmarks, hol.HAND_CONNECTIONS)

        img = cv2.resize(img, (240, 180))
        flags = ("L" if r.left_hand_landmarks else "-") + ("R" if r.right_hand_landmarks else "-")
        pose = "P" if r.pose_landmarks else "-"
        col = (0, 0, 255) if filled[t] else (0, 255, 0)

        cv2.putText(img, f"t{t} f{i} {pose}{flags}" + (" FILLED" if filled[t] else ""),
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        tiles.append(img)

    cv2.imwrite(path, _montage(tiles, cols))
 
 
# --------------------------------------------------------- 2. 64x64 frames
def save_frames64(frame_seq, path, cols=5, scale=3):
    tiles = []

    for t, f in enumerate(frame_seq):
        img = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"t{t}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        tiles.append(img)

    cv2.imwrite(path, _montage(tiles, cols))
 
 
# ------------------------------------------------------ 3. stick figures
def _to_px(xy, w, h, xr=(-3.0, 3.0), yr=(-1.8, 3.6)):
    x = (xy[0] - xr[0]) / (xr[1] - xr[0]) * w
    y = (xy[1] - yr[0]) / (yr[1] - yr[0]) * h

    return int(x), int(y)
 
 
def draw_skeleton(vec, w=160, h=200):
    # Draw 1 landmark vector (body-normalized part) as a stick figure.
    img = np.full((h, w, 3), 30, np.uint8)
    pose = vec[:POSE_DIM].reshape(7, 2)

    for a, b in POSE_CONN:
        cv2.line(img, _to_px(pose[a], w, h), _to_px(pose[b], w, h), C_POSE, 2)

    cv2.circle(img, _to_px(pose[0], w, h), 5, C_POSE, -1)          # nose

    for k in (1, 2):
        cv2.circle(img, _to_px(pose[k], w, h), 3, (255, 255, 255), -1)  # shoulders
 
    for start, color in ((POSE_DIM, C_LEFT), (POSE_DIM + HAND_DIM, C_RIGHT)):
        hv = vec[start:start + HAND_DIM]

        if hv[-1] != 1.0:
            continue

        pts = hv[:42].reshape(21, 2)                                # body-normalized

        for a, b in HAND_CONN:
            cv2.line(img, _to_px(pts[a], w, h), _to_px(pts[b], w, h), color, 1)

        for p in pts:
            cv2.circle(img, _to_px(p, w, h), 1, (255, 255, 255), -1)

    return img
 
 
def save_skeleton_montage(lm_seq, path, cols=5):
    tiles = []

    for t, v in enumerate(lm_seq):
        img = draw_skeleton(v)
        lr = ("L" if v[_LEFT_FLAG_POS] == 1 else "-") + ("R" if v[_RIGHT_FLAG_POS] == 1 else "-")
        cv2.putText(img, f"t{t} {lr}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        tiles.append(img)

    cv2.imwrite(path, _montage(tiles, cols))
 
 
# --------------------------------------------------------------- 4. heatmap
def save_heatmap(lm_seq, path):
    fig, ax = plt.subplots(figsize=(14, 5))

    im = ax.imshow(lm_seq.T, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)

    for x in (POSE_DIM - 0.5, POSE_DIM + HAND_DIM - 0.5):
        ax.axhline(x, color="k", lw=1.5)

    ax.set_yticks([POSE_DIM / 2, POSE_DIM + HAND_DIM / 2, POSE_DIM + HAND_DIM * 1.5])
    ax.set_yticklabels(["pose (14)", "left hand (85)", "right hand (85)"])
    ax.set_xlabel("timestep t")
    ax.set_title("Landmark matrix fed to the LSTM (rows of zeros = hand not detected)")

    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130)

    plt.close(fig)
 
# ------------------------------------------------- 5. prediction + importance
def _model_kind(model):
    shapes = [tuple(i.shape) for i in model.inputs]
    if len(shapes) == 2:
        return "fusion"
    
    return "frames" if len(shapes[0]) == 5 else "landmark"

 
def _predict(model, kind, fr, lm):
    if kind == "fusion":
        return model.predict([fr, lm], verbose=0)
    
    return model.predict(fr if kind == "frames" else lm, verbose=0)
 
 
def occlusion_importance(model, frame_seq, lm_seq, target):

    kind = _model_kind(model)
    T = len(lm_seq)
    frs, lms = [frame_seq.copy()], [lm_seq.copy()]

    for t in range(T):
        f2, l2 = frame_seq.copy(), lm_seq.copy()
        f2[t], l2[t] = frame_seq.mean(0), lm_seq.mean(0)

        frs.append(f2)
        lms.append(l2)

    probs = _predict(model, kind, np.stack(frs), np.stack(lms))

    return probs[0], probs[0][target] - probs[1:, target]
 

def save_importance(probs, classes, importance, path):
    order = np.argsort(-probs)[:5]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))

    a1.barh([classes[i] for i in order][::-1], probs[order][::-1])
    a1.set_title("Top-5 probabilities")
    a1.set_xlim(0, 1)
    a2.bar(range(len(importance)), importance,
           color=["tab:red" if v > 0 else "tab:gray" for v in importance])
    
    a2.set_title("Frame importance (drop in P(pred) when frame is removed)")
    a2.set_xlabel("timestep t")

    fig.tight_layout()
    fig.savefig(path, dpi=130)

    plt.close(fig)
 
 
def main():
    video, model_path = VIDEO_PATH, MODEL_PATH
    prefix = OUT_PREFIX or f"inspect_{os.path.splitext(os.path.basename(video))[0]}"
 
    small, vectors, full, raw = _read_and_detect(video, keep_full_frames=True)

    if not small:
        raise SystemExit("Video tidak bisa dibaca.")
    
    indices = _select_indices(vectors, T)

    if indices is None:
        raise SystemExit("Pose tubuh tidak terdeteksi sama sekali -> model tidak akan melihat apa pun.")
 
    filled = [vectors[i] is None for i in indices]
    lm_seq = _forward_fill(vectors, indices)
    frame_seq = np.stack([small[i] for i in indices]).astype(np.float32)
 
    # ---- console report
    n = len(small)

    pose_n = sum(r.pose_landmarks is not None for r in raw)
    lh_n = sum(r.left_hand_landmarks is not None for r in raw)
    rh_n = sum(r.right_hand_landmarks is not None for r in raw)
    active = [i for i, v in enumerate(vectors) if v is not None and
              (v[_LEFT_FLAG_POS] == 1 or v[_RIGHT_FLAG_POS] == 1)]
    
    print(f"\nVideo: {video}  ({n} frames)")
    print(f"Pose detected      : {pose_n}/{n} ({100 * pose_n / n:.0f}%)")
    print(f"Left hand detected : {lh_n}/{n} ({100 * lh_n / n:.0f}%)")
    print(f"Right hand detected: {rh_n}/{n} ({100 * rh_n / n:.0f}%)")

    if active:
        print(f"Active span        : frames {active[0]}..{active[-1]}  "
              f"(model ignores the other {n - (active[-1] - active[0] + 1)} idle frames)")
    else:
        print("Active span        : NO hand ever detected -> falling back to pose-only span!")

    print(f"Filled (forward-fill) timesteps: {sum(filled)}/{len(filled)}")
    print(f"\n{'t':>3} {'frame':>5} {'L':>2} {'R':>2} {'fill':>4} | pose: nose(x,y)  Lwrist(x,y)  Rwrist(x,y)")

    for t, (i, v) in enumerate(zip(indices, lm_seq)):
        p = v[:POSE_DIM].reshape(7, 2)
        print(f"{t:>3} {i:>5} {'Y' if v[_LEFT_FLAG_POS] else '-':>2} "
              f"{'Y' if v[_RIGHT_FLAG_POS] else '-':>2} {'*' if filled[t] else '':>4} | "
              f"({p[0, 0]:+.2f},{p[0, 1]:+.2f})  ({p[5, 0]:+.2f},{p[5, 1]:+.2f})  ({p[6, 0]:+.2f},{p[6, 1]:+.2f})")

    # ---- images
    save_overlay(full, raw, indices, filled, f"out/{prefix}_1_overlay.png")
    save_frames64(frame_seq, f"out/{prefix}_2_frames64.png")
    save_skeleton_montage(lm_seq, f"out/{prefix}_3_skeleton.png")
    save_heatmap(lm_seq, f"out/{prefix}_4_heatmap.png")

    print(f"\nSaved: out/{prefix}_1_overlay.png, out/{prefix}_2_frames64.png, out/{prefix}_3_skeleton.png, out/{prefix}_4_heatmap.png")
 
    # ---- optional model analysis
    if model_path:
        custom = {"_vgg16_preprocess_from_unit_range": _vgg16_preprocess_from_unit_range}

        try:
            model = load_model(model_path, compile=False, custom_objects=custom)
        except Exception:  # Keras 3 refuses Lambda layers unless safe_mode is off
            model = load_model(model_path, compile=False, custom_objects=custom, safe_mode=False)

        classes = sorted(set(load_labels(DATASET_DIR_FOR_CLASSES).values())) \
            if os.path.isdir(DATASET_DIR_FOR_CLASSES) else None
        n_out = model.output_shape[-1]
        if classes is None or len(classes) != n_out:
            classes = [f"class_{i}" for i in range(n_out)]
 
        kind = _model_kind(model)
        probs = _predict(model, kind, frame_seq[None], lm_seq[None])[0]
        pred = int(np.argmax(probs))

        print(f"\nModel type: {kind}")
        print(f"Prediction: {classes[pred]} ({probs[pred] * 100:.1f}%)")

        for i in np.argsort(-probs)[:5]:
            print(f"  {classes[i]:<15s} {probs[i] * 100:6.2f}%")

        _, imp = occlusion_importance(model, frame_seq, lm_seq, pred)

        save_importance(probs, classes, imp, f"out/{prefix}_5_importance.png")

        top = np.argsort(-imp)[:3]

        print("Most influential timesteps:", ", ".join(f"t{t} (-{imp[t] * 100:.1f}%)" for t in top))
        print(f"Saved: out/{prefix}_5_importance.png")
 
 
if __name__ == "__main__":
    main()