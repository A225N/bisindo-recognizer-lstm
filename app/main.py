import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# preprocessing_w_mediapipe.py harus satu folder dengan main.py: logika landmark
# dipakai ulang persis seperti training, jadi tidak bisa menyimpang.
from preprocess import (
    sequence_length as T,
    forward_fill,
    new_holistic,
    results_to_vector,
    select_indices,
    load_labels,
)

MODEL_PATH = os.getenv("MODEL_PATH", "model_09_10_2026(13-26-44).h5")
DATASET_DIR = os.getenv("DATASET_DIR", "../dataset")  # folder training, sumber daftar kelas
MAX_FRAMES = 150


@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model(MODEL_PATH, compile=False)
    # sama dengan create_dataset_pose(): indeks kelas = sorted(set(label))
    classes = sorted(set(load_labels(DATASET_DIR).values()))
    n = app.state.model.output_shape[-1]
    assert len(classes) == n, f"model {n} kelas, tapi '{DATASET_DIR}' berisi {len(classes)}: {classes}"
    app.state.classes = classes
    yield


app = FastAPI(lifespan=lifespan)


class Req(BaseModel):
    frames: list[str]


def extract(frames):
    """Frame JPEG berurutan -> landmark [1, T, 184], sama seperti landmarks_extraction() saat training."""
    vectors, last_state = [], None
    with new_holistic() as holistic:  # satu klip = satu sesi tracking
        for f in frames:
            buf = np.frombuffer(base64.b64decode(f.split(",")[-1]), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, sama seperti cv2.VideoCapture
            if img is None:
                raise ValueError("frame bukan gambar valid")
            results = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            vec, last_state = results_to_vector(results, last_state)
            vectors.append(vec)
    idx = select_indices(vectors, T)
    if idx is None:
        raise ValueError("pose tubuh tidak terdeteksi, pastikan bahu terlihat")
    return forward_fill(vectors, idx)[None]


@app.post("/predict")
def predict(req: Req):
    if not T <= len(req.frames) <= MAX_FRAMES:
        raise HTTPException(422, f"butuh {T}-{MAX_FRAMES} frame, dapat {len(req.frames)}")
    try:
        x = extract(req.frames)
    except (ValueError, cv2.error) as e:
        raise HTTPException(422, str(e))
    p = app.state.model.predict(x, verbose=0)[0]
    top = np.argsort(p)[::-1][:3]
    return [{"label": app.state.classes[i], "prob": float(p[i])} for i in top]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")