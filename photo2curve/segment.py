from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent          # fourier-epicycles/
IMG_PATH = Path(__file__).with_name("images.jpg")      # photo2curve/images.jpg
MODEL_PATH = ROOT / "models" / "u2net.onnx"

SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # R, G, B
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_image(path) -> np.ndarray:
    """path to image file -> (H, W, 3) BGR uint8"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img

def load_session(model_path) -> ort.InferenceSession:
    with open(model_path, "rb") as f:
        return ort.InferenceSession(f.read(), providers=["CPUExecutionProvider"])

def preprocess(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) BGR uint8 -> (1, 3, 320, 320) float32"""
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #RGB 순서로 변경
    x = cv2.resize(x, (SIZE, SIZE)) # 320x320으로 리사이즈
    x = x.astype(np.float32) # float32로 변환
    x = x / max(x.max(), 1e-6)             # divide by max value (U²-Net original method)
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]        # (H,W,C) -> (1,C,H,W)
    return x.astype(np.float32)

def predict(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    """Run the model. Return the first output (fused) as (320, 320)"""
    name = session.get_inputs()[0].name
    if x.dtype != np.float32:
        raise ValueError(f"Input data type is not float32: {x.dtype}")
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3 or x.shape[2] != SIZE or x.shape[3] != SIZE:
        raise ValueError(f"Input data shape is not (1,3,320,320): {x.shape}")
    outputs = session.run(None, {name: x})
    return outputs[0][0, 0]

def postprocess(prob: np.ndarray, height: int, width: int, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(320,320) probability -> original size probability matrix and 0/255 mask"""
    prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8) #normalize to [0,1]
    prob = cv2.resize(prob, (width, height))      # (width, height) order
    mask = ((prob > threshold) * 255).astype(np.uint8) #threshold 기준으로 0/255 mask 생성 && dtype uint8로 변환
    return prob, mask

