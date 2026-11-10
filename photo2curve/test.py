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
    """한글 경로에서도 읽히도록 imdecode를 사용. 반환: (H, W, 3) BGR uint8"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def load_session(model_path) -> ort.InferenceSession:
    """바이트로 읽어서 넘기면 한글 경로 문제를 피할 수 있다."""
    with open(model_path, "rb") as f:
        return ort.InferenceSession(f.read(), providers=["CPUExecutionProvider"])


def preprocess(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) BGR uint8 -> (1, 3, 320, 320) float32"""
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (SIZE, SIZE))
    x = x.astype(np.float32)
    x = x / max(x.max(), 1e-6)            # 이미지 최댓값으로 나눔 (U²-Net 원본 방식)
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]        # (H,W,C) -> (1,C,H,W)
    return x.astype(np.float32)


def predict(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    """모델 실행. 7개 출력 중 첫 번째(fused)를 (320, 320)으로 반환"""
    name = session.get_inputs()[0].name
    if x.dtype != np.float32:
        raise ValueError(f"입력 데이터 타입이 float32가 아닙니다: {x.dtype}")
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3 or x.shape[2] != SIZE or x.shape[3] != SIZE:
        raise ValueError(f"입력 데이터 shape이 (1,3,320,320)이 아닙니다: {x.shape}")
    outputs = session.run(None, {name: x})
    return outputs[0][0, 0]


def postprocess(prob: np.ndarray, height: int, width: int, threshold: float = 0.5):
    """(320,320) 확률 -> 원본 크기의 확률 행렬과 0/255 마스크"""
    prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
    prob = cv2.resize(prob, (width, height))      # (가로, 세로) 순서
    mask = ((prob > threshold) * 255).astype(np.uint8)
    return prob, mask


if __name__ == "__main__":
    img = load_image(IMG_PATH)
    h, w = img.shape[:2]
    print("img:", img.shape, img.dtype)

    session = load_session(MODEL_PATH)

    x = preprocess(img)
    print("input:", x.shape, x.dtype)

    prob_small = predict(session, x)
    print("output:", prob_small.shape, prob_small.min(), prob_small.max())

    prob, mask = postprocess(prob_small, h, w)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("image")
    axes[1].imshow(prob, cmap="gray")
    axes[1].set_title("probability")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("mask")
    for ax in axes:
        ax.axis("off")
    plt.show()