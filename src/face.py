"""Face detection and expression scoring; no identity recognition or storage."""
from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/face/emotion-ferplus-8.onnx"
FACE_LABELS = ("neutral", "happy", "surprised", "sad", "angry", "disgust", "fearful", "contempt")


@lru_cache(maxsize=1)
def load_face_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Face model is missing. Run python scripts/download_face_model.py")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def decode_image(payload):
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be decoded")
    return image


@lru_cache(maxsize=1)
def _face_detector():
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV face detector is unavailable")
    return detector


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = _face_detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return sorted([tuple(map(int, box)) for box in boxes], key=lambda b: b[2] * b[3], reverse=True)


def score_frame(frame, session):
    boxes = detect_faces(frame)
    if not boxes:
        return None
    x, y, width, height = boxes[0]  # Largest face only; never merge people.
    gray = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32).reshape(1, 1, 64, 64)
    logits = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0], dtype=np.float64).reshape(-1)
    if logits.shape != (8,) or not np.isfinite(logits).all():
        raise ValueError("Facial model returned invalid scores")
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    return {"probabilities": dict(zip(FACE_LABELS, map(float, probs))), "face_count": len(boxes), "box": boxes[0]}


def aggregate_frames(frames, session):
    scored = [score_frame(frame, session) for frame in frames]
    valid = [item for item in scored if item is not None]
    if not valid:
        return None
    average = np.mean([[item["probabilities"][label] for label in FACE_LABELS] for item in valid], axis=0)
    probs = dict(zip(FACE_LABELS, map(float, average)))
    return {"emotion": max(probs, key=probs.get), "probabilities": probs,
            "frames_sampled": len(scored), "frames_with_face": len(valid),
            "multi_face_frames": sum(item["face_count"] > 1 for item in valid)}
