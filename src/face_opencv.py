"""OpenCV Zoo Progressive Teacher with YuNet landmarks and 112px alignment."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
import cv2
import numpy as np
import onnxruntime as ort

from .face import decode_image

MODEL_DIR = Path(__file__).resolve().parents[1] / "models/face"
MODEL_PATH = MODEL_DIR / "facial_expression_recognition_mobilefacenet_2022july.onnx"
DETECTOR_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
STANDARD_POINTS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                            [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
_detector_lock = Lock()


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists() or not DETECTOR_PATH.exists():
        raise FileNotFoundError("OpenCV face models are missing. Run python scripts/download_face_model.py")
    ort.set_default_logger_severity(3)
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    try:
        return ort.InferenceSession(str(MODEL_PATH), providers=providers)
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


@lru_cache(maxsize=6)
def _detector(width, height):
    return cv2.FaceDetectorYN.create(str(DETECTOR_PATH), "", (width, height), score_threshold=0.8,
                                     nms_threshold=0.3, top_k=100)


def detect_faces(frame):
    height, width = frame.shape[:2]
    with _detector_lock:
        _, found = _detector(width, height).detect(frame)
    if found is None:
        return []
    return sorted(found, key=lambda row: float(row[2] * row[3]), reverse=True)


def score_face(frame, detection, session):
    points = np.asarray(detection[4:14], dtype=np.float32).reshape(5, 2)
    matrix, _ = cv2.estimateAffinePartial2D(points, STANDARD_POINTS, method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("Face landmarks could not be aligned")
    aligned = cv2.warpAffine(frame, matrix, (112, 112), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    logits = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0], dtype=np.float64).reshape(-1)
    if logits.shape != (7,) or not np.isfinite(logits).all():
        raise ValueError("OpenCV expression model returned invalid scores")
    values = np.exp(logits - logits.max())
    values /= values.sum()
    probabilities = dict(zip(LABELS, map(float, values)))
    x, y, width, height = map(int, detection[:4])
    return {"emotion": max(probabilities, key=probabilities.get), "probabilities": probabilities,
            "box": (x, y, width, height), "detection_score": float(detection[14])}


def score_frame(frame, session):
    found = detect_faces(frame)
    if not found:
        return None
    return {**score_face(frame, found[0], session), "face_count": len(found)}


def aggregate_frames(frames, session):
    scored = [score_frame(frame, session) for frame in frames]
    valid = [item for item in scored if item is not None]
    if not valid:
        return None
    average = np.mean([[item["probabilities"][label] for label in LABELS] for item in valid], axis=0)
    probs = dict(zip(LABELS, map(float, average)))
    return {"emotion": max(probs, key=probs.get), "probabilities": probs,
            "frames_sampled": len(scored), "frames_with_face": len(valid),
            "multi_face_frames": sum(item["face_count"] > 1 for item in valid)}


def analyze_photo(payload, session):
    frame = decode_image(payload)
    found = detect_faces(frame)
    annotated = frame.copy()
    faces = []
    for number, detection in enumerate(found, 1):
        scored = score_face(frame, detection, session)
        x, y, width, height = scored["box"]
        x2, y2 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
        x, y = max(0, x), max(0, y)
        crop = frame[y:y2, x:x2]
        if crop.size == 0:
            continue
        brightness = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
        warnings = []
        if min(width, height) < 60:
            warnings.append("Detected face is small; expression detail may be insufficient.")
        if brightness < 40:
            warnings.append("Face appears dim; expression scoring may be unreliable.")
        if brightness > 220:
            warnings.append("Face appears overexposed; expression scoring may be unreliable.")
        faces.append({**scored, "number": number, "crop": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                      "warnings": warnings})
        cv2.rectangle(annotated, (x, y), (x2, y2), (114, 107, 228), 3)
        cv2.putText(annotated, str(number), (x + 4, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (114, 107, 228), 2, cv2.LINE_AA)
    return {"image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "faces": faces,
            "width": frame.shape[1], "height": frame.shape[0]}
