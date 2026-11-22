"""Face detection and expression scoring; no identity recognition or storage."""
from pathlib import Path
from functools import lru_cache
from io import BytesIO
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/face/emotion-ferplus-8.onnx"
FACE_LABELS = ("neutral", "happy", "surprised", "sad", "angry", "disgust", "fearful", "contempt")
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
MAX_PROCESSING_EDGE = 1600


@lru_cache(maxsize=1)
def load_face_model(path=MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("FER+ baseline model is missing. Run python scripts/download_face_model.py --include-ferplus")
    available = ort.get_available_providers()
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available
                 else ["CPUExecutionProvider"])
    try:
        return ort.InferenceSession(str(path), providers=providers)
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def decode_image(payload):
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("Choose a nonempty image under 12 MB")
    try:
        with Image.open(BytesIO(payload)) as source:
            if source.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Supported photos: JPG, PNG or WebP")
            width, height = source.size
            if width * height > MAX_IMAGE_PIXELS or min(width, height) < 64:
                raise ValueError("Image dimensions must be at least 64 pixels and at most 24 megapixels")
            rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Image could not be decoded. Try JPG, PNG or WebP.") from exc
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    edge = max(image.shape[:2])
    if edge > MAX_PROCESSING_EDGE:
        scale = MAX_PROCESSING_EDGE / edge
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
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
    result = score_face(frame, boxes[0], session)
    return {**result, "face_count": len(boxes)}


def score_face(frame, box, session):
    x, y, width, height = box
    gray = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32).reshape(1, 1, 64, 64)
    logits = np.asarray(session.run(None, {session.get_inputs()[0].name: tensor})[0], dtype=np.float64).reshape(-1)
    if logits.shape != (8,) or not np.isfinite(logits).all():
        raise ValueError("Facial model returned invalid scores")
    shifted = logits - logits.max()
    probs = np.exp(shifted)
    probs /= probs.sum()
    return {"probabilities": dict(zip(FACE_LABELS, map(float, probs))), "box": box}


def analyze_photo(payload, session):
    """Return separately scored detected faces and a numbered preview; no identity tracking."""
    frame = decode_image(payload)
    boxes = detect_faces(frame)
    faces = []
    annotated = frame.copy()
    for number, box in enumerate(boxes, 1):
        scored = score_face(frame, box, session)
        probs = scored["probabilities"]
        x, y, width, height = box
        crop = frame[y:y + height, x:x + width].copy()
        brightness = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
        warnings = []
        if min(width, height) < 60:
            warnings.append("Detected face is small; expression detail may be insufficient.")
        if brightness < 40:
            warnings.append("Face appears dim; expression scoring may be unreliable.")
        if brightness > 220:
            warnings.append("Face appears overexposed; expression scoring may be unreliable.")
        faces.append({"number": number, "box": box, "emotion": max(probs, key=probs.get),
                      "probabilities": probs, "crop": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                      "warnings": warnings})
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (114, 107, 228), 3)
        cv2.putText(annotated, str(number), (x + 4, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (114, 107, 228), 2, cv2.LINE_AA)
    return {"image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "faces": faces,
            "width": frame.shape[1], "height": frame.shape[0]}


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
