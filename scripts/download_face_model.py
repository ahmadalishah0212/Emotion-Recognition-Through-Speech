"""Download and verify retained FER+ and deployed OpenCV Zoo face models."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
SHA256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
DEST = Path(__file__).resolve().parents[1] / "models/face/emotion-ferplus-8.onnx"
OPENCV_MODELS = {
    "facial_expression_recognition_mobilefacenet_2022july.onnx": (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx",
        "4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9"),
    "face_detection_yunet_2023mar.onnx": (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
}


def download():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and hashlib.sha256(DEST.read_bytes()).hexdigest() == SHA256:
        print(f"Verified existing model: {DEST}")
        return DEST
    with urlopen(URL, timeout=90) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("FER+ model checksum mismatch; download was not saved")
    DEST.write_bytes(payload)
    print(f"Downloaded and verified FER+ model: {DEST}")
    return DEST


def download_opencv():
    for filename, (url, expected) in OPENCV_MODELS.items():
        target = DEST.parent / filename
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            print(f"Verified existing model: {target}")
            continue
        with urlopen(url, timeout=90) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"OpenCV Zoo model checksum mismatch: {filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        print(f"Downloaded and verified: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-ferplus", action="store_true",
                        help="Also fetch the 35 MB retired FER+ baseline for comparison")
    args = parser.parse_args()
    download_opencv()
    if args.include_ferplus:
        download()
