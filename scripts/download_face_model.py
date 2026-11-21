"""Download the official ONNX Model Zoo FER+ model and verify its bytes."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
SHA256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
DEST = Path(__file__).resolve().parents[1] / "models/face/emotion-ferplus-8.onnx"


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


if __name__ == "__main__":
    download()
