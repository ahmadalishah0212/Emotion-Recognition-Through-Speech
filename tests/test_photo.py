"""Photo path checks that exercise real decoding and per-face output."""
from io import BytesIO
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from src.face import analyze_photo, decode_image, load_face_model


@pytest.mark.parametrize("format_name", ["JPEG", "PNG", "WEBP"])
def test_common_photo_formats_decode(format_name):
    image = Image.new("RGB", (96, 80), (35, 65, 95))
    output = BytesIO()
    image.save(output, format=format_name)
    decoded = decode_image(output.getvalue())
    assert decoded.shape == (80, 96, 3)


def test_corrupt_and_oversized_photo_rejected():
    with pytest.raises(ValueError, match="could not be decoded"):
        decode_image(b"invalid image")
    with pytest.raises(ValueError, match="under 12 MB"):
        decode_image(b"x" * (12 * 1024 * 1024 + 1))


def test_blank_photo_has_no_prediction():
    output = BytesIO()
    Image.new("RGB", (256, 256), (0, 0, 0)).save(output, format="PNG")
    result = analyze_photo(output.getvalue(), session=None)
    assert result["faces"] == []


def test_local_sample_face_returns_linked_region_if_available():
    sample = Path(__file__).resolve().parents[1] / "data/cache/lena.jpg"
    if not sample.exists():
        pytest.skip("Optional uncommitted face fixture absent")
    result = analyze_photo(sample.read_bytes(), load_face_model())
    assert result["faces"]
    first = result["faces"][0]
    assert first["number"] == 1
    assert first["crop"].size > 0
    assert first["emotion"] == max(first["probabilities"], key=first["probabilities"].get)
    assert sum(first["probabilities"].values()) == pytest.approx(1)


def test_two_local_faces_have_separate_regions_if_available():
    sample = Path(__file__).resolve().parents[1] / "data/cache/lena.jpg"
    if not sample.exists():
        pytest.skip("Optional uncommitted face fixture absent")
    face = decode_image(sample.read_bytes())
    from PIL import Image as PILImage
    import cv2
    duplicated = np.concatenate([face, face], axis=1)
    output = BytesIO()
    PILImage.fromarray(cv2.cvtColor(duplicated, cv2.COLOR_BGR2RGB)).save(output, format="PNG")
    result = analyze_photo(output.getvalue(), load_face_model())
    assert len(result["faces"]) >= 2
    assert result["faces"][0]["box"] != result["faces"][1]["box"]


def test_aligned_opencv_model_scores_local_face_if_available():
    from src.face_opencv import MODEL_PATH, DETECTOR_PATH, analyze_photo as analyze_zoo_photo, load_model
    sample = Path(__file__).resolve().parents[1] / "data/cache/lena.jpg"
    if not sample.exists() or not MODEL_PATH.exists() or not DETECTOR_PATH.exists():
        pytest.skip("Optional face fixture or OpenCV Zoo models absent")
    result = analyze_zoo_photo(sample.read_bytes(), load_model())
    assert result["faces"]
    assert set(result["faces"][0]["probabilities"]) == {
        "angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised"}
    assert sum(result["faces"][0]["probabilities"].values()) == pytest.approx(1)


def test_aligned_model_keeps_two_faces_separate_if_available():
    from src.face_opencv import MODEL_PATH, DETECTOR_PATH, analyze_photo as analyze_zoo_photo, load_model
    import cv2

    sample = Path(__file__).resolve().parents[1] / "data/cache/lena.jpg"
    if not sample.exists() or not MODEL_PATH.exists() or not DETECTOR_PATH.exists():
        pytest.skip("Optional face fixture or OpenCV Zoo models absent")
    duplicated = np.concatenate([decode_image(sample.read_bytes())] * 2, axis=1)
    ok, encoded = cv2.imencode(".png", duplicated)
    assert ok
    result = analyze_zoo_photo(encoded.tobytes(), load_model())
    assert len(result["faces"]) >= 2
    assert result["faces"][0]["number"] == 1
    assert result["faces"][1]["number"] == 2
    assert result["faces"][0]["box"] != result["faces"][1]["box"]
