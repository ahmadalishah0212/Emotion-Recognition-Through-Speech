"""Bounded video frame sampling and local FFmpeg audio extraction."""
import subprocess
import tempfile
from pathlib import Path
import cv2
import imageio_ffmpeg

MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_VIDEO_SECONDS = 120
MAX_FRAMES = 12


def analyze_video_payload(payload, suffix=".mp4"):
    if not payload or len(payload) > MAX_VIDEO_BYTES:
        raise ValueError("Choose a nonempty video under 50 MB")
    if suffix.lower() not in (".mp4", ".mov", ".webm", ".avi"):
        raise ValueError("Supported videos: MP4, MOV, WebM, AVI")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / ("upload" + suffix.lower())
        path.write_bytes(payload)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("Video could not be decoded")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if count <= 0 or fps <= 0:
            capture.release()
            raise ValueError("Video has no readable frames")
        duration = count / fps
        if duration > MAX_VIDEO_SECONDS:
            capture.release()
            raise ValueError("Video must be at most 2 minutes long")
        indices = sorted(set(round(i * (count - 1) / (min(MAX_FRAMES, count) - 1)) for i in range(min(MAX_FRAMES, count)))) if count > 1 else [0]
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)
        capture.release()
        if not frames:
            raise ValueError("No video frames could be read")
        audio_path = Path(folder) / "audio.wav"
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
                   "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "-y", str(audio_path)]
        completed = subprocess.run(command, capture_output=True, timeout=35, check=False)
        audio = audio_path.read_bytes() if completed.returncode == 0 and audio_path.exists() else None
        return {"frames": frames, "audio": audio, "duration": duration,
                "frame_count": count, "fps": fps, "sampled": len(frames)}
