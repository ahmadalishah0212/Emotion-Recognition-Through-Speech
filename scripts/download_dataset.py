"""Download only the official RAVDESS audio-only speech ZIP and extract safely."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import hashlib

URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
MD5 = "bc696df654c87fed845eb13823edef8a"


def main():
    target = Path("data/raw")
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "Audio_Speech_Actors_01-24.zip"
    if not archive.exists():
        print("Downloading official RAVDESS speech audio (~209 MB)...")
        urlretrieve(URL, archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != MD5:
        archive.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch ({digest}); download again from {URL}")
    with ZipFile(archive) as zf:
        for item in zf.infolist():
            destination = (target / item.filename).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise ValueError("Unsafe archive path")
        zf.extractall(target)
    archive.unlink()
    print("Extracted to data/raw. Run python scripts/validate_dataset.py")


if __name__ == "__main__":
    main()
