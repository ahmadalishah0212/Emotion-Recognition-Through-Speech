"""Fetch the two public external corpora from their official repositories."""
from hashlib import sha256
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "tess": ("https://borealisdata.ca/api/access/dataset/:persistentId/?persistentId=doi:10.5683/SP2/E8H2MF",
             "45debfce1306d58d15b204cf4b0df5fae1a1fe7133acfead0f3b2d1b91a88b74"),
    "emodb": ("https://zenodo.org/api/records/7447302/files/emodb.zip/content",
              "1beeeee9276a43ab0c81fce3954ed7827afb677d2be5a336ac1ccac05e360155"),
}


def extract_safe(archive, target, strip_prefix=""):
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            relative = item.filename.removeprefix(strip_prefix)
            if not relative:
                continue
            destination = (target / relative).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise ValueError(f"Archive contains path outside destination: {item.filename}")
        for item in package.infolist():
            relative = item.filename.removeprefix(strip_prefix)
            if not relative:
                continue
            destination = target / relative
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with package.open(item) as source, destination.open("wb") as output:
                    import shutil
                    shutil.copyfileobj(source, output)


def main():
    chosen = sys.argv[1:] or list(SOURCES)
    for name in chosen:
        if name not in SOURCES:
            raise ValueError(f"Unknown dataset {name}; use tess or emodb")
        url, expected = SOURCES[name]
        archive = ROOT / "data/cache" / f"{name}.zip"
        target = ROOT / "data/external" / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            print(f"Downloading {name} from its official repository...", flush=True)
            urllib.request.urlretrieve(url, archive)
        actual = sha256(archive.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name} archive SHA-256 mismatch: {actual}")
        extract_safe(archive, target, "emodb/" if name == "emodb" else "")
        print(f"Verified and extracted {name} to {target}", flush=True)


if __name__ == "__main__":
    main()
