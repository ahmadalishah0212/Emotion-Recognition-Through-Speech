"""Download optional official actor ZIPs for face and audiovisual evaluation."""
import argparse
from hashlib import md5
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS = {"01": "3c8ececaf392b4a9b11b32271f4f6d01",
             "05": "b00e17d61374ef6ba86fef35d50a20fb"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actors", nargs="*", default=["01", "05"], choices=CHECKSUMS)
    args = parser.parse_args()
    cache = ROOT / "data/cache"
    cache.mkdir(parents=True, exist_ok=True)
    for actor in args.actors:
        filename = f"Video_Speech_Actor_{actor}.zip"
        path = cache / filename
        if not path.exists():
            url = f"https://zenodo.org/api/records/1188976/files/{filename}/content"
            print(f"Downloading {filename} from the official RAVDESS record...", flush=True)
            urllib.request.urlretrieve(url, path)
        checksum = md5()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        digest = checksum.hexdigest()
        if digest != CHECKSUMS[actor]:
            raise ValueError(f"{filename} official MD5 mismatch: {digest}")
        print(f"Verified {path}", flush=True)


if __name__ == "__main__":
    main()
