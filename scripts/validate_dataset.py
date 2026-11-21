from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_loader import scan_dataset, dataset_summary

if __name__ == "__main__":
    rows, errors = scan_dataset()
    summary = dataset_summary(rows)
    duplicates = [item for item in errors if item["error"] == "duplicate audio bytes"]
    invalid = [item for item in errors if item["error"] != "duplicate audio bytes"]
    raw_count = len(rows) + len(errors)
    complete = raw_count == 1440 and summary["actors"] == 24 and not invalid
    print(json.dumps({"summary": summary, "raw_files": raw_count,
                      "duplicate_files_excluded": duplicates, "invalid_files": invalid,
                      "complete_official_speech_archive": complete}, indent=2))
    if not complete:
        print("Expected 1,440 speech WAV filenames from 24 actors with no corrupt files. Check extraction.")
        raise SystemExit(1)
