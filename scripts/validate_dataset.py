from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_loader import scan_dataset, dataset_summary

if __name__ == "__main__":
    rows, errors = scan_dataset()
    summary = dataset_summary(rows)
    complete = summary["files"] == 1440 and summary["actors"] == 24
    print(json.dumps({"summary": summary, "invalid_files": errors,
                      "complete_official_speech_archive": complete}, indent=2))
    if errors or not complete:
        print("Expected 1,440 valid speech WAV files from 24 actors. Check extraction and corruption.")
        raise SystemExit(1)
