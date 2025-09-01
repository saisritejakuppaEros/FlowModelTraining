#!/usr/bin/env python3
import os
import json
import csv
import sys
import traceback
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ====== CONFIG ======
# Folder that contains movie/shot/frame JSONs
JSON_ROOT = Path("/data0/teja_codes/ImmersoAiResearch/ImageModelTraining/annotation/output_annotations/captions")
# Output CSV path
CSV_OUT = Path("./captions_dataset.csv")
# Number of threads (IO-bound -> many threads helps). None = sensible default.
MAX_WORKERS = None  # e.g. 32
# Max images to process (set high or None to take all)
MAX_IMAGES = 200
# ====================

def find_json_files(root: Path):
    """Recursively find all .json files under root."""
    return [p for p in root.rglob("*.json") if p.is_file()]

def process_one(json_path: Path):
    """
    Read a single JSON and return the CSV row:
    [caption, image_path, aesthetic_score]
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = data.get("image_path")
        caption_text = (data.get("caption").replace(",", " ") or "").strip()

        # Skip invalid entries
        if not image_path or not caption_text:
            return None

        # Sanitize caption to avoid multi-line CSV issues
        caption_text = " ".join(caption_text.split())

        # CSV row: caption text, absolute image path, score 0
        return [caption_text, image_path, 0]

    except Exception as e:
        # Log and continue
        sys.stderr.write(f"[ERROR] {json_path}: {e}\n")
        traceback.print_exc(file=sys.stderr)
        return None

def main():
    json_files = find_json_files(JSON_ROOT)
    if not json_files:
        print(f"No JSON files found under: {JSON_ROOT}")
        return

    # Shuffle and limit for load distribution / sampling
    random.shuffle(json_files)
    if MAX_IMAGES is not None:
        json_files = json_files[:MAX_IMAGES]

    rows = []
    workers = MAX_WORKERS or min(64, (os.cpu_count() or 8) * 5)

    print(f"Discovered {len(json_files)} JSON files (after shuffle/limit). Processing with {workers} threads...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one, jp) for jp in json_files]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing JSONs"):
            res = fut.result()
            if res is not None:
                rows.append(res)

    # Write CSV with caption text, absolute image path, aesthetic_score=0
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["caption", "image_path", "aesthetic_score"])
        writer.writerows(rows)

    print(f"Done! Wrote {len(rows)} rows to {CSV_OUT}")

if __name__ == "__main__":
    main()
