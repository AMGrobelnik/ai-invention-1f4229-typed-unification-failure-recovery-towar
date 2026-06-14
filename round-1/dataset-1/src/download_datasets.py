#!/usr/bin/env python3
"""Download RuleTaker and ProofWriter datasets from HuggingFace and save to temp/datasets/."""

import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/download.log", rotation="30 MB", level="DEBUG")

Path("logs").mkdir(exist_ok=True)
OUT = Path("temp/datasets")
OUT.mkdir(parents=True, exist_ok=True)


def save_dataset(name: str, ds, split: str, max_rows: int | None = None) -> None:
    safe = name.replace("/", "_")
    rows = list(ds)
    if max_rows:
        rows = rows[:max_rows]
    full_path = OUT / f"full_{safe}_{split}.json"
    mini_path = OUT / f"mini_{safe}_{split}.json"
    preview_path = OUT / f"preview_{safe}_{split}.json"
    full_path.write_text(json.dumps(rows, indent=2))
    mini_path.write_text(json.dumps(rows[:3], indent=2))
    preview_path.write_text(json.dumps(rows[:3], indent=2))
    logger.info(f"Saved {len(rows)} rows → {full_path.name}")


def download_ruletaker() -> None:
    from datasets import load_dataset
    logger.info("Downloading tasksource/ruletaker train split (streaming)...")
    ds = load_dataset("tasksource/ruletaker", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if (i + 1) % 10000 == 0:
            logger.info(f"  ruletaker: {i+1} rows collected")
        if i >= 49999:  # cap at 50k
            break
    logger.info(f"Collected {len(rows)} ruletaker train rows")
    safe = "tasksource_ruletaker"
    full_path = OUT / f"full_{safe}_train.json"
    mini_path = OUT / f"mini_{safe}_train.json"
    preview_path = OUT / f"preview_{safe}_train.json"
    full_path.write_text(json.dumps(rows, indent=2))
    mini_path.write_text(json.dumps(rows[:3], indent=2))
    preview_path.write_text(json.dumps(rows[:3], indent=2))
    logger.info(f"ruletaker saved: {full_path}")

    # also get test split (smaller)
    logger.info("Downloading tasksource/ruletaker test split...")
    ds_test = load_dataset("tasksource/ruletaker", split="test", streaming=True)
    test_rows = []
    for i, row in enumerate(ds_test):
        test_rows.append(row)
        if i >= 9999:
            break
    full_test = OUT / f"full_{safe}_test.json"
    full_test.write_text(json.dumps(test_rows, indent=2))
    (OUT / f"mini_{safe}_test.json").write_text(json.dumps(test_rows[:3], indent=2))
    (OUT / f"preview_{safe}_test.json").write_text(json.dumps(test_rows[:3], indent=2))
    logger.info(f"ruletaker test saved: {len(test_rows)} rows → {full_test.name}")


def download_proofwriter() -> None:
    from datasets import load_dataset
    logger.info("Downloading tasksource/proofwriter train split (streaming)...")
    ds = load_dataset("tasksource/proofwriter", split="train", streaming=True)
    rows = []
    for i, row in enumerate(ds):
        rows.append(row)
        if (i + 1) % 10000 == 0:
            logger.info(f"  proofwriter: {i+1} rows collected")
        if i >= 49999:
            break
    logger.info(f"Collected {len(rows)} proofwriter train rows")
    safe = "tasksource_proofwriter"
    (OUT / f"full_{safe}_train.json").write_text(json.dumps(rows, indent=2))
    (OUT / f"mini_{safe}_train.json").write_text(json.dumps(rows[:3], indent=2))
    (OUT / f"preview_{safe}_train.json").write_text(json.dumps(rows[:3], indent=2))
    logger.info(f"proofwriter saved")

    logger.info("Downloading tasksource/proofwriter test split...")
    ds_test = load_dataset("tasksource/proofwriter", split="test", streaming=True)
    test_rows = []
    for i, row in enumerate(ds_test):
        test_rows.append(row)
        if i >= 9999:
            break
    (OUT / f"full_{safe}_test.json").write_text(json.dumps(test_rows, indent=2))
    (OUT / f"mini_{safe}_test.json").write_text(json.dumps(test_rows[:3], indent=2))
    (OUT / f"preview_{safe}_test.json").write_text(json.dumps(test_rows[:3], indent=2))
    logger.info(f"proofwriter test: {len(test_rows)} rows")


def main() -> None:
    logger.info("=== Dataset Download Script ===")
    try:
        download_ruletaker()
    except Exception:
        logger.error("Failed to download ruletaker")

    try:
        download_proofwriter()
    except Exception:
        logger.error("Failed to download proofwriter")

    logger.info("=== Download complete ===")
    files = list(OUT.glob("full_*.json"))
    for f in sorted(files):
        size_mb = f.stat().st_size / 1e6
        logger.info(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
