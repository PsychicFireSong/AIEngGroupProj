"""
prepare_codebrim.py — Extract CODEBRIM and integrate into severity_dataset.

CODEBRIM balanced dataset (Kaggle sristi29raj/codebrim-balanced-dataset):
  Classes: Crack, Spalling, ExposedRebars, Corrosion, Efflorescence, Background

Severity mapping:
  ExposedRebars -> critical   (metal bars = structural failure, visually very distinct)
  Crack         -> moderate   (visible structural crack)
  Spalling      -> moderate   (surface breaking off)
  Corrosion     -> moderate   (metal corrosion)
  Efflorescence -> minor      (mineral deposit, cosmetic)
  Background    -> skip

Why this matters:
  Our current severity_dataset has type-based labels (all 5 defect types in each
  severity class). CODEBRIM gives genuine visual severity signals, especially for
  "critical" (ExposedRebars are unmistakably structural failure — you can see metal bars).

Run after downloading codebrim-balanced-dataset.zip:
  python scripts/prepare_codebrim.py

After this script, run:
  python scripts/train_severity_cascade.py   # (clears old cached datasets first)
"""
from __future__ import annotations

import csv
import random
import shutil
import sys
import zipfile
from pathlib import Path

ZIP_PATH = Path(
    r"C:\Users\User\AIEngGroupProj\output\downloaded_datasets"
    r"\codebrim\codebrim-balanced-dataset.zip"
)
EXTRACT_DIR = Path(
    r"C:\Users\User\AIEngGroupProj\output\downloaded_datasets\codebrim_ext"
)
SEV_DS = Path(
    r"C:\Users\User\AIEngGroupProj\output"
    r"\continued_hn_weak\current\datasets\severity_dataset"
)
# Binary cascade dataset dirs (will be deleted so cascade script rebuilds them)
CASCADE_DS = Path(
    r"C:\Users\User\AIEngGroupProj\output\severity_cascade_datasets"
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Multi-label -> single-label severity assignment (most severe wins)
# Priority: ExposedBars (critical) > Crack/Spallation/CorrosionStain (moderate) > Efflorescence (minor)
# Background-only images are skipped.
SEVERITY_PRIORITY = [
    (["ExposedBars"], "critical"),
    (["Crack", "Spallation", "CorrosionStain"], "moderate"),
    (["Efflorescence"], "minor"),
]


def parse_codebrim_xml(xml_path: Path) -> dict[str, str]:
    """Returns {filename: severity_label} using most-severe-wins assignment."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()

    result = {}
    for defect in root.findall("Defect"):
        fname = defect.get("name", "")
        if not fname:
            continue
        labels = {child.tag: int(child.text or 0) for child in defect}
        # Most severe wins
        sev = None
        for tags, label in SEVERITY_PRIORITY:
            if any(labels.get(t, 0) for t in tags):
                sev = label
                break
        if sev:
            result[fname] = sev
    return result


def extract_zip() -> Path:
    if not ZIP_PATH.exists():
        print(f"ERROR: zip not found: {ZIP_PATH}")
        sys.exit(1)

    # Verify zip integrity (check EOCD)
    with open(ZIP_PATH, "rb") as f:
        f.seek(-100, 2)
        tail = f.read(100)
    if b"PK\x05\x06" not in tail and b"PK\x06\x06" not in tail:
        print("ERROR: ZIP file is incomplete (no EOCD). Re-download required.")
        sys.exit(1)

    print(f"ZIP verified ({ZIP_PATH.stat().st_size/1e9:.1f} GB). Extracting to {EXTRACT_DIR} ...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print("Extraction complete.")
    return EXTRACT_DIR


def find_codebrim_root(base: Path) -> Path:
    """Find the folder that contains the CODEBRIM class subdirectories."""
    for pattern in ["ExposedRebars", "exposedrebars", "Crack", "crack"]:
        for p in base.rglob(pattern):
            if p.is_dir():
                return p.parent
    return base


def integrate_into_severity(cb_root: Path) -> dict[str, int]:
    """Parse CODEBRIM metadata XML and copy images with severity labels.

    CODEBRIM balanced dataset structure:
      <root>/classification_dataset_balanced/
        metadata/defects.xml  — multi-label XML per image
        train/defects/        — actual image files
        val/defects/
        test/defects/
    """
    # Try to find the classification_dataset_balanced root
    cdb_root = None
    for p in cb_root.rglob("classification_dataset_balanced"):
        if p.is_dir():
            cdb_root = p
            break
    # Also try direct
    if cdb_root is None and (cb_root / "classification_dataset_balanced").is_dir():
        cdb_root = cb_root / "classification_dataset_balanced"
    if cdb_root is None:
        print(f"  [warn] Could not find classification_dataset_balanced in {cb_root}, trying root")
        cdb_root = cb_root

    print(f"\nIntegrating CODEBRIM from: {cdb_root}")

    # Parse all metadata XMLs
    fname_to_sev: dict[str, str] = {}
    for xml_file in ["metadata/defects.xml", "metadata/background.xml"]:
        xp = cdb_root / xml_file
        if xp.exists():
            fname_to_sev.update(parse_codebrim_xml(xp))
            print(f"  Parsed {xp.name}: {len(fname_to_sev)} entries so far")

    if not fname_to_sev:
        print("  ERROR: No metadata XML found — skipping CODEBRIM integration")
        return {}

    # Count severity distribution in metadata
    from collections import Counter
    sev_count = Counter(fname_to_sev.values())
    print(f"  Metadata severity distribution: {dict(sev_count)}")

    added: dict[str, int] = {"critical": 0, "moderate": 0, "minor": 0}

    # Process train split
    for split_src in ["train", "val"]:  # use CODEBRIM train+val as our train
        src_dir = cdb_root / split_src / "defects"
        if not src_dir.exists():
            print(f"  [skip] {split_src}/defects not found")
            continue

        imgs = [f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        for img in imgs:
            sev = fname_to_sev.get(img.name)
            if sev is None:
                continue
            dst = SEV_DS / "train" / sev
            dst.mkdir(parents=True, exist_ok=True)
            target = dst / f"cb_{img.name}"
            if not target.exists():
                shutil.copy2(img, target)
                added[sev] += 1

    # Use CODEBRIM test split as our held-out codebrim test set
    src_test = cdb_root / "test" / "defects"
    if src_test.exists():
        n_test = 0
        for img in src_test.iterdir():
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            sev = fname_to_sev.get(img.name)
            if sev is None:
                continue
            dst = SEV_DS / "test_codebrim" / sev
            dst.mkdir(parents=True, exist_ok=True)
            target = dst / f"cb_{img.name}"
            if not target.exists():
                shutil.copy2(img, target)
                n_test += 1
        print(f"  CODEBRIM test set: +{n_test} images -> severity_dataset/test_codebrim/")

    for sev, n in added.items():
        print(f"  -> severity/train/{sev}: +{n}")

    return added


def clear_cascade_cache() -> None:
    """Delete old binary cascade datasets so train_severity_cascade.py rebuilds them."""
    if CASCADE_DS.exists():
        shutil.rmtree(CASCADE_DS)
        print(f"\nCleared old cascade datasets: {CASCADE_DS}")


def report_sizes() -> None:
    print("\nNew severity_dataset sizes:")
    for split in ["train", "val", "test", "test_codebrim"]:
        sp = SEV_DS / split
        if not sp.exists():
            continue
        print(f"  {split}:")
        for cls in sorted(sp.iterdir()):
            if cls.is_dir():
                n = sum(1 for f in cls.glob("*") if f.suffix.lower() in IMAGE_EXTS)
                print(f"    {cls.name}: {n}")


def main() -> None:
    print("=" * 60)
    print("CODEBRIM integration for severity_dataset")
    print("=" * 60)

    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found. Download from Kaggle first.")
        sys.exit(1)

    # Extract
    cb_root_base = extract_zip()
    cb_root = find_codebrim_root(cb_root_base)
    print(f"CODEBRIM class root: {cb_root}")

    # Integrate into severity_dataset
    added = integrate_into_severity(cb_root)

    # Clear old cascade cache (forces rebuild with new data)
    clear_cascade_cache()

    # Report final sizes
    report_sizes()

    total_added = sum(added.values())
    print(f"\nTotal added to severity train: {total_added} images")
    print("\nNext step:")
    print("  python scripts/train_severity_cascade.py")
    print("  (Will rebuild binary cascade datasets including CODEBRIM data)")


if __name__ == "__main__":
    main()
