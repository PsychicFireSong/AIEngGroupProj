"""
integrate_extra_data.py — Merge downloaded external datasets into v7_lean or severity_dataset.

Datasets:

Stage 1 (detector):
  corrosion-bi3q3 (Roboflow, CC BY 4.0, ~1,249 imgs):
    https://universe.roboflow.com/roboflow-100/corrosion-bi3q3
    Download → YOLOv8 format → extract to a local folder

Stage 2 (severity classifier):
  BD3 Dataset (GitHub, ~3,965 imgs, has major_crack / minor_crack):
    https://github.com/Praveenkottari/BD3-Dataset
    Clone or download ZIP

  CODEBRIM balanced (Kaggle sristi29raj/codebrim-balanced-dataset, ~7.4GB):
    Folder structure: <root>/{Crack,Spalling,ExposedRebars,Corrosion,Efflorescence,...}/
    ExposedRebars → critical (metal bars visible, structural failure)
    Crack + Spalling + Corrosion → moderate (visible structural damage)
    Efflorescence → minor (surface mineral deposits, cosmetic)
    Background → not added (no defect)

Usage:
  # Add corrosion-bi3q3 to Stage 1 v7_lean dataset:
  python scripts/integrate_extra_data.py --task detector --src path\\to\\corrosion-bi3q3

  # Add BD3 major/minor crack to Stage 2 severity dataset:
  python scripts/integrate_extra_data.py --task severity --src path\\to\\BD3-Dataset

  # Add CODEBRIM to Stage 2 severity dataset:
  python scripts/integrate_extra_data.py --task severity-codebrim --src path\\to\\codebrim_balanced
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

V7_LEAN   = Path(r"C:\Users\User\AIEngGroupProj\output\v7_lean_dataset")
SEV_DS    = Path(
    r"C:\Users\User\AIEngGroupProj\output"
    r"\continued_hn_weak\current\datasets\severity_dataset"
)

# v7_lean class mapping (YOLO class IDs)
DETECTOR_CLASS_MAP = {
    # source label → v7_lean class id
    "corrosion": 2,
    "crack":     0,
    "slippage":  1,   # treat as spalling (closest structural equivalent)
    # BD3 detector labels (not used for severity task)
    "major_crack": 0,
    "minor_crack": 0,
    "spalling":    1,
    "peeling":     4,  # paint_degradation
    "stain":       4,
    "algae":       4,
}

# Severity class mapping for BD3
BD3_SEVERITY_MAP = {
    "major_crack": "moderate",   # visible gap → structural concern
    "minor_crack": "minor",      # surface-level crack
    "spalling":    "moderate",
    "peeling":     "minor",
    "stain":       "minor",
    "algae":       "minor",
}


def copy_yolo_split(src_root: Path, dst_root: Path, class_map: dict[str, int],
                    split: str = "train") -> int:
    """Copy images + remap labels from a YOLO-format dataset split."""
    src_imgs  = src_root / "images" / split
    src_lbls  = src_root / "labels" / split
    dst_imgs  = dst_root / "images" / split
    dst_lbls  = dst_root / "labels" / split

    if not src_imgs.exists():
        print(f"  [skip] {split} images not found at {src_imgs}")
        return 0

    dst_imgs.mkdir(parents=True, exist_ok=True)
    dst_lbls.mkdir(parents=True, exist_ok=True)

    # Read source data.yaml for class names
    yaml_path = src_root / "data.yaml"
    src_names: list[str] = []
    if yaml_path.exists():
        import yaml
        data = yaml.safe_load(yaml_path.read_text())
        names = data.get("names", [])
        if isinstance(names, dict):
            src_names = [names[i] for i in sorted(names)]
        else:
            src_names = list(names)
    src_names_lower = [n.lower().replace(" ", "_") for n in src_names]

    copied = 0
    skipped = 0
    for img in src_imgs.glob("*.*"):
        lbl = (src_lbls / img.stem).with_suffix(".txt")
        if not lbl.exists():
            continue

        # Remap labels
        new_lines = []
        for line in lbl.read_text().strip().splitlines():
            if not line:
                continue
            parts = line.split()
            src_cls_id = int(parts[0])
            if src_cls_id < len(src_names_lower):
                src_name = src_names_lower[src_cls_id]
            else:
                src_name = str(src_cls_id)

            # Try direct name match first, then substring match
            dst_cls = None
            for key, val in class_map.items():
                if key.lower() in src_name or src_name in key.lower():
                    dst_cls = val
                    break

            if dst_cls is None:
                skipped += 1
                continue

            new_lines.append(f"{dst_cls} " + " ".join(parts[1:]))

        if not new_lines:
            skipped += 1
            continue

        # Unique filename to avoid collision
        unique_stem = f"ext_{img.stem}"
        dst_img = dst_imgs / (unique_stem + img.suffix)
        dst_lbl = dst_lbls / (unique_stem + ".txt")

        shutil.copy2(img, dst_img)
        dst_lbl.write_text("\n".join(new_lines))
        copied += 1

    print(f"  {split}: copied {copied}, skipped {skipped}")
    return copied


def integrate_detector(src: Path) -> None:
    print(f"\nIntegrating detector data from: {src}")
    total = 0
    for split in ["train", "valid", "test"]:
        dst_split = "val" if split in ("valid", "val") else split
        n = copy_yolo_split(src, V7_LEAN, DETECTOR_CLASS_MAP, split)
        total += n
    print(f"Total images added to v7_lean: {total}")
    print("Re-run training (train_staged.py or train_stage1_v2.py) to use the expanded dataset.")


def integrate_severity(src: Path) -> None:
    """Integrate BD3-style severity dataset into severity_dataset folder structure."""
    print(f"\nIntegrating severity data from: {src}")
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    added = 0

    # BD3 structure: src/major_crack/, src/minor_crack/, src/spalling/, etc.
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir():
            continue
        src_cls = cls_dir.name.lower().replace(" ", "_")
        sev_cls = BD3_SEVERITY_MAP.get(src_cls)
        if sev_cls is None:
            print(f"  [skip] unknown class: {cls_dir.name}")
            continue
        dst_dir = SEV_DS / "train" / sev_cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img in cls_dir.glob("*"):
            if img.suffix.lower() in IMAGE_EXTS:
                dst = dst_dir / f"bd3_{cls_dir.name}_{img.name}"
                if not dst.exists():
                    shutil.copy2(img, dst)
                    added += 1
        print(f"  {cls_dir.name} → severity/{sev_cls}: added to train/")

    print(f"Total images added to severity train: {added}")
    print("Re-run train_severity_frozen.py to use the expanded dataset.")


# CODEBRIM class → severity mapping
# ExposedRebars: metal bars visible = structural failure = critical
# Crack/Spalling/Corrosion: visible structural damage = moderate
# Efflorescence: mineral surface deposits = minor (cosmetic)
# Background: no defect — skip
CODEBRIM_SEVERITY_MAP = {
    "exposedrebars": "critical",
    "exposed_rebars": "critical",
    "crack": "moderate",
    "spalling": "moderate",
    "corrosion": "moderate",
    "efflorescence": "minor",
    "background": None,   # skip
}


def integrate_severity_codebrim(src: Path) -> None:
    """Integrate CODEBRIM balanced dataset into severity_dataset folder structure.

    CODEBRIM structure (balanced version from Kaggle):
      <root>/Crack/, ExposedRebars/, Spalling/, Corrosion/, Efflorescence/, Background/

    Each subfolder contains images of that defect type. CODEBRIM is multi-label in origin
    but the balanced Kaggle version provides single-label folder structure.
    Only adds to train/ split.
    """
    print(f"\nIntegrating CODEBRIM severity data from: {src}")
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    added_by_class: dict[str, int] = {}
    added = 0

    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir():
            continue
        src_cls = cls_dir.name.lower().replace("-", "_")
        sev_cls = CODEBRIM_SEVERITY_MAP.get(src_cls)
        if sev_cls is None:
            print(f"  [skip] {cls_dir.name} (background or unmapped)")
            continue

        dst_dir = SEV_DS / "train" / sev_cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for img in cls_dir.rglob("*"):
            if img.suffix.lower() in IMAGE_EXTS:
                dst = dst_dir / f"cb_{cls_dir.name}_{img.name}"
                if not dst.exists():
                    shutil.copy2(img, dst)
                    n += 1
                    added += 1
        added_by_class[f"{cls_dir.name}→{sev_cls}"] = n
        print(f"  {cls_dir.name} → severity/{sev_cls}: +{n}")

    print(f"\nTotal CODEBRIM images added to severity train: {added}")
    print("Run train_severity_cascade.py to retrain with CODEBRIM-augmented data.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["detector", "severity", "severity-codebrim"], required=True)
    ap.add_argument("--src", required=True, help="Path to the downloaded dataset root")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: source not found: {src}")
        sys.exit(1)

    if args.task == "detector":
        integrate_detector(src)
    elif args.task == "severity":
        integrate_severity(src)
    else:
        integrate_severity_codebrim(src)


if __name__ == "__main__":
    main()
