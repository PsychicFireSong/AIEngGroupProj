"""
build_v7_fixed.py - Targeted data fixes based on per-class diagnosis.

Problems addressed:
  1. crack (50% recall) and corrosion (42% recall): oversample 2x by copying
     images whose label files contain mostly those classes.
  2. paint_degradation (65% FP rate): add hard-negative images - images that
     contain NO defect annotations, so the model learns to suppress firing on
     clean textures.

Input : output/v7_lean_dataset  (20,166 train + 3,855 val)
Output: output/v7_fixed_dataset (train enlarged, val unchanged)
"""
from __future__ import annotations

import collections
import shutil
from pathlib import Path

SRC_DS  = Path(r"C:\Users\User\AIEngGroupProj\output\v7_lean_dataset")
OUT_DS  = Path(r"C:\Users\User\AIEngGroupProj\output\v7_fixed_dataset")
CLASSES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Classes to oversample (index -> copies to ADD, so 1 = double)
OVERSAMPLE = {0: 1, 2: 1}  # crack x2, corrosion x2


def dominant_class(lbl_path: Path) -> int | None:
    """Return the most frequent class index in a label file, or None if empty."""
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return None
    counts: dict[int, int] = collections.defaultdict(int)
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if parts:
            try:
                counts[int(float(parts[0]))] += 1
            except ValueError:
                pass
    return max(counts, key=counts.__getitem__) if counts else None


def has_any_label(lbl_path: Path) -> bool:
    if not lbl_path.exists():
        return False
    return lbl_path.stat().st_size > 0


def main() -> None:
    print("=" * 60)
    print("build_v7_fixed.py  -  targeted data fixes")
    print("=" * 60)

    src_img = SRC_DS / "images" / "train"
    src_lbl = SRC_DS / "labels" / "train"
    out_img = OUT_DS / "images" / "train"
    out_lbl = OUT_DS / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    (OUT_DS / "images" / "val").mkdir(parents=True, exist_ok=True)
    (OUT_DS / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # ── 1. Copy all existing train images ────────────────────────────────────
    print("\n[1/4] Copying base train set...")
    n_base = 0
    for img in sorted(src_img.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        shutil.copy2(img, out_img / img.name)
        lbl = src_lbl / (img.stem + ".txt")
        dst = out_lbl / (img.stem + ".txt")
        shutil.copy2(lbl, dst) if lbl.exists() else dst.write_text("")
        n_base += 1
    print(f"  Copied: {n_base:,}")

    # ── 2. Oversample crack and corrosion images ──────────────────────────────
    print("\n[2/4] Oversampling crack (cls 0) and corrosion (cls 2) x2...")
    oversample_counts: dict[int, int] = collections.defaultdict(int)
    for img in sorted(src_img.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = src_lbl / (img.stem + ".txt")
        dom = dominant_class(lbl)
        if dom not in OVERSAMPLE:
            continue
        for copy_n in range(1, OVERSAMPLE[dom] + 1):
            new_stem = f"boost{copy_n}__{img.stem}"
            shutil.copy2(img, out_img / (new_stem + img.suffix))
            dst_lbl = out_lbl / (new_stem + ".txt")
            shutil.copy2(lbl, dst_lbl) if lbl.exists() else dst_lbl.write_text("")
            oversample_counts[dom] += 1
    for cls_id, cnt in sorted(oversample_counts.items()):
        print(f"  {CLASSES[cls_id]}: +{cnt:,} copies added")

    # ── 3. Hard-negative mining for paint_degradation FP ─────────────────────
    # Use existing CDD images that had no valid class mapping (empty label files).
    # Cap at 2000 to avoid drowning out positive examples.
    print("\n[3/4] Adding hard-negative images (empty-label background images)...")
    HN_CAP = 2000
    n_hn = 0
    for img in sorted(src_img.iterdir()):
        if n_hn >= HN_CAP:
            break
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = src_lbl / (img.stem + ".txt")
        if has_any_label(lbl):
            continue
        new_name = "hn__" + img.name
        if (out_img / new_name).exists():
            continue
        shutil.copy2(img, out_img / new_name)
        dst_lbl = out_lbl / (Path(new_name).stem + ".txt")
        dst_lbl.write_text("")
        n_hn += 1
    print(f"  Hard negatives added: {n_hn:,}  (cap={HN_CAP})")

    # ── 4. Copy val set unchanged ─────────────────────────────────────────────
    print("\n[4/4] Copying val set (unchanged)...")
    n_val = 0
    for img in sorted((SRC_DS / "images" / "val").iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        shutil.copy2(img, OUT_DS / "images" / "val" / img.name)
        lbl = (SRC_DS / "labels" / "val") / (img.stem + ".txt")
        dst = (OUT_DS / "labels" / "val") / (img.stem + ".txt")
        shutil.copy2(lbl, dst) if lbl.exists() else dst.write_text("")
        n_val += 1
    print(f"  Copied: {n_val:,}")

    # ── Write data.yaml ───────────────────────────────────────────────────────
    (OUT_DS / "data.yaml").write_text(
        f"path: {OUT_DS.as_posix()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "names:\n- crack\n- spalling\n- corrosion\n- pothole\n- paint_degradation\n",
        encoding="utf-8",
    )

    final = len(list(out_img.glob("*")))
    print("\n" + "=" * 60)
    print(f"Final train : {final:,}")
    print(f"Final val   : {n_val:,}")
    print(f"data.yaml   : {OUT_DS / 'data.yaml'}")
    print("Done.")


if __name__ == "__main__":
    main()
