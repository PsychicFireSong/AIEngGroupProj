"""
build_v7_lean.py — Build a lean ~15K dataset for fast local training.

The base 33K dataset has 4 Roboflow-augmented copies of every source image
(__r01, __r02, __r03, __r04 suffix). We only need one — YOLO's own runtime
augmentation (mosaic, HSV, flip, scale) handles variety far better than
static pre-augmented copies.

Strategy:
  - Base 33K  → keep only __r01 variants (or non-rXX files)  ≈ 8-9K unique
  - CDD 6.8K  → keep all (new domain, no duplicates)          ≈ 6.8K
  - Total     ≈ 15K  (~3-4× less disk I/O per epoch)

Expected epoch time: 18-22 min → 50 epochs ≈ 15-18 h locally
"""
from __future__ import annotations
import re, shutil
from pathlib import Path

BASE_DS  = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current"
                r"\datasets\merged_dataset_hn_weak_continue")
EXTRA_DS = Path(r"C:\Users\User\AIEngGroupProj\Concrete defect detection.yolov11")
OUT_DS   = Path(r"C:\Users\User\AIEngGroupProj\output\v7_lean_dataset")

EXTRA_CLASS_MAP: dict[int, int | None] = {
    0: None, 1: 2, 2: 1, 3: 1, 4: 0, 5: None,
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_RXX = re.compile(r'__r\d+$')   # matches __r01, __r02 … at end of stem


def _is_r01_or_unique(stem: str) -> bool:
    """Keep __r01 variants and any image with no __rXX suffix."""
    m = _RXX.search(stem)
    if m is None:
        return True          # not a Roboflow multi-aug image — keep
    return stem.endswith("__r01")   # keep only the first copy


def _lbl_dir(root: Path, split: str) -> Path:
    p = root / "labels" / split
    return p if p.exists() else root / split / "labels"


def _img_dir(root: Path, split: str) -> Path:
    p = root / "images" / split
    return p if p.exists() else root / split / "images"


def _remap_label(src: Path, dst: Path) -> bool:
    if not src.exists():
        dst.write_text("")
        return False
    mapped = []
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        new_cls = EXTRA_CLASS_MAP.get(cls)
        if new_cls is None:
            continue
        mapped.append(f"{new_cls} " + " ".join(parts[1:]))
    dst.write_text("\n".join(mapped) + ("\n" if mapped else ""), encoding="utf-8")
    return bool(mapped)


def main() -> None:
    print("=" * 60)
    print("build_v7_lean.py  —  lean ~15K dataset")
    print("=" * 60)

    for split in ("train", "val"):
        (OUT_DS / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DS / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── 1. Base train — keep only __r01 / non-rXX ────────────────────────
    print("\n[1/3] Subsampling base train set (r01 only)...")
    base_img = _img_dir(BASE_DS, "train")
    base_lbl = _lbl_dir(BASE_DS, "train")
    kept = skipped = 0
    for img in sorted(base_img.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        if not _is_r01_or_unique(img.stem):
            skipped += 1
            continue
        shutil.copy2(img, OUT_DS / "images" / "train" / img.name)
        lbl = base_lbl / (img.stem + ".txt")
        dst_lbl = OUT_DS / "labels" / "train" / (img.stem + ".txt")
        shutil.copy2(lbl, dst_lbl) if lbl.exists() else dst_lbl.write_text("")
        kept += 1
    print(f"  Kept   : {kept:,}")
    print(f"  Skipped: {skipped:,}  (r02/r03/r04 duplicates)")

    # ── 2. Val set — unchanged ────────────────────────────────────────────
    print("\n[2/3] Copying val set (unchanged)...")
    n_val = 0
    for img in sorted(_img_dir(BASE_DS, "val").iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        shutil.copy2(img, OUT_DS / "images" / "val" / img.name)
        lbl = _lbl_dir(BASE_DS, "val") / (img.stem + ".txt")
        dst = OUT_DS / "labels" / "val" / (img.stem + ".txt")
        shutil.copy2(lbl, dst) if lbl.exists() else dst.write_text("")
        n_val += 1
    print(f"  Copied: {n_val:,}")

    # ── 3. CDD — all splits → train, with remapping ───────────────────────
    print("\n[3/3] Adding CDD images (all splits -> train)...")
    PREFIX = "cdd__"
    cdd_labeled = cdd_hn = 0
    for split in ("train", "valid", "test"):
        src_imgs = _img_dir(EXTRA_DS, split)
        src_lbls = _lbl_dir(EXTRA_DS, split)
        if not src_imgs.exists():
            continue
        for img in sorted(src_imgs.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            out_name = PREFIX + img.name
            if (OUT_DS / "images" / "train" / out_name).exists():
                continue
            shutil.copy2(img, OUT_DS / "images" / "train" / out_name)
            kept_lbl = _remap_label(
                src_lbls / (img.stem + ".txt"),
                OUT_DS / "labels" / "train" / (Path(out_name).stem + ".txt"),
            )
            if kept_lbl:
                cdd_labeled += 1
            else:
                cdd_hn += 1
    print(f"  Labeled    : {cdd_labeled:,}")
    print(f"  Hard-neg   : {cdd_hn:,}")

    # ── 4. data.yaml ─────────────────────────────────────────────────────
    (OUT_DS / "data.yaml").write_text(
        f"path: {OUT_DS.as_posix()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "names:\n- crack\n- spalling\n- corrosion\n- pothole\n- paint_degradation\n",
        encoding="utf-8",
    )

    final = len(list((OUT_DS / "images" / "train").glob("*")))
    print("\n" + "=" * 60)
    print(f"Final train : {final:,}")
    print(f"Final val   : {n_val:,}")
    print(f"data.yaml   : {OUT_DS / 'data.yaml'}")
    print("Done. Run train_v7.py (update DATASET_YAML path).")


if __name__ == "__main__":
    main()
