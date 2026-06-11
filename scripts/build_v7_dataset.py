"""
build_v7_dataset.py — Build the v7 training dataset.

Merges:
  A) Existing merged_dataset_hn_weak_continue (33,096 train images)  ← kept as-is
  B) Concrete defect detection.yolov11 root-level (6,806 images)     ← remapped

Class mapping for source B:
  0  Exposed_reinforcement  → SKIP annotation  (keep image as hard-negative)
  1  Ruststrain             → 2  corrosion
  2  Scaling                → 1  spalling
  3  Spalling               → 1  spalling
  4  crack                  → 0  crack
  5  efflorescence          → SKIP annotation  (keep image as hard-negative)

Images whose labels reduce to EMPTY after remapping are kept as hard-negatives
(empty .txt = no-defect background image, hurts FP rate).

Validation set is kept IDENTICAL to the current one for apples-to-apples
comparison with the 0.694 baseline.

Output layout:
  C:/Users/User/AIEngGroupProj/output/v7_dataset/
    images/train/   (~39-40 K)
    images/val/     (3,855 — unchanged)
    labels/train/
    labels/val/
    data.yaml
"""
from __future__ import annotations

import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DS   = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current"
                 r"\datasets\merged_dataset_hn_weak_continue")
EXTRA_DS  = Path(r"C:\Users\User\AIEngGroupProj\Concrete defect detection.yolov11")
OUT_DS    = Path(r"C:\Users\User\AIEngGroupProj\output\v7_dataset")

# Class mapping: extra_class_id → our_class_id  (None = skip this annotation)
EXTRA_CLASS_MAP: dict[int, int | None] = {
    0: None,   # Exposed_reinforcement → skip
    1: 2,      # Ruststrain            → corrosion
    2: 1,      # Scaling               → spalling
    3: 1,      # Spalling              → spalling
    4: 0,      # crack                 → crack
    5: None,   # efflorescence         → skip
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Helpers ───────────────────────────────────────────────────────────────

def _img_dir(root: Path, split: str) -> Path:
    """Handles both images/<split> and <split>/images layout."""
    p = root / "images" / split
    if p.exists():
        return p
    return root / split / "images"


def _lbl_dir(root: Path, split: str) -> Path:
    p = root / "labels" / split
    if p.exists():
        return p
    return root / split / "labels"


def _copy_split(src_img: Path, src_lbl: Path, dst_img: Path, dst_lbl: Path,
                prefix: str = "") -> int:
    """Copies all images + labels verbatim. Returns count."""
    count = 0
    for img in sorted(src_img.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = (prefix + img.name) if prefix else img.name
        shutil.copy2(img, dst_img / stem)
        lbl = src_lbl / (img.stem + ".txt")
        dst_stem = Path(stem).stem + ".txt"
        if lbl.exists():
            shutil.copy2(lbl, dst_lbl / dst_stem)
        else:
            (dst_lbl / dst_stem).write_text("")  # empty hard-negative
        count += 1
    return count


def _remap_label(src_lbl: Path, dst_lbl: Path) -> bool:
    """
    Reads src_lbl, remaps class IDs using EXTRA_CLASS_MAP, writes dst_lbl.
    Returns True if at least one annotation was kept (not all skipped).
    """
    if not src_lbl.exists():
        dst_lbl.write_text("")
        return False  # hard-negative

    mapped: list[str] = []
    for line in src_lbl.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        new_cls = EXTRA_CLASS_MAP.get(cls)
        if new_cls is None:
            continue  # skip this annotation
        mapped.append(f"{new_cls} " + " ".join(parts[1:]))

    dst_lbl.write_text("\n".join(mapped) + ("\n" if mapped else ""), encoding="utf-8")
    return bool(mapped)


def _add_extra_split(src_root: Path, split_name: str,
                     dst_img: Path, dst_lbl: Path, prefix: str) -> tuple[int, int]:
    """
    Adds remapped images from src_root/<split_name> into dst dirs.
    Returns (images_added, hard_negatives_added).
    """
    src_imgs = _img_dir(src_root, split_name)
    src_lbls = _lbl_dir(src_root, split_name)
    if not src_imgs.exists():
        print(f"  [skip] {src_root.name}/{split_name} — images dir not found")
        return 0, 0

    added = hard_neg = 0
    for img in sorted(src_imgs.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue

        out_name = prefix + img.name
        # Skip if already present (guards against accidental duplicate runs)
        if (dst_img / out_name).exists():
            continue

        shutil.copy2(img, dst_img / out_name)
        src_lbl  = src_lbls / (img.stem + ".txt")
        dst_file = dst_lbl  / (Path(out_name).stem + ".txt")
        kept = _remap_label(src_lbl, dst_file)
        if kept:
            added += 1
        else:
            hard_neg += 1

    return added, hard_neg


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("build_v7_dataset.py  —  building v7 training dataset")
    print("=" * 62)

    # --- 0. Validate sources -------------------------------------------------
    for p in (BASE_DS, EXTRA_DS):
        if not p.exists():
            raise FileNotFoundError(f"Source not found: {p}")

    # --- 1. Create output dirs -----------------------------------------------
    for split in ("train", "val"):
        (OUT_DS / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DS / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {OUT_DS}")

    # --- 2. Copy base training set verbatim ----------------------------------
    print("\n[1/3] Copying base training set …")
    base_train_img = _img_dir(BASE_DS, "train")
    base_train_lbl = _lbl_dir(BASE_DS, "train")
    if not base_train_img.exists():
        raise FileNotFoundError(f"Base train images not found: {base_train_img}")

    n_base = _copy_split(
        base_train_img, base_train_lbl,
        OUT_DS / "images" / "train",
        OUT_DS / "labels" / "train",
    )
    print(f"  Copied {n_base:,} base training images")

    # --- 3. Copy val set verbatim (keep identical for comparison) ------------
    print("\n[2/3] Copying val set (unchanged) …")
    n_val = _copy_split(
        _img_dir(BASE_DS, "val"), _lbl_dir(BASE_DS, "val"),
        OUT_DS / "images" / "val",
        OUT_DS / "labels" / "val",
    )
    print(f"  Copied {n_val:,} val images")

    # --- 4. Add extra dataset (train + valid + test all go to train) ---------
    print("\n[3/3] Adding Concrete defect detection dataset (remapped) …")
    PREFIX = "cdd__"  # Concrete Defect Detection prefix — avoids name collisions

    total_added = total_hn = 0
    for split in ("train", "valid", "test"):
        a, h = _add_extra_split(
            EXTRA_DS, split,
            OUT_DS / "images" / "train",
            OUT_DS / "labels" / "train",
            prefix=PREFIX,
        )
        if a + h > 0:
            print(f"  {split:6s}: {a} labeled  +  {h} hard-negatives")
        total_added += a
        total_hn    += h

    # --- 5. Write data.yaml --------------------------------------------------
    yaml_text = (
        f"path: {OUT_DS.as_posix()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "names:\n"
        "- crack\n"
        "- spalling\n"
        "- corrosion\n"
        "- pothole\n"
        "- paint_degradation\n"
    )
    (OUT_DS / "data.yaml").write_text(yaml_text, encoding="utf-8")

    # --- 6. Summary ----------------------------------------------------------
    final_train = len(list((OUT_DS / "images" / "train").glob("*")))
    final_val   = len(list((OUT_DS / "images" / "val"  ).glob("*")))

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(f"  Base training images   : {n_base:>6,}")
    print(f"  Added (labeled)        : {total_added:>6,}")
    print(f"  Added (hard-negatives) : {total_hn:>6,}")
    print(f"  -----------------------------")
    print(f"  Final train total      : {final_train:>6,}")
    print(f"  Final val total        : {final_val:>6,}  (unchanged)")
    print(f"\n  data.yaml → {OUT_DS / 'data.yaml'}")
    print("\nDone. Run train_v7.py next.")


if __name__ == "__main__":
    main()
