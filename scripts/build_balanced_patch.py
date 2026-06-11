"""
build_balanced_patch.py
Creates a balanced patch dataset by extracting images from the original source
datasets for each class. Remaps class IDs to our 5-class schema.

Our class IDs:
  0 = crack
  1 = spalling
  2 = corrosion
  3 = pothole
  4 = paint_degradation

Source datasets and their class mappings:
  crack (0):      Concrete defect detection.yolov11  class 4 -> 0
  spalling (1):   Concrete defect detection.yolov11  class 3 -> 1
  corrosion (2):  Corrosion YOLOv8.yolov11            class 0 -> 2
  pothole (3):    Pothole detection YOLOv8.yolov11    class 0 -> 3
  paint_deg (4):  Internal Wall Defect.yolov11        class 1 -> 4  (peeling paint)

Strategy:
  - For each class, scan the source dataset labels directory
  - Keep only images that contain at least one annotation of that class
  - Write remapped label files (only target-class lines kept)
  - Target ~500 images per class; subsample if over, repeat if under
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import shutil, random, textwrap

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE          = Path(r"C:\Users\User\AIEngGroupProj")
MAIN_TRAIN    = BASE / "output/continued_hn_weak/current/datasets/merged_dataset_hn_weak_continue/images/train"
MAIN_VAL      = BASE / "output/continued_hn_weak/current/datasets/merged_dataset_hn_weak_continue/images/val"
OUT_BASE      = BASE / "output/patch_round6/balanced_patch"
YAML_OUT      = BASE / "output/patch_round6/dataset/data.yaml"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".PNG", ".JPEG", ".BMP", ".WEBP"}

# ── Per-class extraction specs ─────────────────────────────────────────────
# (our_class_id, our_class_name, source_dataset_dir, source_class_id_to_keep, target_images)
SPECS = [
    (0, "crack",
     BASE / "Concrete defect detection.yolov11",
     {4: 0},       # source class 4 (crack) -> our 0
     500),
    (1, "spalling",
     BASE / "Concrete defect detection.yolov11",
     {3: 1},       # source class 3 (Spalling) -> our 1
     500),
    (2, "corrosion",
     BASE / "Corrosion YOLOv8.yolov11",
     {0: 2},       # source class 0 (Corrosion) -> our 2
     500),
    (3, "pothole",
     BASE / "Pothole detection YOLOv8.yolov11",
     {0: 3},       # source class 0 (Potholes) -> our 3
     425),         # use all 425 available
    (4, "paint_degradation",
     BASE / "Internal Wall Defect.yolov11",
     {1: 4},       # source class 1 (peeling paint) -> our 4
     500),
]


def find_img(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def find_labels_dir(dataset_root: Path) -> Path | None:
    for candidate in [
        dataset_root / "train" / "labels",
        dataset_root / "labels" / "train",
        dataset_root / "valid" / "labels",
        dataset_root / "labels" / "valid",
    ]:
        if candidate.exists():
            return candidate
    return None


def find_images_dir(dataset_root: Path) -> Path | None:
    for candidate in [
        dataset_root / "train" / "images",
        dataset_root / "images" / "train",
    ]:
        if candidate.exists():
            return candidate
    return None


def extract_class(
    our_id: int,
    cls_name: str,
    src_root: Path,
    class_map: dict[int, int],
    target_n: int,
) -> Path | None:
    lbl_dir = find_labels_dir(src_root)
    img_dir = find_images_dir(src_root)
    if lbl_dir is None or img_dir is None:
        print(f"  [{cls_name}] ERROR: cannot find labels/images in {src_root}")
        return None

    # Collect all images containing at least one annotation of source classes
    candidates: list[tuple[Path, Path, list[str]]] = []
    for lf in sorted(lbl_dir.glob("*.txt")):
        lines = [l.strip() for l in lf.read_text(encoding="utf-8", errors="ignore")
                 .strip().splitlines() if l.strip()]
        keep = [l for l in lines if int(l.split()[0]) in class_map]
        if not keep:
            continue
        ip = find_img(img_dir, lf.stem)
        if ip is None:
            continue
        candidates.append((ip, lf, keep))

    print(f"  [{cls_name}] found {len(candidates)} source images -> target {target_n}")

    # Sub/over-sample to target_n
    if len(candidates) >= target_n:
        chosen = random.sample(candidates, target_n)
        repeats = 1
    else:
        repeats = max(1, -(-target_n // len(candidates)))  # ceiling div
        chosen = candidates

    # Write out
    out_img = OUT_BASE / f"cls{our_id}_{cls_name}" / "images" / "train"
    out_lbl = OUT_BASE / f"cls{our_id}_{cls_name}" / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    written = 0
    for r in range(repeats):
        for ip, lf, keep_lines in chosen:
            suffix   = ip.suffix
            new_stem = f"{ip.stem}_r{r}" if repeats > 1 else ip.stem
            dst_img  = out_img / f"{new_stem}{suffix}"
            dst_lbl  = out_lbl / f"{new_stem}.txt"
            if not dst_img.exists():
                shutil.copy2(ip, dst_img)
            if not dst_lbl.exists():
                remapped = []
                for line in keep_lines:
                    parts   = line.split()
                    src_cls = int(parts[0])
                    new_cls = class_map[src_cls]
                    remapped.append(f"{new_cls} " + " ".join(parts[1:]))
                dst_lbl.write_text("\n".join(remapped) + "\n", encoding="utf-8")
            written += 1

    inst_count = sum(len(k) for _, _, k in chosen) * repeats
    print(f"           -> {written} files written, ~{inst_count} instances "
          f"(repeat={repeats}x)")
    return out_img


def build() -> None:
    print("=" * 60)
    print("Building balanced patch dataset (v6)")
    print("=" * 60)

    train_dirs: list[str] = []

    for our_id, cls_name, src_root, class_map, target_n in SPECS:
        out_img = extract_class(our_id, cls_name, src_root, class_map, target_n)
        if out_img is not None:
            train_dirs.append(str(out_img).replace("\\", "/"))

    # Main training set last
    train_dirs.append(str(MAIN_TRAIN).replace("\\", "/"))

    # Write YAML
    YAML_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["train:"]
    for d in train_dirs:
        lines.append(f"  - {d}")
    lines += [
        f"val: {str(MAIN_VAL).replace(chr(92), '/')}",
        "",
        "nc: 5",
        "names: ['crack', 'spalling', 'corrosion', 'pothole', 'paint_degradation']",
    ]
    YAML_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nYAML written: {YAML_OUT}")
    print("\nTrain sources:")
    for d in train_dirs:
        print(f"  {d}")
    print(f"\nVal: {MAIN_VAL}")
    print("\nDone. Run patch_finetune_stage1_v6.py next.")


if __name__ == "__main__":
    build()
