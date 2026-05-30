from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def image_label_dirs(root: Path, split: str) -> tuple[Path, Path]:
    return root / "images" / split, root / "labels" / split


def source_image_label_dirs(root: Path, split: str) -> tuple[Path, Path]:
    direct_images = root / split / "images"
    direct_labels = root / split / "labels"
    if direct_images.exists():
        return direct_images, direct_labels
    return root / "images" / split, root / "labels" / split


def ensure_output_structure(output: Path) -> None:
    for split in ("train", "val", "test"):
        images_dir, labels_dir = image_label_dirs(output, split)
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)


def copy_base_dataset(base: Path, output: Path) -> Counter:
    counts: Counter = Counter()
    for split in ("train", "val", "test"):
        src_images, src_labels = image_label_dirs(base, split)
        dst_images, dst_labels = image_label_dirs(output, split)
        if not src_images.exists():
            continue
        for image_path in src_images.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = src_labels / f"{image_path.stem}.txt"
            shutil.copy2(image_path, dst_images / image_path.name)
            if label_path.exists():
                shutil.copy2(label_path, dst_labels / label_path.name)
            else:
                (dst_labels / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
            counts[f"base/{split}"] += 1
    return counts


def copy_target_dataset(target: Path, output: Path, repeat: int, prefix: str) -> Counter:
    counts: Counter = Counter()
    if not target.exists():
        return counts
    for source_split, output_split in SPLIT_ALIASES.items():
        src_images, src_labels = source_image_label_dirs(target, source_split)
        if not src_images.exists():
            continue
        dst_images, dst_labels = image_label_dirs(output, output_split)
        for image_path in src_images.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = src_labels / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            label_text = label_path.read_text(encoding="utf-8").strip()
            if not label_text:
                continue
            copies = repeat if output_split == "train" else 1
            for copy_index in range(copies):
                suffix = f"_r{copy_index}" if copies > 1 else ""
                output_name = f"{prefix}_{output_split}_{image_path.stem}{suffix}{image_path.suffix.lower()}"
                output_label = f"{Path(output_name).stem}.txt"
                shutil.copy2(image_path, dst_images / output_name)
                shutil.copy2(label_path, dst_labels / output_label)
                counts[f"target/{output_split}"] += 1
    return counts


def copy_hard_negatives(hard_negatives: Path, output: Path, repeat: int, prefix: str) -> Counter:
    counts: Counter = Counter()
    if not hard_negatives.exists():
        return counts
    for source_split, output_split in SPLIT_ALIASES.items():
        src_images, _ = source_image_label_dirs(hard_negatives, source_split)
        if not src_images.exists():
            continue
        dst_images, dst_labels = image_label_dirs(output, output_split)
        for image_path in src_images.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            copies = repeat if output_split == "train" else 1
            for copy_index in range(copies):
                suffix = f"_r{copy_index}" if copies > 1 else ""
                output_name = f"{prefix}_{output_split}_{image_path.stem}{suffix}{image_path.suffix.lower()}"
                shutil.copy2(image_path, dst_images / output_name)
                (dst_labels / f"{Path(output_name).stem}.txt").write_text("", encoding="utf-8")
                counts[f"hard_negative/{output_split}"] += 1
    return counts


def count_classes(output: Path, names: list[str]) -> dict:
    class_counts = Counter()
    empty_labels = 0
    for label_path in (output / "labels").glob("*/*.txt"):
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            empty_labels += 1
            continue
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            class_counts[name] += 1
    return {"class_boxes": dict(class_counts), "empty_label_images": empty_labels}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a domain-adaptation detection dataset.")
    parser.add_argument("--base", default="merged_dataset", help="Base merged YOLO dataset")
    parser.add_argument("--target", default="domain_adaptation/target_yolo", help="Manually annotated target-domain YOLO dataset")
    parser.add_argument("--hard-negatives", default="domain_adaptation/hard_negatives", help="Images with no defect labels")
    parser.add_argument("--output", default="merged_dataset_domain_adapted")
    parser.add_argument("--target-repeat", type=int, default=3, help="Oversample target train images")
    parser.add_argument("--negative-repeat", type=int, default=2, help="Oversample hard-negative train images")
    args = parser.parse_args()

    base = Path(args.base)
    target = Path(args.target)
    hard_negatives = Path(args.hard_negatives)
    output = Path(args.output)

    data_yaml = base / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing base data.yaml: {data_yaml}")

    base_data = load_yaml(data_yaml)
    names = base_data.get("names") or []
    clean_dir(output)
    ensure_output_structure(output)

    counts = Counter()
    counts.update(copy_base_dataset(base, output))
    counts.update(copy_target_dataset(target, output, max(args.target_repeat, 1), "target"))
    counts.update(copy_hard_negatives(hard_negatives, output, max(args.negative_repeat, 1), "negative"))

    save_yaml(
        output / "data.yaml",
        {
            "path": str(output.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names,
        },
    )

    summary = {
        "base": str(base),
        "target": str(target),
        "hard_negatives": str(hard_negatives),
        "output": str(output),
        "counts": dict(counts),
        "labels": count_classes(output, names),
    }
    (output / "domain_adaptation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
