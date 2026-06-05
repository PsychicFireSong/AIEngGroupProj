from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_TO_OUTPUT_SPLIT = {"train": "train", "valid": "val", "val": "val", "test": "test"}
SEVERITY_CLASSES = ["minor", "moderate", "critical"]

# Detailed source labels are mapped into the three-stage severity taxonomy.
# Required examples: scaling -> minor, rebar -> critical.
SEVERITY_MAP = {
    "scaling": "minor",
    "crazing": "minor",
    "tile crack": "minor",
    "fessura": "minor",
    "fessura diagonale": "minor",
    "fessura orizzontale": "minor",
    "fessura verticale": "minor",
    "rust stain": "minor",
    "ruststrain": "minor",
    "rolled in scale": "minor",
    "rolled-in scale": "minor",
    "abrasione": "minor",
    "paint drips": "minor",
    "pin holes": "minor",
    "stain marks": "minor",
    "trowels marks": "minor",
    "patches": "minor",
    "scratches": "minor",
    "inclusion": "minor",
    "rough and patchy surface": "minor",
    "efflorescence": "minor",
    "efflorescenza": "minor",
    "moisture marks": "minor",
    "tracce umidita": "minor",
    "dirt": "minor",
    "dirt algae and mold": "minor",
    "dirt mold": "minor",
    "dirty mold": "minor",
    "mold": "minor",
    "fungus": "minor",
    "crack": "moderate",
    "cracking": "moderate",
    "7 cracking": "moderate",
    "7 cr": "moderate",
    "wall crack": "moderate",
    "wall cracks": "moderate",
    "concrete crack": "moderate",
    "concrete cracks": "moderate",
    "horizontal crack": "moderate",
    "vertical crack": "moderate",
    "diagonal crack": "moderate",
    "stairstep crack": "moderate",
    "stair step crack": "moderate",
    "minor crack": "moderate",
    "major crack": "moderate",
    "spalling": "moderate",
    "spallingw": "moderate",
    "spall": "moderate",
    "corrosion": "moderate",
    "rust": "moderate",
    "rusted": "moderate",
    "4 corrosion": "moderate",
    "pitted surface": "moderate",
    "pitting": "moderate",
    "metal corrosion": "moderate",
    "peeling paint": "moderate",
    "paint degradation": "moderate",
    "paint_degradation": "moderate",
    "peeling": "moderate",
    "paint defect": "moderate",
    "paint defects": "moderate",
    "flaking": "moderate",
    "flaking paint": "moderate",
    "flaking plaster": "moderate",
    "damp": "moderate",
    "dampness": "moderate",
    "dampness with fungus": "moderate",
    "water seepage": "moderate",
    "waterseepage": "moderate",
    "water leakage": "moderate",
    "water leak": "moderate",
    "seepage": "moderate",
    "moisture": "moderate",
    "moisture damage": "moderate",
    "rebar": "critical",
    "exposed rebar": "critical",
    "exposed reinforcement": "critical",
    "exposed_reinforcement": "critical",
    "exposed reinforcements": "critical",
    "exposed iron": "critical",
    "esposizione ferri": "critical",
    "delamination": "critical",
    "delaminazione": "critical",
    "honeycombing": "critical",
    "vespai": "critical",
    "6 spalling": "critical",
    "pothole": "critical",
    "potholes": "critical",
    "pot hole": "critical",
    "pot holes": "critical",
    "road pothole": "critical",
    "marked pothole": "critical",
    "asphalt pothole": "critical",
}


def normalize_label(value: str) -> str:
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    return " ".join(value.split())


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip())
    return value.strip("_").lower() or "dataset"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_names(data: dict) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        ordered: list[str] = []
        for key, value in sorted(
            names.items(),
            key=lambda item: (0, int(item[0])) if str(item[0]).isdigit() else (1, str(item[0])),
        ):
            if not str(key).isdigit():
                continue
            index = int(key)
            while len(ordered) <= index:
                ordered.append("")
            ordered[index] = str(value)
        return ordered
    return [str(name) for name in names]


def read_image(path: Path):
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def bbox_from_values(values: list[float]) -> tuple[float, float, float, float] | None:
    if len(values) == 4:
        x_center, y_center, width, height = values
        return x_center, y_center, width, height
    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
    return None


def yolo_to_xyxy(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x_center, y_center, box_width, box_height = box
    x1 = int(round((x_center - box_width / 2) * width))
    y1 = int(round((y_center - box_height / 2) * height))
    x2 = int(round((x_center + box_width / 2) * width))
    y2 = int(round((y_center + box_height / 2) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def ensure_structure(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        for severity in SEVERITY_CLASSES:
            (output_dir / split / severity).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path) -> None:
    data = {
        "path": str(output_dir.resolve()),
        "train": "train",
        "val": "val",
        "test": "test",
        "names": SEVERITY_CLASSES,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def extract_dataset_crops(dataset: dict, output_dir: Path, min_size: int, severity_by_label: dict[str, str]) -> Counter:
    root = Path(dataset["path"])
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    data_path = root / "data.yaml"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.yaml in {root}")

    names = read_names(load_yaml(data_path))
    dataset_key = sanitize_name(dataset.get("name", root.name))
    counts: Counter = Counter()

    for source_split, output_split in SOURCE_TO_OUTPUT_SPLIT.items():
        images_dir = root / source_split / "images"
        labels_dir = root / source_split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = next((images_dir / f"{label_path.stem}{ext}" for ext in IMAGE_EXTS if (images_dir / f"{label_path.stem}{ext}").exists()), None)
            if image_path is None:
                counts["missing_image"] += 1
                continue
            image = read_image(image_path)
            if image is None:
                counts["unreadable_image"] += 1
                continue
            height, width = image.shape[:2]

            for row_index, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = raw_line.strip().split()
                if len(parts) < 5:
                    counts["malformed_label"] += 1
                    continue
                try:
                    class_id = int(float(parts[0]))
                    values = [float(item) for item in parts[1:]]
                except ValueError:
                    counts["bad_label_values"] += 1
                    continue
                if class_id < 0 or class_id >= len(names):
                    counts["class_id_out_of_range"] += 1
                    continue

                original_label = normalize_label(names[class_id])
                severity = severity_by_label.get(original_label)
                if severity is None:
                    counts[f"unmapped/{original_label or '<blank>'}"] += 1
                    continue

                box = bbox_from_values(values)
                if box is None:
                    counts["bad_box"] += 1
                    continue
                x1, y1, x2, y2 = yolo_to_xyxy(box, width, height)
                if x2 - x1 < min_size or y2 - y1 < min_size:
                    counts["too_small"] += 1
                    continue

                crop = image[y1:y2, x1:x2]
                output_name = f"{dataset_key}_{source_split}_{label_path.stem}_{row_index:04d}.jpg"
                output_path = output_dir / output_split / severity / output_name
                cv2.imwrite(str(output_path), crop)
                counts[f"{output_split}/{severity}"] += 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract severity classification crops from original YOLO datasets.")
    parser.add_argument("--config", default="configs/merge_config.yaml")
    parser.add_argument("--output", default="severity_dataset")
    parser.add_argument("--min-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    output_dir = Path(args.output)
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_structure(output_dir)

    severity_by_label = {normalize_label(label): severity for label, severity in SEVERITY_MAP.items()}
    counts: Counter = Counter()
    for dataset in config.get("datasets", []):
        counts.update(extract_dataset_crops(dataset, output_dir, args.min_size, severity_by_label))

    write_data_yaml(output_dir)
    summary = {"output": str(output_dir.resolve()), "counts": dict(counts)}
    (output_dir / "severity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
