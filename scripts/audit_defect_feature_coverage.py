from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_names(data_yaml: Path) -> list[str]:
    data = load_yaml(data_yaml)
    names = data.get("names", [])
    if isinstance(names, dict):
        ordered: list[str] = []
        for key, value in sorted(names.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 9999):
            if not str(key).isdigit():
                continue
            index = int(key)
            while len(ordered) <= index:
                ordered.append("")
            ordered[index] = str(value)
        return ordered
    return [str(name) for name in names]


def parse_labels(path: Path, names: list[str]) -> list[dict]:
    labels: list[dict] = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, width, height = [float(item) for item in parts[1:]]
        except ValueError:
            continue
        if 0 <= class_id < len(names) and 0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1:
            labels.append({"class_name": names[class_id], "x": x, "y": y, "w": width, "h": height})
    return labels


def image_label_dirs(root: Path, split: str) -> tuple[Path, Path]:
    direct_images = root / split / "images"
    direct_labels = root / split / "labels"
    if direct_images.exists():
        return direct_images, direct_labels
    return root / "images" / split, root / "labels" / split


def crop_for_label(image: np.ndarray, label: dict, pad: float = 0.05) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1 = int((label["x"] - label["w"] / 2 - pad * label["w"]) * width)
    y1 = int((label["y"] - label["h"] / 2 - pad * label["h"]) * height)
    x2 = int((label["x"] + label["w"] / 2 + pad * label["w"]) * width)
    y2 = int((label["y"] + label["h"] / 2 + pad * label["h"]) * height)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def bin_value(value: float, thresholds: list[float], labels: list[str]) -> str:
    for threshold, label in zip(thresholds, labels):
        if value < threshold:
            return label
    return labels[-1]


def feature_bins(image: np.ndarray, label: dict) -> dict[str, str]:
    crop = crop_for_label(image, label)
    if crop is None or crop.size == 0:
        return {}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)
    area_ratio = float(label["w"] * label["h"])
    aspect = max(label["w"], label["h"]) / max(min(label["w"], label["h"]), 1e-6)
    edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(hsv[..., 2]))
    saturation = float(np.mean(hsv[..., 1]))
    return {
        "scale": bin_value(area_ratio, [0.006, 0.025, 0.12], ["tiny", "small", "medium", "large"]),
        "aspect": "thin_linear" if aspect >= 4.0 else "elongated" if aspect >= 2.0 else "blocky",
        "edge_density": bin_value(edge_density, [0.035, 0.12], ["sparse_edges", "moderate_edges", "dense_edges"]),
        "texture": bin_value(texture, [18.0, 90.0], ["low_texture", "medium_texture", "high_texture"]),
        "brightness": bin_value(brightness, [75.0, 185.0], ["dark", "normal_light", "bright"]),
        "saturation": bin_value(saturation, [45.0, 130.0], ["low_saturation", "medium_saturation", "high_saturation"]),
    }


def audit_dataset(root: Path, args: argparse.Namespace) -> dict:
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")
    names = read_names(data_yaml)
    counts = {split: {name: 0 for name in names} for split in ("train", "val", "test")}
    feature_counts: dict[str, dict[str, dict[str, Counter]]] = {
        split: {name: defaultdict(Counter) for name in names} for split in ("train", "val", "test")
    }
    unreadable = Counter()
    empty_labels = Counter()

    for split in ("train", "val", "test"):
        images_dir, labels_dir = image_label_dirs(root, split)
        if not images_dir.exists():
            continue
        for image_path in images_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            labels = parse_labels(label_path, names)
            if not labels:
                empty_labels[split] += 1
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                unreadable[split] += 1
                continue
            for label in labels:
                class_name = label["class_name"]
                counts[split][class_name] += 1
                bins = feature_bins(image, label)
                for feature_name, bucket in bins.items():
                    feature_counts[split][class_name][feature_name][bucket] += 1

    feature_summary = {}
    warnings = []
    for split in ("train", "val", "test"):
        feature_summary[split] = {}
        for class_name in names:
            feature_summary[split][class_name] = {
                feature: dict(counter)
                for feature, counter in feature_counts[split][class_name].items()
            }

    for class_name in names:
        train_count = counts["train"].get(class_name, 0)
        val_count = counts["val"].get(class_name, 0)
        if train_count < args.min_train_boxes_per_class:
            warnings.append(f"{class_name}: train boxes below minimum ({train_count} < {args.min_train_boxes_per_class})")
        if val_count < args.min_val_boxes_per_class:
            warnings.append(f"{class_name}: val boxes below minimum ({val_count} < {args.min_val_boxes_per_class})")
        if train_count >= args.min_train_boxes_per_class:
            for feature_name in ("scale", "edge_density", "texture", "brightness"):
                represented = sum(
                    1 for value in feature_counts["train"][class_name][feature_name].values() if value >= args.min_bin_boxes
                )
                if represented < args.min_feature_bins:
                    warnings.append(
                        f"{class_name}: weak {feature_name} diversity ({represented} bins with >= {args.min_bin_boxes} boxes)"
                    )

    train_values = [counts["train"].get(name, 0) for name in names if counts["train"].get(name, 0) > 0]
    train_ratio = max(train_values) / min(train_values) if train_values else math.inf
    guard = {
        "ok": not any("below minimum" in warning for warning in warnings)
        and train_ratio <= args.max_train_class_ratio,
        "train_max_min_ratio": train_ratio,
        "max_train_class_ratio": args.max_train_class_ratio,
        "warning_count": len(warnings),
    }
    return {
        "dataset": str(root),
        "names": names,
        "parameters": vars(args),
        "class_counts_by_split": counts,
        "feature_counts_by_split": feature_summary,
        "empty_label_images": dict(empty_labels),
        "unreadable_images": dict(unreadable),
        "warnings": warnings,
        "guard": guard,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit YOLO defect dataset coverage across CV feature cues.")
    parser.add_argument("--dataset", default="merged_dataset_anchor_robust")
    parser.add_argument("--output", default="output/defect_feature_coverage_summary.json")
    parser.add_argument("--min-train-boxes-per-class", type=int, default=1200)
    parser.add_argument("--min-val-boxes-per-class", type=int, default=30)
    parser.add_argument("--max-train-class-ratio", type=float, default=6.0)
    parser.add_argument("--min-feature-bins", type=int, default=2)
    parser.add_argument("--min-bin-boxes", type=int, default=40)
    args = parser.parse_args()

    summary = audit_dataset(Path(args.dataset), args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["guard"]["ok"]:
        raise SystemExit("Feature coverage guard failed: " + json.dumps(summary["guard"], sort_keys=True))


if __name__ == "__main__":
    main()
