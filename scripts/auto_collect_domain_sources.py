from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
TARGET_NAMES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
TARGET_ID = {name: index for index, name in enumerate(TARGET_NAMES)}

EXTRA_ROBOFLOW_SOURCES = [
    {
        "folder": "Facade Defects Detection",
        "workspace": "defects-detection",
        "project": "facade-defects-detection",
        "version": 9,
        "source_url": "https://universe.roboflow.com/defects-detection/facade-defects-detection",
    },
    {
        "folder": "Defects in Facade Building",
        "workspace": "defects-in-facade-building",
        "project": "defects-in-facade-building",
        "version": 1,
        "source_url": "https://universe.roboflow.com/defects-in-facade-building/defects-in-facade-building",
    },
    {
        "folder": "Facade Building Defect",
        "workspace": "defects-in-facade-building",
        "project": "defect-bjlhe",
        "version": 6,
        "source_url": "https://universe.roboflow.com/defects-in-facade-building/defect-bjlhe",
    },
    {
        "folder": "Building Defect V2",
        "workspace": "building-defect-e69vu",
        "project": "building-defectv2-0sl5l",
        "version": 2,
        "source_url": "https://universe.roboflow.com/building-defect-e69vu/building-defectv2-0sl5l",
    },
    {
        "folder": "BuildingDamage Spalling",
        "workspace": "buildingdamage",
        "project": "spalling-wcoze-osekr",
        "version": 3,
        "source_url": "https://universe.roboflow.com/buildingdamage/spalling-wcoze-osekr",
    },
]

LABEL_SYNONYMS = {
    "crack": "crack",
    "cracks": "crack",
    "cracking": "crack",
    "corrosion": "corrosion",
    "rust": "corrosion",
    "rust stain": "corrosion",
    "spalling": "spalling",
    "delamination": "spalling",
    "peeling": "paint_degradation",
    "paint defect": "paint_degradation",
    "paint defects": "paint_degradation",
    "paint_defect": "paint_degradation",
    "dirt": "paint_degradation",
    "dirt algae and mold": "paint_degradation",
    "dirt mold": "paint_degradation",
    "dirty mold": "paint_degradation",
    "mold": "paint_degradation",
}


def normalize_label(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def install_package(package: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_data_yaml(root: Path) -> Path | None:
    candidates = [root / "data.yaml", *root.glob("**/data.yaml")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def download_roboflow_sources(api_key: str, raw_root: Path, force: bool = False) -> list[dict]:
    if not api_key:
        raise ValueError("Roboflow API key is required for automated domain collection.")
    install_package("roboflow")
    from roboflow import Roboflow

    raw_root.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    downloaded = []
    for source in EXTRA_ROBOFLOW_SOURCES:
        destination = raw_root / source["folder"]
        data_yaml = find_data_yaml(destination)
        if data_yaml and not force:
            downloaded.append({**source, "path": str(data_yaml.parent), "status": "cached"})
            continue
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        try:
            project = rf.workspace(source["workspace"]).project(source["project"])
            version = project.version(int(source["version"]))
            last_error: Exception | None = None
            for export_format in ("yolov11", "yolov8"):
                try:
                    version.download(export_format, location=str(destination), overwrite=True)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            data_yaml = find_data_yaml(destination)
            if not data_yaml:
                raise RuntimeError(f"No data.yaml found after downloading {source['folder']}")
            downloaded.append({**source, "path": str(data_yaml.parent), "status": "downloaded"})
        except Exception as exc:
            downloaded.append({**source, "path": "", "status": f"failed: {exc}"})
    return downloaded


def image_label_dirs(root: Path, split: str) -> tuple[Path, Path]:
    direct_images = root / split / "images"
    direct_labels = root / split / "labels"
    if direct_images.exists():
        return direct_images, direct_labels
    return root / "images" / split, root / "labels" / split


def convert_polygon_to_box(coords: list[float]) -> list[float] | None:
    if len(coords) < 4:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


def normalize_line(line: str, names: list[str]) -> tuple[str | None, str | None]:
    parts = line.split()
    if len(parts) < 5:
        return None, "malformed"
    try:
        class_id = int(float(parts[0]))
    except ValueError:
        return None, "bad_class_id"
    if class_id < 0 or class_id >= len(names):
        return None, "class_id_out_of_range"
    unified = LABEL_SYNONYMS.get(normalize_label(str(names[class_id])))
    if unified is None:
        return None, f"unmapped:{names[class_id]}"
    try:
        values = [float(item) for item in parts[1:]]
    except ValueError:
        return None, "bad_coordinates"
    if len(values) == 4:
        x, y, w, h = values
    else:
        box = convert_polygon_to_box(values)
        if box is None:
            return None, "bad_polygon"
        x, y, w, h = box
    if w <= 0 or h <= 0:
        return None, "empty_box"
    return f"{TARGET_ID[unified]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}", None


def append_dataset(dataset_root: Path, output_root: Path, prefix: str) -> Counter:
    data_yaml = find_data_yaml(dataset_root)
    if not data_yaml:
        raise FileNotFoundError(f"Missing data.yaml under {dataset_root}")
    data = load_yaml(data_yaml)
    names = data.get("names") or []
    counts: Counter = Counter()
    for source_split, output_split in SPLIT_ALIASES.items():
        images_dir, labels_dir = image_label_dirs(data_yaml.parent, source_split)
        if not images_dir.exists():
            continue
        out_images = output_root / output_split / "images"
        out_labels = output_root / output_split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        for image_path in images_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            mapped_lines = []
            with label_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    mapped, error = normalize_line(line, names)
                    if mapped:
                        mapped_lines.append(mapped)
                        counts[f"mapped/{output_split}"] += 1
                    elif error:
                        counts[error] += 1
            if not mapped_lines:
                continue
            output_name = f"{prefix}_{output_split}_{image_path.stem}{image_path.suffix.lower()}"
            shutil.copy2(image_path, out_images / output_name)
            (out_labels / f"{Path(output_name).stem}.txt").write_text("\n".join(mapped_lines) + "\n", encoding="utf-8")
            counts[f"images/{output_split}"] += 1
    return counts


def build_target_dataset(downloads: list[dict], output_root: Path, force: bool = False) -> dict:
    if output_root.exists() and force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    total_counts: Counter = Counter()
    source_summaries = []
    for index, item in enumerate(downloads):
        if not item.get("path") or not str(item.get("status", "")).startswith(("downloaded", "cached")):
            source_summaries.append({**item, "counts": {}})
            continue
        prefix = re.sub(r"[^a-zA-Z0-9]+", "_", item["folder"]).strip("_").lower() or f"source_{index}"
        counts = append_dataset(Path(item["path"]), output_root, prefix)
        total_counts.update(counts)
        source_summaries.append({**item, "counts": dict(counts)})
    summary = {
        "output": str(output_root),
        "target_names": TARGET_NAMES,
        "counts": dict(total_counts),
        "sources": source_summaries,
    }
    (output_root / "auto_collection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically collect target-domain facility/facade datasets.")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--raw-output", default="domain_adaptation/auto_raw")
    parser.add_argument("--target-output", default="domain_adaptation/target_yolo")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    downloads = download_roboflow_sources(args.api_key, Path(args.raw_output), args.force_download)
    summary = build_target_dataset(downloads, Path(args.target_output), args.force_rebuild)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
