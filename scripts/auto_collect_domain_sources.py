from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import site
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
        "folder": "Concrete Crack Broad",
        "workspace": "concrete-crack",
        "project": "concrete-crack-6loqq",
        "version": 1,
        "domain_group": "closeup_concrete_crack",
        "source_url": "https://universe.roboflow.com/concrete-crack/concrete-crack-6loqq/dataset/1",
    },
    {
        "folder": "Wall Cracks",
        "workspace": "object-detection-rrer5",
        "project": "wall-cracks",
        "version": 1,
        "domain_group": "wall_crack_scene",
        "source_url": "https://universe.roboflow.com/object-detection-rrer5/wall-cracks/dataset/1",
    },
    {
        "folder": "Pothole Detection YOLOv8 TPLPK",
        "workspace": "tgu-lo22r",
        "project": "pothole-detection-yolov8-tplpk",
        "version": 1,
        "domain_group": "road_pothole_scene",
        "source_url": "https://universe.roboflow.com/tgu-lo22r/pothole-detection-yolov8-tplpk/model/1",
    },
    {
        "folder": "Pothole Detection YOLOv8 Baseline Mirror",
        "workspace": "pe1-dtzop",
        "project": "pothole-detection-yolov8-8mspr",
        "version": 1,
        "domain_group": "road_pothole_scene",
        "source_url": "https://universe.roboflow.com/pe1-dtzop/pothole-detection-yolov8-8mspr",
    },
    {
        "folder": "Facade Defects Detection",
        "workspace": "defects-detection",
        "project": "facade-defects-detection",
        "version": 9,
        "domain_group": "facade_wall_scene",
        "source_url": "https://universe.roboflow.com/defects-detection/facade-defects-detection",
    },
    {
        "folder": "Defects in Facade Building",
        "workspace": "defects-in-facade-building",
        "project": "defects-in-facade-building",
        "version": 1,
        "domain_group": "facade_wall_scene",
        "source_url": "https://universe.roboflow.com/defects-in-facade-building/defects-in-facade-building",
    },
    {
        "folder": "Facade Building Defect",
        "workspace": "defects-in-facade-building",
        "project": "defect-bjlhe",
        "version": 6,
        "domain_group": "facade_wall_scene",
        "source_url": "https://universe.roboflow.com/defects-in-facade-building/defect-bjlhe",
    },
    {
        "folder": "Building Defect V2",
        "workspace": "building-defect-e69vu",
        "project": "building-defectv2-0sl5l",
        "version": 2,
        "domain_group": "mixed_building_scene",
        "source_url": "https://universe.roboflow.com/building-defect-e69vu/building-defectv2-0sl5l",
    },
    {
        "folder": "BuildingDamage Spalling",
        "workspace": "buildingdamage",
        "project": "spalling-wcoze-osekr",
        "version": 3,
        "domain_group": "structural_spalling_scene",
        "source_url": "https://universe.roboflow.com/buildingdamage/spalling-wcoze-osekr",
    },
    {
        "folder": "Building Damage Insurance Wall Defects",
        "workspace": "tennis-jbaz6",
        "project": "building-damage-insurance",
        "version": 1,
        "domain_group": "wall_maintenance_scene",
        "source_url": "https://universe.roboflow.com/tennis-jbaz6/building-damage-insurance",
    },
    {
        "folder": "Building Defect On Walls",
        "workspace": "builddef2",
        "project": "building-defect-on-walls",
        "version": 3,
        "domain_group": "wall_maintenance_scene",
        "source_url": "https://universe.roboflow.com/builddef2/building-defect-on-walls",
    },
    {
        "folder": "Wall Surface Crack Mold Peeling Seepage",
        "workspace": "aakashs-workspace-zqqzu",
        "project": "training-dataset-1gvqr",
        "version": 1,
        "domain_group": "wall_maintenance_scene",
        "source_url": "https://universe.roboflow.com/aakashs-workspace-zqqzu/training-dataset-1gvqr",
    },
]

LABEL_SYNONYMS = {
    "crack": "crack",
    "cracks": "crack",
    "cracking": "crack",
    "wall crack": "crack",
    "wall cracks": "crack",
    "concrete crack": "crack",
    "concrete cracks": "crack",
    "wall concrete cracks": "crack",
    "horizontal crack": "crack",
    "vertical crack": "crack",
    "diagonal crack": "crack",
    "tile crack": "crack",
    "crazing": "crack",
    "stairstep crack": "crack",
    "stair step crack": "crack",
    "stair step": "crack",
    "stair stepcrack": "crack",
    "stairstepcrack": "crack",
    "minor crack": "crack",
    "major crack": "crack",
    "7 cracking": "crack",
    "7 cr": "crack",
    "fessura": "crack",
    "fessura diagonale": "crack",
    "fessura orizzontale": "crack",
    "fessura verticale": "crack",
    "corrosion": "corrosion",
    "corroded": "corrosion",
    "rust": "corrosion",
    "rusted": "corrosion",
    "rust stain": "corrosion",
    "ruststrain": "corrosion",
    "metal corrosion": "corrosion",
    "pitting": "corrosion",
    "pitted surface": "corrosion",
    "rolled in scale": "corrosion",
    "rolled-in scale": "corrosion",
    "4 corrosion": "corrosion",
    "spalling": "spalling",
    "spallingw": "spalling",
    "spall": "spalling",
    "concrete spalling": "spalling",
    "delamination": "spalling",
    "delaminazione": "spalling",
    "scaling": "spalling",
    "honeycombing": "spalling",
    "exposed rebar": "spalling",
    "exposed reinforcement": "spalling",
    "exposed reinforcements": "spalling",
    "exposed iron": "spalling",
    "esposizione ferri": "spalling",
    "spalling and exposed rebar": "spalling",
    "rebar": "spalling",
    "6 spalling": "spalling",
    "vespai": "spalling",
    "pothole": "pothole",
    "potholes": "pothole",
    "pot hole": "pothole",
    "pot holes": "pothole",
    "road pothole": "pothole",
    "marked pothole": "pothole",
    "asphalt pothole": "pothole",
    "abrasione": "paint_degradation",
    "peeling": "paint_degradation",
    "peeling paint": "paint_degradation",
    "paint degradation": "paint_degradation",
    "paint degradation ": "paint_degradation",
    "paint defect": "paint_degradation",
    "paint defects": "paint_degradation",
    "paint_defect": "paint_degradation",
    "peeling_paint": "paint_degradation",
    "flaking": "paint_degradation",
    "flaking paint": "paint_degradation",
    "flaking plaster": "paint_degradation",
    "paint drips": "paint_degradation",
    "pin holes": "paint_degradation",
    "rough and patchy surface": "paint_degradation",
    "stain marks": "paint_degradation",
    "trowels marks": "paint_degradation",
    "patches": "paint_degradation",
    "scratches": "paint_degradation",
    "inclusion": "paint_degradation",
    "efflorescence": "paint_degradation",
    "efflorescenza": "paint_degradation",
    "dampness": "paint_degradation",
    "damp": "paint_degradation",
    "dampness with fungus": "paint_degradation",
    "water seepage": "paint_degradation",
    "waterseepage": "paint_degradation",
    "water leakage": "paint_degradation",
    "water leak": "paint_degradation",
    "seepage": "paint_degradation",
    "moisture": "paint_degradation",
    "moisture damage": "paint_degradation",
    "moisture marks": "paint_degradation",
    "tracce umidita": "paint_degradation",
    "stain": "paint_degradation",
    "stains": "paint_degradation",
    "dirt": "paint_degradation",
    "dirt algae and mold": "paint_degradation",
    "dirt mold": "paint_degradation",
    "dirty mold": "paint_degradation",
    "mold": "paint_degradation",
    "fungus": "paint_degradation",
}

SOURCE_SPECIFIC_LABEL_MAP = {
    # Roboflow lists these two classes literally as "0" and "1" for this
    # wall-damage dataset. Because the same source is otherwise dominated by
    # stain, damp, mold, peeling_paint, and water_seepage classes, keep them
    # under the nearest project taxonomy bucket instead of dropping them.
    "building damage insurance wall defects": {
        "0": "paint_degradation",
        "1": "paint_degradation",
    },
    "building damage insurance": {
        "0": "paint_degradation",
        "1": "paint_degradation",
    },
}


def normalize_label(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def map_source_label(value: str, source_key: str = "") -> tuple[str | None, str | None]:
    normalized = normalize_label(value)
    source_mapping = SOURCE_SPECIFIC_LABEL_MAP.get(normalize_label(source_key), {})
    if normalized in source_mapping:
        return source_mapping[normalized], None
    mapped = LABEL_SYNONYMS.get(normalized)
    if mapped:
        return mapped, None
    without_numeric_prefix = re.sub(r"^\d+\s+", "", normalized).strip()
    if without_numeric_prefix and without_numeric_prefix != normalized:
        mapped = LABEL_SYNONYMS.get(without_numeric_prefix)
        if mapped:
            return mapped, None
    return None, None


def refresh_python_package_paths() -> None:
    candidates: list[str] = []
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    for candidate in candidates:
        if candidate and Path(candidate).exists() and candidate not in sys.path:
            sys.path.insert(0, candidate)
    importlib.invalidate_caches()


def install_package(package: str, import_name: str | None = None) -> None:
    import_name = import_name or package.replace("-", "_")
    refresh_python_package_paths()
    if importlib.util.find_spec(import_name) is not None:
        return
    commands = [
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", package],
    ]
    last_error: Exception | None = None
    for command in commands:
        try:
            subprocess.run(command, check=True)
            refresh_python_package_paths()
            if importlib.util.find_spec(import_name) is not None:
                return
        except Exception as exc:
            last_error = exc
    runtime_target = Path(".runtime_python_packages").resolve()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "--target", str(runtime_target), package],
        check=True,
    )
    if str(runtime_target) not in sys.path:
        sys.path.insert(0, str(runtime_target))
    importlib.invalidate_caches()
    if importlib.util.find_spec(import_name) is None:
        raise ModuleNotFoundError(f"Installed {package}, but {import_name} is still not importable.") from last_error


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


def find_data_yaml(root: Path) -> Path | None:
    candidates = [root / "data.yaml", *root.glob("**/data.yaml")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def download_roboflow_sources(api_key: str, raw_root: Path, force: bool = False) -> list[dict]:
    if not api_key:
        raise ValueError("Roboflow API key is required for automated domain collection.")
    # Roboflow's downloader imports these lazily. In Colab, a previous no-deps
    # install can leave roboflow importable but downloads still broken.
    for package, import_name in [
        ("filetype", "filetype"),
        ("pillow-avif-plugin", "pillow_avif"),
        ("python-dotenv", "dotenv"),
        ("requests-toolbelt", "requests_toolbelt"),
    ]:
        install_package(package, import_name)
    try:
        import pillow_avif  # noqa: F401
    except Exception:
        pass
    install_package("roboflow", "roboflow")
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


def normalize_line(line: str, names: list[str], source_key: str = "") -> tuple[str | None, str | None]:
    parts = line.split()
    if len(parts) < 5:
        return None, "malformed"
    try:
        class_id = int(float(parts[0]))
    except ValueError:
        return None, "bad_class_id"
    if class_id < 0 or class_id >= len(names):
        return None, "class_id_out_of_range"
    source_label = str(names[class_id])
    unified, ignore_reason = map_source_label(source_label, source_key)
    if ignore_reason:
        return None, f"ignored:{ignore_reason}:{source_label}"
    if unified is None:
        return None, f"unmapped:{source_label}"
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
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return None, "box_out_of_range"
    if x - w / 2 < -0.03 or x + w / 2 > 1.03 or y - h / 2 < -0.03 or y + h / 2 > 1.03:
        return None, "box_extends_too_far"
    return f"{TARGET_ID[unified]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}", None


def append_dataset(dataset_root: Path, output_root: Path, prefix: str, source_key: str = "") -> Counter:
    data_yaml = find_data_yaml(dataset_root)
    if not data_yaml:
        raise FileNotFoundError(f"Missing data.yaml under {dataset_root}")
    data = load_yaml(data_yaml)
    names = read_names(data)
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
                    mapped, error = normalize_line(line, names, source_key or dataset_root.name)
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


def write_target_data_yaml(output_root: Path) -> None:
    data = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": TARGET_NAMES,
    }
    with (output_root / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def build_target_dataset(downloads: list[dict], output_root: Path, force: bool = False, fail_on_unmapped: bool = True) -> dict:
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
        counts = append_dataset(Path(item["path"]), output_root, prefix, item.get("folder", prefix))
        total_counts.update(counts)
        source_summaries.append({**item, "counts": dict(counts)})
    summary = {
        "output": str(output_root),
        "target_names": TARGET_NAMES,
        "counts": dict(total_counts),
        "sources": source_summaries,
    }
    write_target_data_yaml(output_root)
    (output_root / "auto_collection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    unmapped = {key: value for key, value in total_counts.items() if str(key).startswith("unmapped:")}
    if fail_on_unmapped and unmapped:
        raise RuntimeError(
            "Unmapped source labels remain after target conversion. "
            "Add them to LABEL_SYNONYMS instead of silently discarding them: "
            + json.dumps(unmapped, sort_keys=True)
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically collect target-domain facility/facade datasets.")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--raw-output", default="domain_adaptation/auto_raw")
    parser.add_argument("--target-output", default="domain_adaptation/target_yolo")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--allow-unmapped", action="store_true")
    args = parser.parse_args()

    downloads = download_roboflow_sources(args.api_key, Path(args.raw_output), args.force_download)
    summary = build_target_dataset(downloads, Path(args.target_output), args.force_rebuild, fail_on_unmapped=not args.allow_unmapped)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
