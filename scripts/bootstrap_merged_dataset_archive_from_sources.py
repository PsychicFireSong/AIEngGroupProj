from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml


DATASET_SOURCES = [
    {
        "folder": "Finale.yolov11",
        "provider": "roboflow",
        "workspace": "ponti",
        "project": "finale-3pwus",
        "version": 1,
        "expected_unified": ["crack", "spalling", "paint_degradation"],
        "min_expected_matches": 3,
    },
    {
        "folder": "Internal Wall Defect.yolov11",
        "provider": "roboflow",
        "workspace": "chew-poh-yee",
        "project": "internal-wall-defect",
        "version": 1,
        "expected_unified": ["paint_degradation"],
        "min_expected_matches": 1,
    },
    {
        "folder": "Concrete defect detection.yolov11",
        "provider": "roboflow",
        "workspace": "defect-detection-0atjo",
        "project": "concrete-defect-detection-zuym8",
        "version": 1,
        "expected_unified": ["crack", "spalling"],
        "min_expected_matches": 2,
    },
    {
        "folder": "Corrosion YOLOv8.yolov11",
        "provider": "roboflow",
        "workspace": "corrosion-yolo-v8",
        "project": "corrosion-yolov8",
        "version": 1,
        "expected_unified": ["corrosion"],
        "min_expected_matches": 1,
    },
    {
        "folder": "metal corrosion.yolov11",
        "provider": "roboflow",
        "workspace": "yolov11-eorob",
        "project": "metal-corrosion",
        "version": 1,
        "version_candidates": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "expected_unified": ["corrosion", "paint_degradation"],
        "min_expected_matches": 2,
    },
    {
        "folder": "Pothole detection YOLOv8.yolov11",
        "provider": "roboflow",
        "workspace": "pe1-dtzop",
        "project": "pothole-detection-yolov8-8mspr",
        "version": 1,
        "expected_unified": ["pothole"],
        "min_expected_matches": 1,
    },
    {
        "folder": "archive",
        "provider": "kaggle",
        "dataset": "muskanverma24/pothole-detection-dataset-yolov11-optimized",
        "expected_unified": ["pothole"],
        "min_expected_matches": 1,
    },
]

REQUIRED_DETECTION_CLASSES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
SOURCE_LABEL_SYNONYMS = {
    "crack": "crack",
    "cracks": "crack",
    "cracking": "crack",
    "wall crack": "crack",
    "concrete crack": "crack",
    "fessura": "crack",
    "fessura diagonale": "crack",
    "fessura orizzontale": "crack",
    "fessura verticale": "crack",
    "spalling": "spalling",
    "spall": "spalling",
    "rebar": "spalling",
    "scaling": "spalling",
    "exposed reinforcement": "spalling",
    "exposed iron": "spalling",
    "esposizione ferri": "spalling",
    "delamination": "spalling",
    "delaminazione": "spalling",
    "honeycombing": "spalling",
    "vespai": "spalling",
    "corrosion": "corrosion",
    "rust": "corrosion",
    "rust stain": "corrosion",
    "ruststrain": "corrosion",
    "metal corrosion": "corrosion",
    "pitted surface": "corrosion",
    "rolled-in scale": "corrosion",
    "rolled in scale": "corrosion",
    "pothole": "pothole",
    "potholes": "pothole",
    "pot hole": "pothole",
    "road pothole": "pothole",
    "paint degradation": "paint_degradation",
    "paint_degradation": "paint_degradation",
    "abrasione": "paint_degradation",
    "peeling paint": "paint_degradation",
    "paint defect": "paint_degradation",
    "paint defects": "paint_degradation",
    "stain marks": "paint_degradation",
    "efflorescence": "paint_degradation",
    "efflorescenza": "paint_degradation",
    "moisture marks": "paint_degradation",
    "tracce umidita": "paint_degradation",
    "patches": "paint_degradation",
    "scratches": "paint_degradation",
    "mold": "paint_degradation",
    "dirt": "paint_degradation",
    "inclusion": "paint_degradation",
    "pin holes": "paint_degradation",
    "rough and patchy surface": "paint_degradation",
    "paint drips": "paint_degradation",
    "trowels marks": "paint_degradation",
}


def run(args: list[str], cwd: Path | None = None) -> None:
    print("Running:", " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=str(cwd) if cwd else None, check=True)


def install_package(package: str) -> None:
    module = package.replace("-", "_")
    if importlib.util.find_spec(module) is not None:
        return
    run([sys.executable, "-m", "pip", "install", "-q", package])


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def read_names(data: dict) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        ordered = []
        for key, value in sorted(names.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 9999):
            if not str(key).isdigit():
                continue
            index = int(key)
            while len(ordered) <= index:
                ordered.append("")
            ordered[index] = str(value)
        return ordered
    return [str(name) for name in names]


def normalize_label(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def dataset_ready(path: Path) -> bool:
    return (path / "data.yaml").exists() and any((path / split / "images").exists() for split in ("train", "valid", "val", "test"))


def dataset_profile(path: Path) -> dict:
    names = []
    if (path / "data.yaml").exists():
        names = read_names(load_yaml(path / "data.yaml"))
    return {"path": str(path), "ready": dataset_ready(path), "names": names}


def unified_labels_from_profile(profile: dict) -> set[str]:
    return {
        mapped
        for name in profile.get("names", [])
        for mapped in [SOURCE_LABEL_SYNONYMS.get(normalize_label(name))]
        if mapped
    }


def dataset_ready_for_source(source: dict, path: Path) -> bool:
    if not dataset_ready(path):
        return False
    expected = set(source.get("expected_unified", []))
    if not expected:
        return True
    profile = dataset_profile(path)
    matched = unified_labels_from_profile(profile) & expected
    return len(matched) >= int(source.get("min_expected_matches", 1))


def find_dataset_root(source: dict, search_root: Path) -> Path | None:
    roots = []
    for data_yaml in search_root.rglob("data.yaml"):
        root = data_yaml.parent
        if dataset_ready_for_source(source, root):
            score = 0
            compact_path = "".join(ch.lower() for ch in str(root.relative_to(search_root)) if ch.isalnum())
            for key in [source["folder"], source.get("project", ""), source.get("dataset", "").split("/")[-1]]:
                compact_key = "".join(ch.lower() for ch in str(key) if ch.isalnum())
                if compact_key and compact_key in compact_path:
                    score += 2
            roots.append((score, len(root.parts), root))
    if not roots:
        return None
    roots.sort(key=lambda item: (-item[0], item[1]))
    return roots[0][2]


def materialize_dataset(source: dict, downloaded_root: Path, repo_root: Path, force: bool) -> bool:
    dataset_root = find_dataset_root(source, downloaded_root)
    if dataset_root is None:
        return False
    target = repo_root / source["folder"]
    if target.exists() and force:
        shutil.rmtree(target)
    if not target.exists():
        shutil.copytree(dataset_root, target)
    return dataset_ready_for_source(source, target)


def download_roboflow_source(source: dict, rf, repo_root: Path, downloads_root: Path, force: bool) -> bool:
    if dataset_ready_for_source(source, repo_root / source["folder"]) and not force:
        return True
    destination = downloads_root / source["folder"]
    if destination.exists() and force:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    project = rf.workspace(source["workspace"]).project(source["project"])
    versions = []
    for value in [source.get("version"), *source.get("version_candidates", [])]:
        if value is not None and int(value) not in versions:
            versions.append(int(value))
    last_error = None
    for version_number in versions:
        for export_format in ("yolov11", "yolov8"):
            try:
                print(f"Downloading {source['folder']} v{version_number} as {export_format}")
                project.version(version_number).download(export_format, location=str(destination), overwrite=True)
                if materialize_dataset(source, destination, repo_root, force=force):
                    return True
            except Exception as exc:
                last_error = exc
                print(f"  failed: {exc}")
    if last_error:
        raise last_error
    return False


def download_kaggle_source(source: dict, repo_root: Path, downloads_root: Path, force: bool, username: str, key: str) -> bool:
    if dataset_ready_for_source(source, repo_root / source["folder"]) and not force:
        return True
    destination = downloads_root / source["folder"]
    if destination.exists() and force:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        install_package("kagglehub")
        import kagglehub

        downloaded = Path(kagglehub.dataset_download(source["dataset"]))
        if materialize_dataset(source, downloaded, repo_root, force=force):
            return True
    except Exception as exc:
        print(f"kagglehub failed: {exc}")
    if username and key:
        install_package("kaggle")
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
        kaggle_json.chmod(0o600)
        run(["kaggle", "datasets", "download", "-d", source["dataset"], "-p", str(destination), "--unzip"])
        return materialize_dataset(source, destination, repo_root, force=force)
    return False


def dataset_counts(dataset_root: Path) -> dict[str, dict[str, int]]:
    counts = {split: {name: 0 for name in REQUIRED_DETECTION_CLASSES} for split in ["train", "val", "test"]}
    for split in counts:
        labels_dir = dataset_root / "labels" / split
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= class_id < len(REQUIRED_DETECTION_CLASSES):
                    counts[split][REQUIRED_DETECTION_CLASSES[class_id]] += 1
    return counts


def validate_merged_dataset(dataset_root: Path, min_train: int, min_val: int) -> dict:
    counts = dataset_counts(dataset_root)
    missing = [
        name
        for name in REQUIRED_DETECTION_CLASSES
        if counts["train"].get(name, 0) < min_train or counts["val"].get(name, 0) < min_val
    ]
    summary = {"dataset_root": str(dataset_root), "counts": counts, "ok": not missing, "missing_or_low": missing}
    if missing:
        raise RuntimeError(json.dumps(summary, indent=2))
    return summary


def create_archive(dataset_root: Path, output: Path, dataset_name: str = "merged_dataset") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    started = time.perf_counter()
    with tarfile.open(tmp, "w:gz") as archive:
        archive.add(dataset_root, arcname=dataset_name)
    tmp.replace(output)
    print(f"Archive created: {output} ({time.perf_counter() - started:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download baseline sources directly, merge them, and create merged_dataset.tar.gz.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--roboflow-api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--kaggle-username", default=os.environ.get("KAGGLE_USERNAME", ""))
    parser.add_argument("--kaggle-key", default=os.environ.get("KAGGLE_KEY", ""))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    downloads_root = repo_root / "raw_source_downloads"
    downloads_root.mkdir(parents=True, exist_ok=True)

    if args.roboflow_api_key:
        install_package("roboflow")
        from roboflow import Roboflow

        rf = Roboflow(api_key=args.roboflow_api_key)
    else:
        rf = None

    failures = []
    for source in DATASET_SOURCES:
        try:
            if source["provider"] == "roboflow":
                if rf is None:
                    raise RuntimeError("ROBOFLOW_API_KEY is required for Roboflow baseline sources.")
                ok = download_roboflow_source(source, rf, repo_root, downloads_root, force=args.force)
            else:
                ok = download_kaggle_source(
                    source,
                    repo_root,
                    downloads_root,
                    force=args.force,
                    username=args.kaggle_username,
                    key=args.kaggle_key,
                )
            print(f"{'Ready' if ok else 'Missing'}: {source['folder']}")
            if not ok:
                failures.append(source["folder"])
        except Exception as exc:
            failures.append(source["folder"])
            print(f"FAILED {source['folder']}: {exc}")

    if failures:
        raise RuntimeError(f"Source downloads failed: {failures}")

    run(
        [
            sys.executable,
            "scripts/merge_datasets.py",
            "--config",
            "configs/merge_config.yaml",
            "--preserve-splits",
            "--force",
        ],
        cwd=repo_root,
    )
    merged = repo_root / "merged_dataset"
    summary = validate_merged_dataset(merged, min_train=1, min_val=1)
    create_archive(merged, Path(args.output).resolve())
    summary_path = Path(args.output).resolve().with_suffix(Path(args.output).resolve().suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
