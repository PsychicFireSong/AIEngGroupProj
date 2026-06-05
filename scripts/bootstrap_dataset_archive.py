from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path


REQUIRED_DETECTION_CLASSES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]


def run(args: list[str], cwd: Path | None = None) -> None:
    print("Running:", " ".join(map(str, args)))
    subprocess.run(list(map(str, args)), cwd=str(cwd) if cwd else None, check=True)


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination)):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        archive.extractall(destination)


def find_dataset_root(root: Path, dataset_name: str) -> Path:
    candidates = [
        root,
        root / dataset_name,
        root / "merged_dataset",
        root / "dataset_cache" / dataset_name,
    ]
    candidates.extend(path.parent for path in root.rglob("data.yaml"))
    for candidate in candidates:
        if (candidate / "data.yaml").exists() and (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    raise FileNotFoundError(f"Could not find a YOLO dataset root under {root}")


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


def validate_dataset(dataset_root: Path, min_train: int, min_val: int) -> dict:
    counts = dataset_counts(dataset_root)
    missing = [
        name
        for name in REQUIRED_DETECTION_CLASSES
        if counts["train"].get(name, 0) < min_train or counts["val"].get(name, 0) < min_val
    ]
    summary = {
        "dataset_root": str(dataset_root),
        "counts": counts,
        "min_train": min_train,
        "min_val": min_val,
        "ok": not missing,
        "missing_or_low": missing,
    }
    if missing:
        raise RuntimeError(json.dumps(summary, indent=2))
    return summary


def download_drive_folder(folder_id_or_url: str, output_dir: Path) -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    output_dir.mkdir(parents=True, exist_ok=True)
    url = folder_id_or_url
    if "drive.google.com" not in url:
        url = f"https://drive.google.com/drive/folders/{folder_id_or_url}"
    run(["gdown", "--folder", url, "--continue", "-O", str(output_dir)])


def create_archive(dataset_root: Path, dataset_name: str, output_archive: Path) -> None:
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive = output_archive.with_suffix(output_archive.suffix + ".tmp")
    if tmp_archive.exists():
        tmp_archive.unlink()
    started = time.perf_counter()
    with tarfile.open(tmp_archive, "w:gz") as archive:
        archive.add(dataset_root, arcname=dataset_name)
    if output_archive.exists():
        output_archive.unlink()
    tmp_archive.replace(output_archive)
    print(f"Archive created: {output_archive} ({(time.perf_counter() - started):.1f}s)")


def verify_archive(output_archive: Path, dataset_name: str, min_train: int, min_val: int) -> dict:
    verify_dir = output_archive.parent / f"_{dataset_name}_verify_extract"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    verify_dir.mkdir(parents=True, exist_ok=True)
    safe_extract_tar(output_archive, verify_dir)
    dataset_root = find_dataset_root(verify_dir, dataset_name)
    summary = validate_dataset(dataset_root, min_train=min_train, min_val=min_val)
    shutil.rmtree(verify_dir, ignore_errors=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fast Colab recovery archive for a YOLO dataset.")
    parser.add_argument("--dataset-name", default="merged_dataset")
    parser.add_argument("--source-dir", default="", help="Existing local or mounted dataset folder.")
    parser.add_argument("--drive-folder-id", default="", help="Public Google Drive folder ID/URL to download first.")
    parser.add_argument("--work-dir", default="_archive_bootstrap_work")
    parser.add_argument("--output", required=True, help="Output .tar.gz path.")
    parser.add_argument("--min-train", type=int, default=1)
    parser.add_argument("--min-val", type=int, default=1)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    output_archive = Path(args.output).resolve()
    dataset_name = args.dataset_name

    if args.source_dir:
        dataset_root = find_dataset_root(Path(args.source_dir).resolve(), dataset_name)
    elif args.drive_folder_id:
        download_root = work_dir / "download"
        download_drive_folder(args.drive_folder_id, download_root)
        dataset_root = find_dataset_root(download_root, dataset_name)
    else:
        raise SystemExit("Provide --source-dir or --drive-folder-id.")

    summary = validate_dataset(dataset_root, min_train=args.min_train, min_val=args.min_val)
    print("Dataset validation:")
    print(json.dumps(summary, indent=2))
    create_archive(dataset_root, dataset_name, output_archive)
    archive_summary = verify_archive(output_archive, dataset_name, min_train=args.min_train, min_val=args.min_val)
    summary_path = output_archive.with_suffix(output_archive.suffix + ".summary.json")
    summary_path.write_text(json.dumps(archive_summary, indent=2), encoding="utf-8")
    print("Archive validation:")
    print(json.dumps(archive_summary, indent=2))
    print(f"Summary saved: {summary_path}")

    if not args.keep_work and not args.source_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
