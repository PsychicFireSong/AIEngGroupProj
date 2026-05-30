from __future__ import annotations

import json
from pathlib import Path


def source_lines(text: str) -> list[str]:
    text = text.strip() + "\n"
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines(text)}


cells = [
    md(
        """
# Two-Stage YOLO Defect Detection Pipeline

Recoverable Google Colab workflow for the AI Engineering Group Project.

## Cell 1: Environment and Recovery Setup

- Mounts Google Drive and saves logs, weights, sweep reports, and pipeline state there.
- Downloads or restores the seven baseline datasets automatically.
- Builds split-safe Stage 1 detection data and Stage 2 severity crops.
- Reuses the domain sweep we already produced when the CSVs exist.
- Recollects target-domain facility/facade data and retrains only when the decision gate says it is useful.

Before running dataset acquisition in Colab, add these values in the Colab Secrets panel or runtime environment:

- `ROBOFLOW_API_KEY`
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
"""
    ),
    code(
        r'''
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as exc:
    print(f"Google Drive mount skipped or unavailable: {exc}")

GITHUB_REPO_URL = ""          # Optional for fresh Colab: https://github.com/<owner>/AIEngGroupProj.git
REPO_ROOT_OVERRIDE = ""       # Optional for fresh Colab: /content/AIEngGroupProj

def get_secret(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        from google.colab import userdata
        value = userdata.get(name)
        return value or default
    except Exception:
        return default


# Credentials are read from Colab Secrets or environment variables, never committed to Git.
ROBOFLOW_API_KEY = get_secret("ROBOFLOW_API_KEY")
KAGGLE_USERNAME = get_secret("KAGGLE_USERNAME")
KAGGLE_KEY = get_secret("KAGGLE_KEY")

FORCE_DOWNLOAD_DATASETS = False
FORCE_REBUILD_DATASETS = False
FORCE_INITIAL_TRAINING = False
FORCE_SWEEP = False
FORCE_DOMAIN_COLLECTION = False
FORCE_DOMAIN_RETRAIN = False
FORCE_SEVERITY_RETRAIN = False
PROMOTE_DOMAIN_ADAPTED_MODEL = True
CACHE_RAW_DATASETS_TO_DRIVE = True

DRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_OUTPUT_ROOT = DRIVE_ROOT / "AIEngGroupProj_colab_outputs"
DRIVE_RUNS_ROOT = DRIVE_OUTPUT_ROOT / "runs"
DRIVE_WEIGHTS_ROOT = DRIVE_OUTPUT_ROOT / "weights"
LOG_ROOT = DRIVE_OUTPUT_ROOT / "logs"
STATE_PATH = DRIVE_OUTPUT_ROOT / "pipeline_state.json"
for path in (DRIVE_OUTPUT_ROOT, DRIVE_RUNS_ROOT, DRIVE_WEIGHTS_ROOT, LOG_ROOT):
    path.mkdir(parents=True, exist_ok=True)


def now_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def redact_args(args: list[str]) -> list[str]:
    redacted, hide_next = [], False
    for item in map(str, args):
        if hide_next:
            redacted.append("<hidden>")
            hide_next = False
            continue
        redacted.append(item)
        if item in {"--api-key", "--key", "-k"}:
            hide_next = True
    return redacted


def run_process(args: list[str], cwd: Path | None = None, log_name: str = "command", env: dict | None = None) -> subprocess.CompletedProcess:
    cwd = Path(cwd or REPO_ROOT)
    log_path = LOG_ROOT / f"{now_token()}_{log_name}.log"
    printable = " ".join(redact_args(args))
    print(f"Running: {printable}")
    started = time.perf_counter()
    result = subprocess.run(
        list(map(str, args)),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(result.stdout or "", encoding="utf-8")
    tail = (result.stdout or "").strip()[-3000:]
    if tail:
        print(tail)
    print(f"Log saved to: {log_path} ({elapsed:.1f}s)")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {printable}")
    return result


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"steps": {}, "decisions": {}}


STATE = load_state()


def save_state() -> None:
    STATE_PATH.write_text(json.dumps(STATE, indent=2), encoding="utf-8")


def paths_exist(paths) -> bool:
    return all(Path(path).exists() for path in paths)


def step_done(step: str, expected_paths=()) -> bool:
    return bool(STATE.get("steps", {}).get(step)) and paths_exist(expected_paths)


def mark_done(step: str, payload: dict | None = None) -> None:
    STATE.setdefault("steps", {})[step] = {"done_at": now_token(), "payload": payload or {}}
    save_state()


def mirror_path(src: Path, dst: Path) -> None:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        print(f"Nothing to mirror; missing: {src}")
        return
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)


def ensure_repo_root() -> Path:
    if REPO_ROOT_OVERRIDE:
        return Path(REPO_ROOT_OVERRIDE).resolve()
    if Path("/content/AIEngGroupProj").exists():
        return Path("/content/AIEngGroupProj").resolve()
    if Path.cwd().name == "AIEngGroupProj":
        return Path.cwd().resolve()
    if GITHUB_REPO_URL:
        run_process(["git", "clone", GITHUB_REPO_URL, "/content/AIEngGroupProj"], cwd=Path("/content"), log_name="git_clone")
        return Path("/content/AIEngGroupProj").resolve()
    repo_zip = DRIVE_ROOT / "AIEngGroupProj.zip"
    if repo_zip.exists():
        with zipfile.ZipFile(repo_zip, "r") as archive:
            archive.extractall("/content")
        for candidate in [Path("/content/AIEngGroupProj"), *Path("/content").glob("*/AIEngGroupProj")]:
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError("Could not find AIEngGroupProj. Set GITHUB_REPO_URL, REPO_ROOT_OVERRIDE, or upload AIEngGroupProj.zip to Drive.")


REPO_ROOT = ensure_repo_root()
os.chdir(REPO_ROOT)
WEIGHTS_DIR = REPO_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Working directory: {REPO_ROOT}")
print(f"Drive output root: {DRIVE_OUTPUT_ROOT}")

# Colab runtimes are ephemeral, so dependencies are installed each session.
if (REPO_ROOT / "requirements.txt").exists():
    run_process([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], log_name="pip_requirements")
'''
    ),
    md(
        """
## Cell 2: Automated Dataset Acquisition

The notebook restores cached raw folders from Drive first, then downloads missing datasets from Roboflow or Kaggle/KaggleHub. Successful downloads are cached back to Drive so a disconnected Colab runtime can recover without starting from zero.
"""
    ),
    code(
        r'''
DATASET_SOURCES = [
    {"folder": "Finale.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/ponti/finale-3pwus", "workspace": "ponti", "project": "finale-3pwus", "version": 1},
    {"folder": "Internal Wall Defect.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/chew-poh-yee/internal-wall-defect", "workspace": "chew-poh-yee", "project": "internal-wall-defect", "version": 1},
    {"folder": "Concrete defect detection.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/defect-detection-0atjo/concrete-defect-detection-zuym8", "workspace": "defect-detection-0atjo", "project": "concrete-defect-detection-zuym8", "version": 1},
    {"folder": "Corrosion YOLOv8.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/corrosion-yolo-v8/corrosion-yolov8", "workspace": "corrosion-yolo-v8", "project": "corrosion-yolov8", "version": 1},
    {"folder": "metal corrosion.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/yolov11-eorob/metal-corrosion", "workspace": "yolov11-eorob", "project": "metal-corrosion", "version": 1},
    {"folder": "Pothole detection YOLOv8.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/pe1-dtzop/pothole-detection-yolov8-8mspr", "workspace": "pe1-dtzop", "project": "pothole-detection-yolov8-8mspr", "version": 1},
    {"folder": "archive", "provider": "kaggle", "url": "https://www.kaggle.com/datasets/muskanverma24/pothole-detection-dataset-yolov11-optimized", "dataset": "muskanverma24/pothole-detection-dataset-yolov11-optimized"},
]
EXPECTED_DATASETS = [source["folder"] for source in DATASET_SOURCES]
RAW_CACHE_DIR = DRIVE_OUTPUT_ROOT / "raw_dataset_cache"
LOCAL_DOWNLOAD_ROOT = REPO_ROOT / "raw_downloads"
RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
_INSTALLED_PACKAGES = set()


def install_package_once(package: str) -> None:
    if package in _INSTALLED_PACKAGES:
        return
    run_process([sys.executable, "-m", "pip", "install", "-q", package], log_name=f"pip_{package.replace('-', '_')}")
    _INSTALLED_PACKAGES.add(package)


def dataset_ready(path: Path) -> bool:
    path = Path(path)
    return (path / "data.yaml").exists() and any((path / split / "images").exists() for split in ("train", "valid", "val", "test"))


def expected_missing() -> list[str]:
    return [name for name in EXPECTED_DATASETS if not dataset_ready(REPO_ROOT / name)]


def find_dataset_root(search_root: Path) -> Path | None:
    candidates = []
    for data_yaml in Path(search_root).rglob("data.yaml"):
        root = data_yaml.parent
        if any((root / split / "images").exists() for split in ("train", "valid", "val", "test")):
            candidates.append(root)
    candidates.sort(key=lambda path: len(path.parts))
    return candidates[0] if candidates else None


def materialize_dataset(source: dict, downloaded_root: Path) -> bool:
    dataset_root = find_dataset_root(downloaded_root)
    if dataset_root is None:
        return False
    target = REPO_ROOT / source["folder"]
    if target.exists() and FORCE_DOWNLOAD_DATASETS:
        shutil.rmtree(target)
    if not dataset_ready(target):
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(dataset_root, target)
    return dataset_ready(target)


def restore_from_drive_cache() -> None:
    raw_zip = DRIVE_ROOT / "raw_datasets.zip"
    if raw_zip.exists():
        extract_root = LOCAL_DOWNLOAD_ROOT / "raw_zip_extract"
        if FORCE_DOWNLOAD_DATASETS and extract_root.exists():
            shutil.rmtree(extract_root)
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(raw_zip, "r") as archive:
                archive.extractall(extract_root)
        for source in DATASET_SOURCES:
            if not dataset_ready(REPO_ROOT / source["folder"]):
                materialize_dataset(source, extract_root)

    for name in EXPECTED_DATASETS:
        cached = RAW_CACHE_DIR / name
        target = REPO_ROOT / name
        if dataset_ready(cached) and not dataset_ready(target):
            print(f"Restoring cached dataset: {name}")
            shutil.copytree(cached, target, dirs_exist_ok=True)


def cache_dataset_to_drive(name: str) -> None:
    if not CACHE_RAW_DATASETS_TO_DRIVE:
        return
    source = REPO_ROOT / name
    target = RAW_CACHE_DIR / name
    if dataset_ready(source) and (FORCE_DOWNLOAD_DATASETS or not dataset_ready(target)):
        print(f"Caching dataset to Drive: {name}")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def download_roboflow_source(source: dict, rf) -> bool:
    if dataset_ready(REPO_ROOT / source["folder"]) and not FORCE_DOWNLOAD_DATASETS:
        return True
    destination = LOCAL_DOWNLOAD_ROOT / source["folder"]
    if destination.exists() and FORCE_DOWNLOAD_DATASETS:
        shutil.rmtree(destination)
    if dataset_ready(destination):
        return materialize_dataset(source, destination)
    destination.mkdir(parents=True, exist_ok=True)
    version = rf.workspace(source["workspace"]).project(source["project"]).version(int(source["version"]))
    last_error = None
    for export_format in ("yolov11", "yolov8"):
        try:
            print(f"Downloading {source['folder']} from Roboflow as {export_format} ...")
            version.download(export_format, location=str(destination), overwrite=True)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"  {export_format} failed: {exc}")
    if last_error is not None:
        raise last_error
    return materialize_dataset(source, destination)


def download_kaggle_source(source: dict) -> bool:
    if dataset_ready(REPO_ROOT / source["folder"]) and not FORCE_DOWNLOAD_DATASETS:
        return True
    destination = LOCAL_DOWNLOAD_ROOT / source["folder"]
    if destination.exists() and FORCE_DOWNLOAD_DATASETS:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        install_package_once("kagglehub")
        import kagglehub
        if materialize_dataset(source, Path(kagglehub.dataset_download(source["dataset"]))):
            return True
    except Exception as exc:
        print(f"KaggleHub download failed for {source['folder']}: {exc}")

    if KAGGLE_USERNAME and KAGGLE_KEY:
        install_package_once("kaggle")
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}), encoding="utf-8")
        kaggle_json.chmod(0o600)
        run_process(["kaggle", "datasets", "download", "-d", source["dataset"], "-p", str(destination), "--unzip"], log_name="kaggle_download")
        return materialize_dataset(source, destination)
    print("Kaggle CLI fallback skipped because KAGGLE_USERNAME is not set.")
    return False


restore_from_drive_cache()
rf = None
for source in DATASET_SOURCES:
    if dataset_ready(REPO_ROOT / source["folder"]) and not FORCE_DOWNLOAD_DATASETS:
        print(f"Ready: {source['folder']}")
        continue
    try:
        if source["provider"] == "roboflow":
            if not ROBOFLOW_API_KEY:
                raise ValueError("ROBOFLOW_API_KEY is empty.")
            if rf is None:
                install_package_once("roboflow")
                from roboflow import Roboflow
                rf = Roboflow(api_key=ROBOFLOW_API_KEY)
            ok = download_roboflow_source(source, rf)
        else:
            ok = download_kaggle_source(source)
        print(f"{'Ready' if ok else 'Missing'}: {source['folder']}")
    except Exception as exc:
        print(f"Download failed for {source['folder']}: {exc}")

for name in EXPECTED_DATASETS:
    cache_dataset_to_drive(name)

missing = expected_missing()
if missing:
    raise RuntimeError(f"Automated dataset acquisition did not complete these folders: {missing}. Check Drive logs and credentials.")
mark_done("dataset_acquisition", {"datasets": EXPECTED_DATASETS})
print("All baseline datasets are ready.")
'''
    ),
    md(
        """
## Cell 3: Verify Project Files

The required scripts live in the repository as normal Python files. This cell checks that they exist and compiles them before execution.
"""
    ),
    code(
        r'''
import py_compile

REQUIRED_PROJECT_FILES = [
    "configs/merge_config.yaml",
    "configs/domain_sweep_manifest.csv",
    "scripts/merge_datasets.py",
    "scripts/extract_severity_crops.py",
    "scripts/domain_sweep.py",
    "scripts/auto_collect_domain_sources.py",
    "scripts/build_domain_adaptation_dataset.py",
    "scripts/build_severity_adaptation_dataset.py",
    "apps/inference_api.py",
]
missing_files = [path for path in REQUIRED_PROJECT_FILES if not (REPO_ROOT / path).exists()]
if missing_files:
    raise FileNotFoundError(f"Missing required project files: {missing_files}")
for relative_path in REQUIRED_PROJECT_FILES:
    if relative_path.endswith(".py"):
        py_compile.compile(str(REPO_ROOT / relative_path), doraise=True)
print("Project files are present and Python scripts compiled successfully.")
mark_done("verify_project_files", {"files": REQUIRED_PROJECT_FILES})
'''
    ),
    md(
        """
## Cell 4: Merge Detection Dataset and Extract Severity Crops

`merge_datasets.py` is called with `--preserve-splits`, preventing global pooling/shuffling leakage. Severity crops are extracted from the original datasets before merging, using normalized YOLO boxes and OpenCV crops.
"""
    ),
    code(
        r'''
MERGED_DATASET = REPO_ROOT / "merged_dataset"
SEVERITY_DATASET = REPO_ROOT / "severity_dataset"

if FORCE_REBUILD_DATASETS or not step_done("merge_detection_dataset", [MERGED_DATASET / "data.yaml"]):
    run_process([sys.executable, "scripts/merge_datasets.py", "--config", "configs/merge_config.yaml", "--preserve-splits", "--force"], log_name="merge_detection_dataset")
    mark_done("merge_detection_dataset", {"data_yaml": str(MERGED_DATASET / "data.yaml")})
else:
    print("Merged detection dataset already exists; skipping rebuild.")

if FORCE_REBUILD_DATASETS or not step_done("extract_severity_crops", [SEVERITY_DATASET / "data.yaml"]):
    run_process([sys.executable, "scripts/extract_severity_crops.py", "--config", "configs/merge_config.yaml", "--output", str(SEVERITY_DATASET), "--force"], log_name="extract_severity_crops")
    mark_done("extract_severity_crops", {"data_yaml": str(SEVERITY_DATASET / "data.yaml")})
else:
    print("Severity crop dataset already exists; skipping extraction.")

for summary in [MERGED_DATASET / "merge_summary.json", SEVERITY_DATASET / "severity_summary.json"]:
    if summary.exists():
        mirror_path(summary, DRIVE_OUTPUT_ROOT / "summaries" / summary.name)
        print(summary.read_text(encoding="utf-8")[:2000])
'''
    ),
    md(
        """
## Cell 5: Initial Model Training

Stage 1 uses `yolo11s.pt` for 100 epochs at 640 resolution. Stage 2 uses `yolo11n-cls.pt` for 50 epochs at 224 resolution. Training runs and final weights are saved to Drive; interrupted runs resume from `last.pt` when possible.
"""
    ),
    code(
        r'''
from ultralytics import YOLO
try:
    import torch
except Exception:
    torch = None


def train_device_kwargs() -> dict:
    return {"device": 0} if torch is not None and torch.cuda.is_available() else {}


def copy_weight_to_local(drive_weight: Path, local_name: str) -> None:
    if drive_weight.exists():
        target = WEIGHTS_DIR / local_name
        shutil.copy2(drive_weight, target)
        print(f"Local model ready: {target}")


def train_or_resume(seed_weights: str | Path, data_path: Path, run_name: str, final_weight: Path, train_args: dict, force: bool = False) -> Path:
    run_dir = DRIVE_RUNS_ROOT / run_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if final_weight.exists() and not force:
        print(f"Using existing trained weight: {final_weight}")
        return final_weight
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if best.exists() and not force:
        print(f"Recovered completed training run from: {best}")
        final_weight.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, final_weight)
        return final_weight
    if last.exists() and not force and not best.exists():
        print(f"Resuming interrupted training from: {last}")
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(str(seed_weights)).train(
            data=str(data_path),
            project=str(DRIVE_RUNS_ROOT),
            name=run_name,
            exist_ok=True,
            plots=True,
            val=True,
            **train_device_kwargs(),
            **train_args,
        )
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best}")
    final_weight.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, final_weight)
    return final_weight


DETECTOR_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector.pt"
SEVERITY_WEIGHT = DRIVE_WEIGHTS_ROOT / "severity_cls.pt"
detector_args = {
    "epochs": 100, "imgsz": 640, "patience": 20, "optimizer": "AdamW", "cos_lr": True,
    "close_mosaic": 10, "mosaic": 0.60, "mixup": 0.05, "copy_paste": 0.10,
    "degrees": 3.0, "translate": 0.08, "scale": 0.35, "fliplr": 0.50,
    "hsv_s": 0.45, "hsv_v": 0.35,
}
severity_args = {"epochs": 50, "imgsz": 224, "patience": 12, "optimizer": "AdamW", "cos_lr": True, "dropout": 0.15}

trained_detector = train_or_resume("yolo11s.pt", MERGED_DATASET / "data.yaml", "stage1_defect_detector", DETECTOR_WEIGHT, detector_args, FORCE_INITIAL_TRAINING)
trained_severity = train_or_resume("yolo11n-cls.pt", SEVERITY_DATASET, "stage2_severity", SEVERITY_WEIGHT, severity_args, FORCE_INITIAL_TRAINING)
copy_weight_to_local(trained_detector, "defect_detector.pt")
copy_weight_to_local(trained_severity, "severity_cls.pt")
mark_done("initial_training", {"detector": str(trained_detector), "severity": str(trained_severity)})
'''
    ),
    md(
        """
## Cell 6: Reuse or Run the Domain Sweep

This cell reuses `output/domain_sweep/domain_sweep_summary.csv` if it exists locally or in Drive. It only runs the sweep again when cached results are missing or `FORCE_SWEEP = True`.
"""
    ),
    code(
        r'''
BASELINE_SWEEP_DIR = REPO_ROOT / "output" / "domain_sweep"
DRIVE_BASELINE_SWEEP_DIR = DRIVE_OUTPUT_ROOT / "domain_sweep" / "baseline"


def sweep_ready(path: Path) -> bool:
    return (path / "domain_sweep_summary.csv").exists() and (path / "summary.json").exists()


def reuse_or_run_sweep(output_dir: Path, drive_dir: Path, detector_weight: Path, name: str, force: bool = False) -> Path:
    if sweep_ready(output_dir) and not force:
        print(f"Reusing local sweep: {output_dir}")
        mirror_path(output_dir, drive_dir)
        return output_dir
    if sweep_ready(drive_dir) and not force:
        print(f"Restoring sweep from Drive: {drive_dir}")
        mirror_path(drive_dir, output_dir)
        return output_dir
    run_process([
        sys.executable, "scripts/domain_sweep.py",
        "--manifest", "configs/domain_sweep_manifest.csv",
        "--detector", str(detector_weight),
        "--severity", str(SEVERITY_WEIGHT),
        "--output", str(output_dir),
        "--thresholds", "0.45,0.30,0.20,0.10",
        "--iou", "0.45",
        "--annotate-conf", "0.20",
    ], log_name=name)
    mirror_path(output_dir, drive_dir)
    return output_dir


baseline_sweep = reuse_or_run_sweep(BASELINE_SWEEP_DIR, DRIVE_BASELINE_SWEEP_DIR, DETECTOR_WEIGHT, "baseline_domain_sweep", FORCE_SWEEP)
print((baseline_sweep / "summary.json").read_text(encoding="utf-8"))
mark_done("baseline_domain_sweep", {"path": str(baseline_sweep)})
'''
    ),
    md(
        """
## Cell 7: Automated Target-Domain Recollection

The weak-domain sweep showed mismatch on exterior facades, paint degradation, wall cracks, spalling, and corrosion. This cell recollects extra annotated facility/facade datasets from Roboflow Universe and remaps them into the five detection classes.
"""
    ),
    code(
        r'''
TARGET_YOLO = REPO_ROOT / "domain_adaptation" / "target_yolo"
AUTO_RAW = REPO_ROOT / "domain_adaptation" / "auto_raw"
TARGET_SUMMARY = TARGET_YOLO / "auto_collection_summary.json"


def count_images(root: Path) -> int:
    return sum(1 for path in Path(root).glob("**/*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}) if Path(root).exists() else 0


if FORCE_DOMAIN_COLLECTION or not TARGET_SUMMARY.exists() or count_images(TARGET_YOLO) == 0:
    run_process([
        sys.executable, "scripts/auto_collect_domain_sources.py",
        "--api-key", ROBOFLOW_API_KEY,
        "--raw-output", str(AUTO_RAW),
        "--target-output", str(TARGET_YOLO),
    ] + (["--force-download", "--force-rebuild"] if FORCE_DOMAIN_COLLECTION else []), log_name="auto_collect_domain_sources")
else:
    print(f"Using existing target-domain collection: {TARGET_YOLO}")

if TARGET_SUMMARY.exists():
    mirror_path(TARGET_SUMMARY, DRIVE_OUTPUT_ROOT / "summaries" / "auto_collection_summary.json")
    print(TARGET_SUMMARY.read_text(encoding="utf-8")[:3000])
mark_done("auto_target_domain_collection", {"target_yolo": str(TARGET_YOLO), "images": count_images(TARGET_YOLO)})
'''
    ),
    md(
        """
## Cell 8: Decide Whether Retraining Is Needed

Domain retraining is triggered only when the sweep has failed rows and the recollection produced usable target-domain labels. Use `FORCE_DOMAIN_RETRAIN = True` to override.
"""
    ),
    code(
        r'''
def load_sweep_rows(summary_csv: Path) -> list[dict]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def failure_count(rows: list[dict], thresholds=("0.45", "0.3", "0.30", "0.2", "0.20")) -> int:
    return sum(1 for row in rows if str(row.get("threshold", "")) in thresholds and row.get("match") == "false")


def target_label_count(target_root: Path) -> int:
    total = 0
    for label_path in target_root.glob("**/*.txt"):
        total += len([line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    return total


rows = load_sweep_rows(BASELINE_SWEEP_DIR / "domain_sweep_summary.csv")
weak_rows = failure_count(rows)
target_images = count_images(TARGET_YOLO)
target_labels = target_label_count(TARGET_YOLO)
NEEDS_DOMAIN_RETRAIN = FORCE_DOMAIN_RETRAIN or (weak_rows > 0 and target_images >= 10 and target_labels >= 10)
STATE.setdefault("decisions", {})["needs_domain_retrain"] = NEEDS_DOMAIN_RETRAIN
STATE["decisions"]["domain_retrain_reason"] = {
    "weak_sweep_rows": weak_rows,
    "target_images": target_images,
    "target_labels": target_labels,
    "force": FORCE_DOMAIN_RETRAIN,
}
save_state()
print(json.dumps(STATE["decisions"], indent=2))
'''
    ),
    md(
        """
## Cell 9: Build the Domain-Adapted Detection Dataset

If retraining is needed, the notebook combines the original merged data, recollected target-domain labels, and optional hard negatives into `merged_dataset_domain_adapted`.
"""
    ),
    code(
        r'''
DOMAIN_ADAPTED_DATASET = REPO_ROOT / "merged_dataset_domain_adapted"
HARD_NEGATIVES = REPO_ROOT / "domain_adaptation" / "hard_negatives"


def create_hard_negatives_from_sweep(resolved_manifest: Path, output_root: Path) -> int:
    if not resolved_manifest.exists():
        return 0
    output_dir = output_root / "valid" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    with resolved_manifest.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("expected_class") != "none":
                continue
            image_path = Path(row.get("resolved_path", ""))
            if image_path.exists():
                shutil.copy2(image_path, output_dir / image_path.name)
                copied += 1
    return copied


if NEEDS_DOMAIN_RETRAIN:
    copied_negatives = create_hard_negatives_from_sweep(BASELINE_SWEEP_DIR / "resolved_manifest.csv", HARD_NEGATIVES)
    print(f"Hard-negative sweep images copied: {copied_negatives}")
    if FORCE_REBUILD_DATASETS or not (DOMAIN_ADAPTED_DATASET / "data.yaml").exists():
        run_process([
            sys.executable, "scripts/build_domain_adaptation_dataset.py",
            "--base", str(MERGED_DATASET),
            "--target", str(TARGET_YOLO),
            "--hard-negatives", str(HARD_NEGATIVES),
            "--output", str(DOMAIN_ADAPTED_DATASET),
            "--target-repeat", "3",
            "--negative-repeat", "2",
        ], log_name="build_domain_adapted_dataset")
    if (DOMAIN_ADAPTED_DATASET / "domain_adaptation_summary.json").exists():
        mirror_path(DOMAIN_ADAPTED_DATASET / "domain_adaptation_summary.json", DRIVE_OUTPUT_ROOT / "summaries" / "domain_adaptation_summary.json")
        print((DOMAIN_ADAPTED_DATASET / "domain_adaptation_summary.json").read_text(encoding="utf-8")[:3000])
else:
    print("Domain retraining is not needed by the current decision gate; dataset build skipped.")
mark_done("build_domain_adapted_dataset", {"needed": NEEDS_DOMAIN_RETRAIN, "output": str(DOMAIN_ADAPTED_DATASET)})
'''
    ),
    md(
        """
## Cell 10: Fine-Tune Stage 1 on Target-Domain Data

This is the core low-confidence/misclassification fix: the detector sees more facility/facade examples, target data is oversampled, and hard negatives reduce false positives. The adapted model can be promoted to the dashboard default.
"""
    ),
    code(
        r'''
DOMAIN_DETECTOR_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector_domain_adapted.pt"

if NEEDS_DOMAIN_RETRAIN:
    baseline_backup = DRIVE_WEIGHTS_ROOT / "defect_detector_baseline.pt"
    if DETECTOR_WEIGHT.exists() and not baseline_backup.exists():
        shutil.copy2(DETECTOR_WEIGHT, baseline_backup)
    domain_args = {
        "epochs": 60, "imgsz": 640, "patience": 15, "optimizer": "AdamW", "lr0": 0.001,
        "cos_lr": True, "close_mosaic": 8, "mosaic": 0.45, "mixup": 0.03,
        "copy_paste": 0.05, "degrees": 2.0, "translate": 0.06, "scale": 0.25,
        "fliplr": 0.50, "hsv_s": 0.35, "hsv_v": 0.30,
    }
    trained_domain_detector = train_or_resume(
        DETECTOR_WEIGHT if DETECTOR_WEIGHT.exists() else "yolo11s.pt",
        DOMAIN_ADAPTED_DATASET / "data.yaml",
        "stage1_domain_adapted_detector",
        DOMAIN_DETECTOR_WEIGHT,
        domain_args,
        FORCE_DOMAIN_RETRAIN,
    )
    copy_weight_to_local(trained_domain_detector, "defect_detector_domain_adapted.pt")
    if PROMOTE_DOMAIN_ADAPTED_MODEL:
        shutil.copy2(trained_domain_detector, DETECTOR_WEIGHT)
        copy_weight_to_local(DETECTOR_WEIGHT, "defect_detector.pt")
        print("Domain-adapted detector promoted as the default deployment detector.")
    mark_done("domain_detector_training", {"detector": str(trained_domain_detector), "promoted": PROMOTE_DOMAIN_ADAPTED_MODEL})
else:
    print("Domain detector fine-tuning skipped because retraining was not needed.")
'''
    ),
    md(
        """
## Cell 11: Optional Automated Stage 2 Severity Adaptation

When raw target-domain downloads include detailed labels, this cell extracts extra severity crops and retrains the classifier only if enough extra crops exist.
"""
    ),
    code(
        r'''
import yaml

SEVERITY_EXTRA = REPO_ROOT / "domain_adaptation" / "severity_extra"
SEVERITY_ADAPTED_DATASET = REPO_ROOT / "severity_dataset_domain_adapted"
DOMAIN_SEVERITY_WEIGHT = DRIVE_WEIGHTS_ROOT / "severity_cls_domain_adapted.pt"


def write_auto_severity_config(raw_root: Path) -> Path | None:
    datasets = []
    for data_yaml in sorted(raw_root.glob("*/**/data.yaml")):
        root = data_yaml.parent
        if any((root / split / "images").exists() for split in ("train", "valid", "val", "test")):
            datasets.append({"name": root.parent.name if root.name == "data" else root.name, "path": str(root)})
    if not datasets:
        return None
    config_path = REPO_ROOT / "domain_adaptation" / "auto_severity_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump({"datasets": datasets}, sort_keys=False), encoding="utf-8")
    return config_path


severity_config = write_auto_severity_config(AUTO_RAW)
if severity_config is not None and (FORCE_REBUILD_DATASETS or not (SEVERITY_EXTRA / "data.yaml").exists()):
    run_process([sys.executable, "scripts/extract_severity_crops.py", "--config", str(severity_config), "--output", str(SEVERITY_EXTRA), "--force"], log_name="extract_target_severity_crops")
elif severity_config is None:
    print("No raw target-domain folders were available for automatic severity crop extraction.")

extra_crops = count_images(SEVERITY_EXTRA)
NEEDS_SEVERITY_RETRAIN = FORCE_SEVERITY_RETRAIN or (extra_crops >= 10 and NEEDS_DOMAIN_RETRAIN)
STATE.setdefault("decisions", {})["needs_severity_retrain"] = NEEDS_SEVERITY_RETRAIN
STATE["decisions"]["severity_retrain_reason"] = {"extra_crops": extra_crops, "force": FORCE_SEVERITY_RETRAIN}
save_state()

if NEEDS_SEVERITY_RETRAIN:
    if FORCE_REBUILD_DATASETS or not (SEVERITY_ADAPTED_DATASET / "severity_adaptation_summary.json").exists():
        run_process([
            sys.executable, "scripts/build_severity_adaptation_dataset.py",
            "--base", str(SEVERITY_DATASET),
            "--target", str(SEVERITY_EXTRA),
            "--output", str(SEVERITY_ADAPTED_DATASET),
            "--target-repeat", "3",
        ], log_name="build_severity_adapted_dataset")
    severity_adapt_args = {"epochs": 40, "imgsz": 224, "patience": 10, "optimizer": "AdamW", "lr0": 0.001, "cos_lr": True, "dropout": 0.20}
    trained_domain_severity = train_or_resume(
        SEVERITY_WEIGHT if SEVERITY_WEIGHT.exists() else "yolo11n-cls.pt",
        SEVERITY_ADAPTED_DATASET,
        "stage2_domain_adapted_severity",
        DOMAIN_SEVERITY_WEIGHT,
        severity_adapt_args,
        FORCE_SEVERITY_RETRAIN,
    )
    shutil.copy2(trained_domain_severity, SEVERITY_WEIGHT)
    copy_weight_to_local(SEVERITY_WEIGHT, "severity_cls.pt")
    mark_done("domain_severity_training", {"severity": str(trained_domain_severity)})
else:
    print("Severity retraining skipped. Extra target-domain severity crops are not sufficient yet.")
print(json.dumps(STATE["decisions"], indent=2))
'''
    ),
    md(
        """
## Cell 12: Post-Adaptation Acceptance Sweep

After retraining, the same sweep is rerun with the adapted detector and a before/after summary is saved to Drive.
"""
    ),
    code(
        r'''
POST_SWEEP_DIR = REPO_ROOT / "output" / "domain_sweep_domain_adapted"
DRIVE_POST_SWEEP_DIR = DRIVE_OUTPUT_ROOT / "domain_sweep" / "domain_adapted"


def summarize_matches(summary_csv: Path, threshold: str = "0.2") -> dict:
    rows = load_sweep_rows(summary_csv)
    selected = [row for row in rows if str(row.get("threshold")) in {threshold, f"{float(threshold):.1f}"}]
    return {
        "rows": len(selected),
        "matches": sum(1 for row in selected if row.get("match") == "true"),
        "failures": sum(1 for row in selected if row.get("match") == "false"),
        "skipped": sum(1 for row in selected if row.get("match") == "skipped"),
        "no_detection": sum(1 for row in selected if row.get("detections") == "0"),
    }


if NEEDS_DOMAIN_RETRAIN and DOMAIN_DETECTOR_WEIGHT.exists():
    post_sweep = reuse_or_run_sweep(POST_SWEEP_DIR, DRIVE_POST_SWEEP_DIR, DOMAIN_DETECTOR_WEIGHT, "domain_adapted_sweep", FORCE_SWEEP)
    comparison = {
        "baseline_at_conf_0_20": summarize_matches(BASELINE_SWEEP_DIR / "domain_sweep_summary.csv", "0.2"),
        "domain_adapted_at_conf_0_20": summarize_matches(post_sweep / "domain_sweep_summary.csv", "0.2"),
        "baseline_summary": str(BASELINE_SWEEP_DIR / "summary.json"),
        "adapted_summary": str(post_sweep / "summary.json"),
    }
    comparison_path = DRIVE_OUTPUT_ROOT / "summaries" / "domain_sweep_before_after.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    mark_done("post_domain_sweep", comparison)
else:
    print("Post-adaptation sweep skipped because no domain-adapted detector was trained in this run.")
    print(json.dumps({"baseline_at_conf_0_20": summarize_matches(BASELINE_SWEEP_DIR / "domain_sweep_summary.csv", "0.2")}, indent=2))
'''
    ),
    md(
        """
## Cell 13: Deployment Artifacts

The dashboard/backend discover `.pt` files from `weights/` and Drive. This cell makes sure the canonical local names are present for deployment.
"""
    ),
    code(
        r'''
for drive_weight, local_name in [
    (DETECTOR_WEIGHT, "defect_detector.pt"),
    (SEVERITY_WEIGHT, "severity_cls.pt"),
    (DOMAIN_DETECTOR_WEIGHT, "defect_detector_domain_adapted.pt"),
    (DOMAIN_SEVERITY_WEIGHT, "severity_cls_domain_adapted.pt"),
]:
    if drive_weight.exists():
        copy_weight_to_local(drive_weight, local_name)

print("Available local deployment weights:")
for model_path in sorted(WEIGHTS_DIR.glob("*.pt")):
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"- {model_path}: {size_mb:.1f} MB")
print(f"Drive weights folder: {DRIVE_WEIGHTS_ROOT}")
print(f"Pipeline state: {STATE_PATH}")
'''
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / "two_stage_yolo_defect_pipeline_colab.ipynb"
out.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {out} with {len(cells)} cells")
print(f"Markdown cells: {sum(1 for cell in cells if cell['cell_type'] == 'markdown')}")
print(f"Code cells: {sum(1 for cell in cells if cell['cell_type'] == 'code')}")
