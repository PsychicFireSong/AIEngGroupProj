from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

MATERIALIZED_FILES = [
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


def source_lines(text: str) -> list[str]:
    text = text.strip() + "\n"
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines(text)}


def materialize_project_cell() -> str:
    contents = {path: (REPO_ROOT / path).read_text(encoding="utf-8") for path in MATERIALIZED_FILES}
    return (
        "PROJECT_FILE_CONTENTS = "
        + repr(contents)
        + r'''

import py_compile


def write_text_if_changed(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


changed = []
for relative_path, content in PROJECT_FILE_CONTENTS.items():
    if write_text_if_changed(REPO_ROOT / relative_path, content):
        changed.append(relative_path)

if changed:
    print("Notebook materialized these project files:")
    for item in changed:
        print(f"- {item}")
else:
    print("Project files already match the notebook copy.")

for relative_path in PROJECT_FILE_CONTENTS:
    if relative_path.endswith(".py"):
        py_compile.compile(str(REPO_ROOT / relative_path), doraise=True)

print("All materialized Python files compiled successfully.")
mark_done("materialize_project_files", {"files": sorted(PROJECT_FILE_CONTENTS)})
'''
    )


cells = [
    md(
        """
# Two-Stage YOLO Defect Detection Pipeline

## Cell 1: Environment, Drive, and Recovery Setup

This is the compact 8-section Colab notebook. It keeps markdown and code separated, but each numbered section maps to one pipeline stage.

Recovery behavior:

- Uses Google Drive for logs, weights, runs, cached datasets, sweep reports, and `pipeline_state.json`.
- Recovers completed training from existing `best.pt` before trying to retrain.
- Resumes interrupted training from `last.pt` only when training data is available.
- Can run inference/sweeps from recovered weights even when raw baseline datasets are not needed in that session.

Before running dataset download cells in Colab, add these in Colab Secrets or environment variables:

- `ROBOFLOW_API_KEY`
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
"""
    ),
    code(
        r'''
from __future__ import annotations

import csv
import hashlib
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

GITHUB_REPO_URL = ""     # Optional fresh Colab clone URL.
REPO_ROOT_OVERRIDE = ""  # Optional existing path, for example: /content/AIEngGroupProj

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


ROBOFLOW_API_KEY = get_secret("ROBOFLOW_API_KEY")
KAGGLE_USERNAME = get_secret("KAGGLE_USERNAME")
KAGGLE_KEY = get_secret("KAGGLE_KEY")


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
    cwd = Path(cwd or globals().get("REPO_ROOT", Path.cwd()))
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

DETECTOR_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector.pt"
SEVERITY_WEIGHT = DRIVE_WEIGHTS_ROOT / "severity_cls.pt"
DOMAIN_DETECTOR_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector_domain_adapted.pt"
DOMAIN_SEVERITY_WEIGHT = DRIVE_WEIGHTS_ROOT / "severity_cls_domain_adapted.pt"


def candidate_run_roots() -> list[Path]:
    roots = [DRIVE_RUNS_ROOT, DRIVE_OUTPUT_ROOT / "runs", DRIVE_ROOT / "runs"]
    return list(dict.fromkeys(path for path in roots if path.exists()))


def checkpoint_candidates(kind: str, checkpoint_name: str = "best.pt") -> list[Path]:
    final_candidates = []
    if kind == "detector":
        final_candidates = [DETECTOR_WEIGHT, DOMAIN_DETECTOR_WEIGHT, WEIGHTS_DIR / "defect_detector.pt", WEIGHTS_DIR / "defect_detector_domain_adapted.pt"]
        include_terms, exclude_terms = ("stage1", "defect", "detector", "detect"), ("severity", "classify", "cls", "stage2")
    else:
        final_candidates = [SEVERITY_WEIGHT, DOMAIN_SEVERITY_WEIGHT, WEIGHTS_DIR / "severity_cls.pt", WEIGHTS_DIR / "severity_cls_domain_adapted.pt"]
        include_terms, exclude_terms = ("severity", "classify", "cls", "stage2"), ("detector", "detect", "stage1")

    candidates = [path for path in final_candidates if path.exists()]
    for root in candidate_run_roots():
        for path in root.glob(f"**/weights/{checkpoint_name}"):
            lowered = str(path).replace("\\", "/").lower()
            if any(term in lowered for term in include_terms) and not any(term in lowered for term in exclude_terms):
                candidates.append(path)
    candidates = [path.resolve() for path in candidates if path.exists()]
    return sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)


def previous_training_available() -> bool:
    return bool(checkpoint_candidates("detector", "best.pt") and checkpoint_candidates("severity", "best.pt"))


def copy_weight_to_local(drive_weight: Path, local_name: str) -> None:
    if Path(drive_weight).exists():
        target = WEIGHTS_DIR / local_name
        shutil.copy2(drive_weight, target)
        print(f"Local model ready: {target}")


print(f"Working directory: {REPO_ROOT}")
print(f"Drive output root: {DRIVE_OUTPUT_ROOT}")
if (REPO_ROOT / "requirements.txt").exists():
    run_process([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], log_name="pip_requirements")
'''
    ),
    md(
        """
## Cell 2: Automated Baseline Dataset Acquisition

This cell tries Drive cache, `raw_datasets.zip`, Roboflow, and Kaggle/KaggleHub. If the seven raw datasets are missing but completed model weights already exist in Drive, it continues instead of blocking recovery.
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
    source, target = REPO_ROOT / name, RAW_CACHE_DIR / name
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
DATASETS_AVAILABLE = not missing
STATE.setdefault("decisions", {})["baseline_datasets_available"] = DATASETS_AVAILABLE
STATE["decisions"]["missing_baseline_datasets"] = missing
save_state()

if missing and previous_training_available():
    print("Baseline datasets are incomplete, but previous Drive training artifacts are available. Recovery/training cells will reuse those weights.")
elif missing:
    raise RuntimeError(f"Missing baseline datasets and no completed training artifacts were found: {missing}")
else:
    print("All baseline datasets are ready.")
    mark_done("dataset_acquisition", {"datasets": EXPECTED_DATASETS})
'''
    ),
    md(
        """
## Cell 3: Self-Materialize Configs and Scripts

The notebook writes the required repository files itself. It does not depend on the clone already having the latest scripts, so Colab can recover from an older clone or a zip extraction.
"""
    ),
    code(materialize_project_cell()),
    md(
        """
## Cell 4: Build Baseline Datasets When Needed

This preserves source train/valid/test splits and extracts severity crops. If datasets are missing but completed baseline weights already exist, this cell skips dataset building so the notebook can recover previous training instead of blocking.
"""
    ),
    code(
        r'''
MERGED_DATASET = REPO_ROOT / "merged_dataset"
SEVERITY_DATASET = REPO_ROOT / "severity_dataset"

if not DATASETS_AVAILABLE and previous_training_available():
    print("Skipping baseline dataset build because previous trained weights are recoverable and raw datasets are incomplete.")
elif not DATASETS_AVAILABLE:
    raise RuntimeError("Cannot build baseline datasets because raw baseline datasets are incomplete.")
else:
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
## Cell 5: Recover or Train Baseline Models

This is the important recovery stage. The notebook first checks canonical Drive weights, then searches Drive run folders for completed `best.pt`, then resumes `last.pt` if data exists. It trains only when no previous usable artifact exists.
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


def recover_completed_weight(kind: str, final_weight: Path, local_name: str) -> Path | None:
    if final_weight.exists() and not FORCE_INITIAL_TRAINING:
        print(f"Using existing canonical {kind} weight: {final_weight}")
        copy_weight_to_local(final_weight, local_name)
        return final_weight
    candidates = checkpoint_candidates(kind, "best.pt")
    if candidates and not FORCE_INITIAL_TRAINING:
        best = candidates[0]
        print(f"Recovered completed {kind} training from: {best}")
        final_weight.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, final_weight)
        copy_weight_to_local(final_weight, local_name)
        return final_weight
    return None


def latest_last_checkpoint(kind: str) -> Path | None:
    candidates = checkpoint_candidates(kind, "last.pt")
    return candidates[0] if candidates else None


def train_or_resume(kind: str, seed_weights: str | Path, data_path: Path, run_name: str, final_weight: Path, local_name: str, train_args: dict, force: bool = False) -> Path:
    recovered = recover_completed_weight(kind, final_weight, local_name)
    if recovered and not force:
        return recovered

    if not Path(data_path).exists():
        last = latest_last_checkpoint(kind)
        if last:
            raise RuntimeError(f"Found interrupted {kind} checkpoint at {last}, but training data is missing, so resume is not safe.")
        raise RuntimeError(f"No recoverable {kind} weight was found and training data is missing: {data_path}")

    run_dir = DRIVE_RUNS_ROOT / run_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if best.exists() and not force:
        print(f"Recovered completed {kind} run from: {best}")
        final_weight.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, final_weight)
        copy_weight_to_local(final_weight, local_name)
        return final_weight
    if last.exists() and not force:
        print(f"Resuming interrupted {kind} training from: {last}")
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
    copy_weight_to_local(final_weight, local_name)
    return final_weight


detector_args = {
    "epochs": 100, "imgsz": 640, "patience": 20, "optimizer": "AdamW", "cos_lr": True,
    "close_mosaic": 10, "mosaic": 0.60, "mixup": 0.05, "copy_paste": 0.10,
    "degrees": 3.0, "translate": 0.08, "scale": 0.35, "fliplr": 0.50,
    "hsv_s": 0.45, "hsv_v": 0.35,
}
severity_args = {"epochs": 50, "imgsz": 224, "patience": 12, "optimizer": "AdamW", "cos_lr": True, "dropout": 0.15}

trained_detector = train_or_resume("detector", "yolo11s.pt", MERGED_DATASET / "data.yaml", "stage1_defect_detector", DETECTOR_WEIGHT, "defect_detector.pt", detector_args, FORCE_INITIAL_TRAINING)
trained_severity = train_or_resume("severity", "yolo11n-cls.pt", SEVERITY_DATASET, "stage2_severity", SEVERITY_WEIGHT, "severity_cls.pt", severity_args, FORCE_INITIAL_TRAINING)
mark_done("baseline_model_recovery_or_training", {"detector": str(trained_detector), "severity": str(trained_severity)})
'''
    ),
    md(
        """
## Cell 6: Domain Sweep With Explicit Image Sources

This cell prints the exact sweep images from `configs/domain_sweep_manifest.csv`. It reuses cached sweep results only when the manifest fingerprint matches, so old local-path sweeps are not silently reused.
"""
    ),
    code(
        r'''
BASELINE_SWEEP_DIR = REPO_ROOT / "output" / "domain_sweep"
DRIVE_BASELINE_SWEEP_DIR = DRIVE_OUTPUT_ROOT / "domain_sweep" / "baseline"
MANIFEST_PATH = REPO_ROOT / "configs" / "domain_sweep_manifest.csv"


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sweep_fingerprint_path(path: Path) -> Path:
    return Path(path) / "manifest_sha256.txt"


def sweep_ready(path: Path, manifest_path: Path) -> bool:
    path = Path(path)
    fingerprint = sweep_fingerprint_path(path)
    return (
        (path / "domain_sweep_summary.csv").exists()
        and (path / "summary.json").exists()
        and fingerprint.exists()
        and fingerprint.read_text(encoding="utf-8").strip() == file_hash(manifest_path)
    )


def write_sweep_fingerprint(path: Path, manifest_path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    sweep_fingerprint_path(path).write_text(file_hash(manifest_path), encoding="utf-8")


def print_sweep_sources(manifest_path: Path) -> None:
    print("Sweep image sources:")
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            chosen = row.get("image_url") or row.get("local_path") or row.get("source_url")
            print(f"- {row['id']} [{row['domain_group']} -> {row['expected_class']}]: {chosen}")


def reuse_or_run_sweep(output_dir: Path, drive_dir: Path, detector_weight: Path, name: str, force: bool = False) -> Path:
    print_sweep_sources(MANIFEST_PATH)
    if sweep_ready(output_dir, MANIFEST_PATH) and not force:
        print(f"Reusing local sweep with matching manifest: {output_dir}")
        mirror_path(output_dir, drive_dir)
        return output_dir
    if sweep_ready(drive_dir, MANIFEST_PATH) and not force:
        print(f"Restoring matching sweep from Drive: {drive_dir}")
        mirror_path(drive_dir, output_dir)
        return output_dir
    run_process([
        sys.executable, "scripts/domain_sweep.py",
        "--manifest", str(MANIFEST_PATH),
        "--detector", str(detector_weight),
        "--severity", str(SEVERITY_WEIGHT),
        "--output", str(output_dir),
        "--thresholds", "0.45,0.30,0.20,0.10",
        "--iou", "0.45",
        "--annotate-conf", "0.20",
    ], log_name=name)
    write_sweep_fingerprint(output_dir, MANIFEST_PATH)
    mirror_path(output_dir, drive_dir)
    return output_dir


baseline_sweep = reuse_or_run_sweep(BASELINE_SWEEP_DIR, DRIVE_BASELINE_SWEEP_DIR, DETECTOR_WEIGHT, "baseline_domain_sweep", FORCE_SWEEP)
print((baseline_sweep / "summary.json").read_text(encoding="utf-8"))
mark_done("baseline_domain_sweep", {"path": str(baseline_sweep), "manifest_sha256": file_hash(MANIFEST_PATH)})
'''
    ),
    md(
        """
## Cell 7: Automated Target-Domain Recollection and Conditional Retraining

This stage collects additional facade/building defect datasets, remaps labels into the five classes, decides whether domain retraining is justified, builds the adapted dataset, and fine-tunes Stage 1 only when useful.
"""
    ),
    code(
        r'''
TARGET_YOLO = REPO_ROOT / "domain_adaptation" / "target_yolo"
AUTO_RAW = REPO_ROOT / "domain_adaptation" / "auto_raw"
TARGET_SUMMARY = TARGET_YOLO / "auto_collection_summary.json"
DOMAIN_ADAPTED_DATASET = REPO_ROOT / "merged_dataset_domain_adapted"
HARD_NEGATIVES = REPO_ROOT / "domain_adaptation" / "hard_negatives"


def count_images(root: Path) -> int:
    return sum(1 for path in Path(root).glob("**/*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}) if Path(root).exists() else 0


def target_label_count(target_root: Path) -> int:
    total = 0
    for label_path in target_root.glob("**/*.txt"):
        total += len([line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    return total


if FORCE_DOMAIN_COLLECTION or not TARGET_SUMMARY.exists() or count_images(TARGET_YOLO) == 0:
    if not ROBOFLOW_API_KEY:
        print("ROBOFLOW_API_KEY is missing; target-domain recollection skipped.")
    else:
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


def load_sweep_rows(summary_csv: Path) -> list[dict]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def failure_count(rows: list[dict], thresholds=("0.45", "0.3", "0.30", "0.2", "0.20")) -> int:
    return sum(1 for row in rows if str(row.get("threshold", "")) in thresholds and row.get("match") == "false")


rows = load_sweep_rows(BASELINE_SWEEP_DIR / "domain_sweep_summary.csv")
weak_rows = failure_count(rows)
target_images = count_images(TARGET_YOLO)
target_labels = target_label_count(TARGET_YOLO)
NEEDS_DOMAIN_RETRAIN = FORCE_DOMAIN_RETRAIN or (DATASETS_AVAILABLE and weak_rows > 0 and target_images >= 10 and target_labels >= 10)
STATE.setdefault("decisions", {})["needs_domain_retrain"] = NEEDS_DOMAIN_RETRAIN
STATE["decisions"]["domain_retrain_reason"] = {
    "datasets_available": DATASETS_AVAILABLE,
    "weak_sweep_rows": weak_rows,
    "target_images": target_images,
    "target_labels": target_labels,
    "force": FORCE_DOMAIN_RETRAIN,
}
save_state()
print(json.dumps(STATE["decisions"]["domain_retrain_reason"], indent=2))


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

    domain_args = {
        "epochs": 60, "imgsz": 640, "patience": 15, "optimizer": "AdamW", "lr0": 0.001,
        "cos_lr": True, "close_mosaic": 8, "mosaic": 0.45, "mixup": 0.03,
        "copy_paste": 0.05, "degrees": 2.0, "translate": 0.06, "scale": 0.25,
        "fliplr": 0.50, "hsv_s": 0.35, "hsv_v": 0.30,
    }
    trained_domain_detector = train_or_resume(
        "detector",
        DETECTOR_WEIGHT if DETECTOR_WEIGHT.exists() else "yolo11s.pt",
        DOMAIN_ADAPTED_DATASET / "data.yaml",
        "stage1_domain_adapted_detector",
        DOMAIN_DETECTOR_WEIGHT,
        "defect_detector_domain_adapted.pt",
        domain_args,
        FORCE_DOMAIN_RETRAIN,
    )
    if PROMOTE_DOMAIN_ADAPTED_MODEL:
        shutil.copy2(trained_domain_detector, DETECTOR_WEIGHT)
        copy_weight_to_local(DETECTOR_WEIGHT, "defect_detector.pt")
    mark_done("domain_detector_training", {"detector": str(trained_domain_detector), "promoted": PROMOTE_DOMAIN_ADAPTED_MODEL})
else:
    print("Domain retraining skipped by decision gate.")
'''
    ),
    md(
        """
## Cell 8: Post-Adaptation Check and Deployment Weights

This final section reruns the sweep only after an adapted detector is trained, then makes canonical local deployment names available for the dashboard/backend.
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


if STATE.get("steps", {}).get("domain_detector_training") and DOMAIN_DETECTOR_WEIGHT.exists():
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
else:
    print("No new domain-adapted detector was trained in this run.")
    if (BASELINE_SWEEP_DIR / "domain_sweep_summary.csv").exists():
        print(json.dumps({"baseline_at_conf_0_20": summarize_matches(BASELINE_SWEEP_DIR / "domain_sweep_summary.csv", "0.2")}, indent=2))

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

output_path = REPO_ROOT / "two_stage_yolo_defect_pipeline_colab.ipynb"
output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {output_path} with {len(cells)} cells")
print(f"Markdown cells: {sum(cell['cell_type'] == 'markdown' for cell in cells)}")
print(f"Code cells: {sum(cell['cell_type'] == 'code' for cell in cells)}")
