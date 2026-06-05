from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

MATERIALIZED_FILES = [
    "configs/merge_config.yaml",
    "configs/domain_sweep_manifest.csv",
    "configs/production_eval_manifest.csv",
    "scripts/merge_datasets.py",
    "scripts/extract_severity_crops.py",
    "scripts/domain_sweep.py",
    "scripts/auto_collect_domain_sources.py",
    "scripts/build_domain_adaptation_dataset.py",
    "scripts/build_balanced_detection_dataset.py",
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


def runtime_bootstrap() -> str:
    return r'''
# Runtime bootstrap: lets this cell recover paths/helpers even when run directly.
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
    if not Path("/content/drive/MyDrive").exists():
        drive.mount("/content/drive")
except Exception as exc:
    print(f"Google Drive mount skipped or unavailable: {exc}")

GITHUB_REPO_URL = globals().get("GITHUB_REPO_URL", "https://github.com/PsychicFireSong/AIEngGroupProj.git")
REPO_ROOT_OVERRIDE = globals().get("REPO_ROOT_OVERRIDE", "")

FORCE_DOWNLOAD_DATASETS = globals().get("FORCE_DOWNLOAD_DATASETS", False)
FORCE_REBUILD_DATASETS = globals().get("FORCE_REBUILD_DATASETS", False)
FORCE_INITIAL_TRAINING = globals().get("FORCE_INITIAL_TRAINING", False)
FORCE_SWEEP = globals().get("FORCE_SWEEP", False)
FORCE_DOMAIN_COLLECTION = globals().get("FORCE_DOMAIN_COLLECTION", False)
FORCE_DOMAIN_RETRAIN = globals().get("FORCE_DOMAIN_RETRAIN", False)
FORCE_SEVERITY_RETRAIN = globals().get("FORCE_SEVERITY_RETRAIN", False)
FORCE_PRODUCTION_CHECK = globals().get("FORCE_PRODUCTION_CHECK", False)
SAFE_RECOVERY_MODE = globals().get("SAFE_RECOVERY_MODE", True)
ALLOW_BASELINE_TRAINING = globals().get("ALLOW_BASELINE_TRAINING", False)
ALLOW_TRAINING_RESUME = globals().get("ALLOW_TRAINING_RESUME", False)
ALLOW_BALANCED_RETRAIN = globals().get("ALLOW_BALANCED_RETRAIN", False)
ALLOW_SEVERITY_RETRAIN = globals().get("ALLOW_SEVERITY_RETRAIN", False)
ALLOW_UNCACHED_SWEEP_RUNS = globals().get("ALLOW_UNCACHED_SWEEP_RUNS", False)
PROMOTE_DOMAIN_ADAPTED_MODEL = globals().get("PROMOTE_DOMAIN_ADAPTED_MODEL", False)
PROMOTE_ONLY_AFTER_PRODUCTION_CHECK = globals().get("PROMOTE_ONLY_AFTER_PRODUCTION_CHECK", True)
CACHE_RAW_DATASETS_TO_DRIVE = globals().get("CACHE_RAW_DATASETS_TO_DRIVE", True)
ALLOW_OVERWRITE_DRIVE_RAW_CACHE = globals().get("ALLOW_OVERWRITE_DRIVE_RAW_CACHE", False)
BALANCED_DETECTOR_MODEL_SEED = globals().get("BALANCED_DETECTOR_MODEL_SEED", "yolo11m.pt")
BALANCED_DETECTOR_BATCH = globals().get("BALANCED_DETECTOR_BATCH", 8)
BALANCED_DETECTOR_WARM_START_FROM_CURRENT = globals().get("BALANCED_DETECTOR_WARM_START_FROM_CURRENT", False)

DRIVE_OUTPUT_FOLDER_ID = "1X4IGra-ySuPqbc_PYIs2pl3yhO1HyIbX"
DRIVE_ROOT = Path("/content/drive/MyDrive")


def resolve_drive_output_root() -> Path:
    candidates = [
        DRIVE_ROOT / "AIEngGroupProj_colab_outputs",
        Path("/content/drive/.shortcut-targets-by-id") / DRIVE_OUTPUT_FOLDER_ID / "AIEngGroupProj_colab_outputs",
    ]
    shortcut_root = Path("/content/drive/.shortcut-targets-by-id")
    if shortcut_root.exists():
        candidates.extend(shortcut_root.glob("*/AIEngGroupProj_colab_outputs"))
        candidates.extend(shortcut_root.glob("*/*/AIEngGroupProj_colab_outputs"))
    for candidate in candidates:
        if (candidate / "runs").exists() or (candidate / "weights").exists():
            return candidate
    return candidates[0]


DRIVE_OUTPUT_ROOT = resolve_drive_output_root()
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


ROBOFLOW_API_KEY = globals().get("ROBOFLOW_API_KEY", get_secret("ROBOFLOW_API_KEY", ""))
KAGGLE_USERNAME = globals().get("KAGGLE_USERNAME", get_secret("KAGGLE_USERNAME"))
KAGGLE_KEY = globals().get("KAGGLE_KEY", get_secret("KAGGLE_KEY", ""))


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
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Could not read pipeline state, starting fresh state: {exc}")
    return {"steps": {}, "decisions": {}}


STATE = globals().get("STATE", load_state())


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
    raise FileNotFoundError("Could not find AIEngGroupProj. Run Cell 1 once, set REPO_ROOT_OVERRIDE, clone the repo, or upload AIEngGroupProj.zip to Drive.")


REPO_ROOT = Path(globals().get("REPO_ROOT", ensure_repo_root())).resolve()
os.chdir(REPO_ROOT)
WEIGHTS_DIR = REPO_ROOT / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

DETECTOR_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector.pt"
SEVERITY_WEIGHT = DRIVE_WEIGHTS_ROOT / "severity_cls.pt"
DOMAIN_DETECTOR_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector_domain_adapted.pt"
DOMAIN_SEVERITY_WEIGHT = DRIVE_WEIGHTS_ROOT / "severity_cls_domain_adapted.pt"
BALANCED_DETECTOR_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector_balanced_candidate.pt"

BASELINE_SWEEP_DIR = REPO_ROOT / "output" / "domain_sweep"
DRIVE_BASELINE_SWEEP_DIR = DRIVE_OUTPUT_ROOT / "domain_sweep" / "baseline"
DRIVE_SWEEP_IMAGE_CACHE = DRIVE_OUTPUT_ROOT / "domain_sweep_image_cache"
DRIVE_SWEEP_IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
REQUIRED_DETECTION_CLASSES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]


def candidate_run_roots() -> list[Path]:
    roots = [DRIVE_RUNS_ROOT, DRIVE_OUTPUT_ROOT / "runs", DRIVE_ROOT / "runs"]
    shortcut_root = Path("/content/drive/.shortcut-targets-by-id")
    if shortcut_root.exists():
        roots.extend(shortcut_root.glob("*/AIEngGroupProj_colab_outputs/runs"))
        roots.extend(shortcut_root.glob("*/*/AIEngGroupProj_colab_outputs/runs"))
    return list(dict.fromkeys(path for path in roots if path.exists()))


def checkpoint_candidates(kind: str, checkpoint_name: str = "best.pt") -> list[Path]:
    final_candidates = []
    if kind == "detector":
        final_candidates = [
            DETECTOR_WEIGHT,
            DOMAIN_DETECTOR_WEIGHT,
            BALANCED_DETECTOR_WEIGHT,
            WEIGHTS_DIR / "defect_detector.pt",
            WEIGHTS_DIR / "defect_detector_domain_adapted.pt",
            WEIGHTS_DIR / "defect_detector_balanced_candidate.pt",
        ]
        include_terms, exclude_terms = ("stage1", "defect", "detector", "detect"), ("severity", "classify", "cls", "stage2")
    else:
        final_candidates = [
            SEVERITY_WEIGHT,
            DOMAIN_SEVERITY_WEIGHT,
            WEIGHTS_DIR / "severity_cls.pt",
            WEIGHTS_DIR / "severity_cls_domain_adapted.pt",
        ]
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


def recover_weight_from_completed_run(
    include_terms: tuple[str, ...],
    target_weight: Path,
    local_name: str,
    checkpoint_name: str = "best.pt",
) -> Path | None:
    if target_weight.exists():
        copy_weight_to_local(target_weight, local_name)
        return target_weight
    candidates = []
    for root in candidate_run_roots():
        for checkpoint in root.glob(f"**/weights/{checkpoint_name}"):
            lowered = str(checkpoint).replace("\\", "/").lower()
            if all(term.lower() in lowered for term in include_terms):
                candidates.append(checkpoint)
    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    best = candidates[0]
    target_weight.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target_weight)
    print(f"Recovered {target_weight.name} from completed run: {best}")
    copy_weight_to_local(target_weight, local_name)
    return target_weight


def latest_last_checkpoint(kind: str) -> Path | None:
    candidates = checkpoint_candidates(kind, "last.pt")
    return candidates[0] if candidates else None


def recover_completed_weight(kind: str, final_weight: Path, local_name: str, allow_kind_search: bool = True) -> Path | None:
    if final_weight.exists() and not FORCE_INITIAL_TRAINING:
        print(f"Using existing canonical {kind} weight: {final_weight}")
        copy_weight_to_local(final_weight, local_name)
        return final_weight
    if not allow_kind_search:
        return None
    candidates = checkpoint_candidates(kind, "best.pt")
    if candidates and not FORCE_INITIAL_TRAINING:
        best = candidates[0]
        print(f"Recovered completed {kind} training from: {best}")
        final_weight.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, final_weight)
        copy_weight_to_local(final_weight, local_name)
        return final_weight
    return None


def train_device_kwargs() -> dict:
    try:
        import torch
        return {"device": 0} if torch.cuda.is_available() else {}
    except Exception:
        return {}


def train_or_resume(
    kind: str,
    seed_weights: str | Path,
    data_path: Path,
    run_name: str,
    final_weight: Path,
    local_name: str,
    train_args: dict,
    force: bool = False,
    allow_kind_search: bool = True,
    allow_new_training: bool = False,
    allow_resume_training: bool = False,
) -> Path:
    from ultralytics import YOLO

    recovered = recover_completed_weight(kind, final_weight, local_name, allow_kind_search)
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
        if not allow_resume_training:
            raise RuntimeError(
                f"Interrupted {kind} checkpoint exists at {last}, but SAFE_RECOVERY_MODE is on. "
                "Set ALLOW_TRAINING_RESUME=True for this run if you intentionally want to spend compute resuming it."
            )
        print(f"Resuming interrupted {kind} training from: {last}")
        YOLO(str(last)).train(resume=True)
    else:
        if not (force or allow_new_training):
            raise RuntimeError(
                f"No completed {kind} weight was recovered, and fresh training is disabled in SAFE_RECOVERY_MODE. "
                "Use existing Drive weights or set the matching ALLOW_*_TRAINING flag to True."
            )
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


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sweep_fingerprint_path(path: Path) -> Path:
    return Path(path) / "manifest_sha256.txt"


def write_sweep_fingerprint(path: Path, manifest_path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
    sweep_fingerprint_path(path).write_text(file_hash(manifest_path), encoding="utf-8")


def seed_sweep_image_cache(output_dir: Path) -> None:
    output_images = Path(output_dir) / "images"
    source_dirs = [DRIVE_SWEEP_IMAGE_CACHE, BASELINE_SWEEP_DIR / "images", DRIVE_BASELINE_SWEEP_DIR / "images"]
    copied = 0
    for source_dir in source_dirs:
        if not source_dir.exists() or source_dir.resolve() == output_images.resolve():
            continue
        output_images.mkdir(parents=True, exist_ok=True)
        for image_path in source_dir.iterdir():
            if image_path.is_file():
                target = output_images / image_path.name
                if not target.exists():
                    shutil.copy2(image_path, target)
                    copied += 1
    if copied:
        print(f"Seeded sweep image cache with {copied} previously downloaded images.")


def refresh_sweep_image_cache(output_dir: Path) -> None:
    image_dir = Path(output_dir) / "images"
    if not image_dir.exists():
        return
    copied = 0
    for image_path in image_dir.iterdir():
        if image_path.is_file():
            target = DRIVE_SWEEP_IMAGE_CACHE / image_path.name
            if not target.exists() or target.stat().st_size == 0:
                shutil.copy2(image_path, target)
                copied += 1
    if copied:
        print(f"Cached {copied} sweep images to Drive.")


def load_sweep_rows(summary_csv: Path) -> list[dict]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


EXPECTED_DATASETS = globals().get("EXPECTED_DATASETS", [
    "Finale.yolov11",
    "Internal Wall Defect.yolov11",
    "Concrete defect detection.yolov11",
    "Corrosion YOLOv8.yolov11",
    "metal corrosion.yolov11",
    "Pothole detection YOLOv8.yolov11",
    "archive",
])


def simple_yolo_dataset_ready(path: Path) -> bool:
    path = Path(path)
    return (path / "data.yaml").exists() and any((path / split / "images").exists() for split in ("train", "valid", "val", "test"))


AVAILABLE_BASELINE_DATASETS = globals().get(
    "AVAILABLE_BASELINE_DATASETS",
    [name for name in EXPECTED_DATASETS if simple_yolo_dataset_ready(REPO_ROOT / name)],
)
DATASETS_AVAILABLE = globals().get("DATASETS_AVAILABLE", len(AVAILABLE_BASELINE_DATASETS) == len(EXPECTED_DATASETS))
missing = globals().get("missing", [name for name in EXPECTED_DATASETS if name not in AVAILABLE_BASELINE_DATASETS])
BASELINE_DATASETS_BUILDABLE = globals().get(
    "BASELINE_DATASETS_BUILDABLE",
    DATASETS_AVAILABLE or bool(AVAILABLE_BASELINE_DATASETS) or previous_training_available(),
)
MERGED_DATASET = globals().get("MERGED_DATASET", REPO_ROOT / "merged_dataset")
SEVERITY_DATASET = globals().get("SEVERITY_DATASET", REPO_ROOT / "severity_dataset")


def detection_labels_from_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return summary.get("counts", {}).get("labels", {})


def dataset_has_all_detection_classes(dataset_root: Path) -> bool:
    labels = detection_labels_from_summary(Path(dataset_root) / "merge_summary.json")
    return all(labels.get(name, 0) > 0 for name in REQUIRED_DETECTION_CLASSES)


MERGED_DATASET_READY_FOR_TRAINING = globals().get(
    "MERGED_DATASET_READY_FOR_TRAINING",
    dataset_has_all_detection_classes(Path(MERGED_DATASET)),
)
print(f"Runtime ready. Repo: {REPO_ROOT}")
'''


def bootstrapped(text: str) -> dict:
    return code(runtime_bootstrap() + "\n\n# --- Cell-specific logic ---\n" + text)


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

Run order:

1. Cell 1 sets Drive, repo paths, recovery flags, and shared helpers.
2. Cell 2 restores/downloads raw baseline datasets and validates source labels.
3. Cell 3 writes the latest configs and scripts into the Colab repo.
4. Cell 4 builds or restores baseline detection/severity datasets.
5. Cell 5 recovers existing model weights first, and trains only when explicitly allowed.
6. Cell 6 restores or runs the baseline domain sweep.
7. Cell 7 recollects target-domain data, builds the balanced candidate dataset, and retrains only when explicitly allowed.
8. Cell 8 restores or runs post-adaptation checks, then prepares deployment weights.

Recommended rerun after notebook edits: Cell 1 -> Cell 2 -> Cell 3 -> Cell 4 -> Cell 5 -> Cell 6 -> Cell 7 -> Cell 8. Cells with cached artifacts will reuse them unless you enable the matching force or allow flag.

Recovery behavior:

- Uses Google Drive for logs, weights, runs, cached datasets, sweep reports, and `pipeline_state.json`.
- Recovers completed training from existing `best.pt` before trying to retrain.
- Runs in `SAFE_RECOVERY_MODE` by default: completed weights are recovered automatically, but fresh/resumed GPU training requires an explicit `ALLOW_*` flag.
- Resumes interrupted training from `last.pt` only when `ALLOW_TRAINING_RESUME=True`.
- Restores cached inference/sweep reports first; uncached sweeps run only when `ALLOW_UNCACHED_SWEEP_RUNS=True` or a force flag is enabled.
- Can recover deployment weights even when raw baseline datasets are not needed in that session.

This local working-copy notebook includes your API keys directly for individual experimentation. Do not push or share this local version unless you remove the embedded keys first.
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

GITHUB_REPO_URL = "https://github.com/PsychicFireSong/AIEngGroupProj.git"     # Optional fresh Colab clone URL.
REPO_ROOT_OVERRIDE = ""  # Optional existing path, for example: /content/AIEngGroupProj

FORCE_DOWNLOAD_DATASETS = False
FORCE_REBUILD_DATASETS = False
FORCE_INITIAL_TRAINING = False
FORCE_SWEEP = False
FORCE_DOMAIN_COLLECTION = False
FORCE_DOMAIN_RETRAIN = False
FORCE_SEVERITY_RETRAIN = False
FORCE_PRODUCTION_CHECK = False
SAFE_RECOVERY_MODE = True
ALLOW_BASELINE_TRAINING = False
ALLOW_TRAINING_RESUME = False
ALLOW_BALANCED_RETRAIN = False
ALLOW_SEVERITY_RETRAIN = False
ALLOW_UNCACHED_SWEEP_RUNS = False
PROMOTE_DOMAIN_ADAPTED_MODEL = False
PROMOTE_ONLY_AFTER_PRODUCTION_CHECK = True
CACHE_RAW_DATASETS_TO_DRIVE = True
ALLOW_OVERWRITE_DRIVE_RAW_CACHE = False
BALANCED_DETECTOR_MODEL_SEED = "yolo11m.pt"  # Better capacity than yolo11s; switch to yolo11s.pt if Colab time is tight.
BALANCED_DETECTOR_BATCH = 8
BALANCED_DETECTOR_WARM_START_FROM_CURRENT = False  # False means true architecture upgrade; True reuses current YOLO11s weights.

DRIVE_OUTPUT_FOLDER_ID = "1X4IGra-ySuPqbc_PYIs2pl3yhO1HyIbX"
DRIVE_ROOT = Path("/content/drive/MyDrive")


def resolve_drive_output_root() -> Path:
    candidates = [
        DRIVE_ROOT / "AIEngGroupProj_colab_outputs",
        Path("/content/drive/.shortcut-targets-by-id") / DRIVE_OUTPUT_FOLDER_ID / "AIEngGroupProj_colab_outputs",
    ]
    shortcut_root = Path("/content/drive/.shortcut-targets-by-id")
    if shortcut_root.exists():
        candidates.extend(shortcut_root.glob("*/AIEngGroupProj_colab_outputs"))
        candidates.extend(shortcut_root.glob("*/*/AIEngGroupProj_colab_outputs"))
    for candidate in candidates:
        if (candidate / "runs").exists() or (candidate / "weights").exists():
            return candidate
    return candidates[0]


DRIVE_OUTPUT_ROOT = resolve_drive_output_root()
DRIVE_RUNS_ROOT = DRIVE_OUTPUT_ROOT / "runs"
DRIVE_WEIGHTS_ROOT = DRIVE_OUTPUT_ROOT / "weights"
LOG_ROOT = DRIVE_OUTPUT_ROOT / "logs"
STATE_PATH = DRIVE_OUTPUT_ROOT / "pipeline_state.json"
for path in (DRIVE_OUTPUT_ROOT, DRIVE_RUNS_ROOT, DRIVE_WEIGHTS_ROOT, LOG_ROOT):
    path.mkdir(parents=True, exist_ok=True)

RUN_SEQUENCE = [
    "Cell 1: setup Drive/repo/recovery flags",
    "Cell 2: acquire and validate raw datasets",
    "Cell 3: materialize latest configs/scripts",
    "Cell 4: build or restore baseline datasets",
    "Cell 5: recover/train baseline weights",
    "Cell 6: restore/run baseline sweep",
    "Cell 7: recollect target data and optionally train balanced candidate",
    "Cell 8: restore/run production check and prepare deployment weights",
]
print("Notebook run sequence:")
for item in RUN_SEQUENCE:
    print(f"- {item}")
print(
    "Safe defaults: fresh/resumed training and uncached sweeps are disabled until you set the matching ALLOW_* flag."
)


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


ROBOFLOW_API_KEY = get_secret("ROBOFLOW_API_KEY", "")
KAGGLE_USERNAME = get_secret("KAGGLE_USERNAME")
KAGGLE_KEY = get_secret("KAGGLE_KEY", "")


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
BALANCED_DETECTOR_WEIGHT = DRIVE_WEIGHTS_ROOT / "defect_detector_balanced_candidate.pt"


def candidate_run_roots() -> list[Path]:
    roots = [DRIVE_RUNS_ROOT, DRIVE_OUTPUT_ROOT / "runs", DRIVE_ROOT / "runs"]
    shortcut_root = Path("/content/drive/.shortcut-targets-by-id")
    if shortcut_root.exists():
        roots.extend(shortcut_root.glob("*/AIEngGroupProj_colab_outputs/runs"))
        roots.extend(shortcut_root.glob("*/*/AIEngGroupProj_colab_outputs/runs"))
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


def recover_weight_from_completed_run(
    include_terms: tuple[str, ...],
    target_weight: Path,
    local_name: str,
    checkpoint_name: str = "best.pt",
) -> Path | None:
    """Recover a completed run checkpoint from Drive roots, including Colab shortcut targets."""
    if target_weight.exists():
        copy_weight_to_local(target_weight, local_name)
        return target_weight
    candidates = []
    for root in candidate_run_roots():
        for checkpoint in root.glob(f"**/weights/{checkpoint_name}"):
            lowered = str(checkpoint).replace("\\", "/").lower()
            if all(term.lower() in lowered for term in include_terms):
                candidates.append(checkpoint)
    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    best = candidates[0]
    target_weight.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target_weight)
    print(f"Recovered {target_weight.name} from completed run: {best}")
    copy_weight_to_local(target_weight, local_name)
    return target_weight


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
    bootstrapped(
        r'''
DATASET_SOURCES = [
    {"folder": "Finale.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/ponti/finale-3pwus", "workspace": "ponti", "project": "finale-3pwus", "version": 1, "expected_unified": ["crack", "spalling", "paint_degradation"], "min_expected_matches": 3},
    {"folder": "Internal Wall Defect.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/chew-poh-yee/internal-wall-defect", "workspace": "chew-poh-yee", "project": "internal-wall-defect", "version": 1, "expected_unified": ["paint_degradation"], "min_expected_matches": 1},
    {"folder": "Concrete defect detection.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/defect-detection-0atjo/concrete-defect-detection-zuym8", "workspace": "defect-detection-0atjo", "project": "concrete-defect-detection-zuym8", "version": 1, "expected_unified": ["crack", "spalling"], "min_expected_matches": 2},
    {"folder": "Corrosion YOLOv8.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/corrosion-yolo-v8/corrosion-yolov8", "workspace": "corrosion-yolo-v8", "project": "corrosion-yolov8", "version": 1, "expected_unified": ["corrosion"], "min_expected_matches": 1},
    {"folder": "metal corrosion.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/yolov11-eorob/metal-corrosion", "workspace": "yolov11-eorob", "project": "metal-corrosion", "version": 1, "version_candidates": [2, 3, 4, 5, 6, 7, 8, 9, 10], "expected_unified": ["corrosion", "paint_degradation"], "min_expected_matches": 2},
    {"folder": "Pothole detection YOLOv8.yolov11", "provider": "roboflow", "url": "https://universe.roboflow.com/pe1-dtzop/pothole-detection-yolov8-8mspr", "workspace": "pe1-dtzop", "project": "pothole-detection-yolov8-8mspr", "version": 1, "expected_unified": ["pothole"], "min_expected_matches": 1},
    {"folder": "archive", "provider": "kaggle", "url": "https://www.kaggle.com/datasets/muskanverma24/pothole-detection-dataset-yolov11-optimized", "dataset": "muskanverma24/pothole-detection-dataset-yolov11-optimized", "expected_unified": ["pothole"], "min_expected_matches": 1},
]
EXPECTED_DATASETS = [source["folder"] for source in DATASET_SOURCES]
SOURCE_BY_FOLDER = {source["folder"]: source for source in DATASET_SOURCES}
SOURCE_LABEL_SYNONYMS = {
    "crack": "crack", "cracks": "crack", "cracking": "crack", "wall crack": "crack", "concrete crack": "crack",
    "fessura": "crack", "fessura diagonale": "crack", "fessura orizzontale": "crack", "fessura verticale": "crack",
    "spalling": "spalling", "spall": "spalling", "rebar": "spalling", "scaling": "spalling", "exposed reinforcement": "spalling",
    "exposed iron": "spalling", "esposizione ferri": "spalling", "delamination": "spalling", "delaminazione": "spalling",
    "honeycombing": "spalling", "vespai": "spalling",
    "corrosion": "corrosion", "rust": "corrosion", "rust stain": "corrosion", "ruststrain": "corrosion", "metal corrosion": "corrosion",
    "pitted surface": "corrosion", "rolled-in scale": "corrosion", "rolled in scale": "corrosion",
    "pothole": "pothole", "potholes": "pothole", "pot hole": "pothole", "road pothole": "pothole",
    "paint degradation": "paint_degradation", "paint_degradation": "paint_degradation", "abrasione": "paint_degradation",
    "peeling paint": "paint_degradation",
    "paint defect": "paint_degradation", "paint defects": "paint_degradation", "stain marks": "paint_degradation",
    "efflorescence": "paint_degradation", "efflorescenza": "paint_degradation", "moisture marks": "paint_degradation",
    "tracce umidita": "paint_degradation",
    "patches": "paint_degradation", "scratches": "paint_degradation", "mold": "paint_degradation", "dirt": "paint_degradation",
    "inclusion": "paint_degradation", "pin holes": "paint_degradation", "rough and patchy surface": "paint_degradation",
    "paint drips": "paint_degradation", "trowels marks": "paint_degradation",
}
RAW_CACHE_DIR = DRIVE_OUTPUT_ROOT / "raw_dataset_cache"
RAW_CACHE_BACKUP_DIR = DRIVE_OUTPUT_ROOT / "raw_dataset_cache_backups"
LOCAL_DOWNLOAD_ROOT = REPO_ROOT / "raw_downloads"
RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_CACHE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
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


def normalize_source_label(value: str) -> str:
    value = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(value.split())


def dataset_profile(path: Path) -> dict:
    path = Path(path)
    data_yaml = path / "data.yaml"
    names = []
    if data_yaml.exists():
        try:
            import yaml
            data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
            raw_names = data.get("names", [])
            if isinstance(raw_names, dict):
                names = [str(value) for _, value in sorted(raw_names.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 9999)]
            else:
                names = [str(value) for value in raw_names]
        except Exception as exc:
            names = [f"profile_error:{exc}"]
    image_counts = {}
    label_counts = {}
    for split in ("train", "valid", "val", "test"):
        image_dir = path / split / "images"
        label_dir = path / split / "labels"
        image_counts[split] = len(list(image_dir.glob("*"))) if image_dir.exists() else 0
        label_counts[split] = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
    return {"path": str(path), "ready": dataset_ready(path), "names": names, "image_counts": image_counts, "label_counts": label_counts}


def unified_labels_from_profile(profile: dict) -> set[str]:
    unified = set()
    for name in profile.get("names", []):
        mapped = SOURCE_LABEL_SYNONYMS.get(normalize_source_label(name))
        if mapped:
            unified.add(mapped)
    return unified


def dataset_ready_for_source(source: dict, path: Path) -> bool:
    if not dataset_ready(path):
        return False
    expected = set(source.get("expected_unified") or [])
    if not expected:
        return True
    profile = dataset_profile(path)
    actual = unified_labels_from_profile(profile)
    matched = actual & expected
    min_expected_matches = int(source.get("min_expected_matches", 1))
    if len(matched) >= min_expected_matches:
        return True
    print(
        "WARNING: Dataset folder looks mismatched and will not be treated as ready: "
        f"{source['folder']} expected at least {min_expected_matches} of {sorted(expected)}, "
        f"matched={sorted(matched)}, got names={profile.get('names', [])}"
    )
    return False


def expected_missing() -> list[str]:
    return [source["folder"] for source in DATASET_SOURCES if not dataset_ready_for_source(source, REPO_ROOT / source["folder"])]


def find_dataset_root(search_root: Path) -> Path | None:
    candidates = []
    for data_yaml in Path(search_root).rglob("data.yaml"):
        root = data_yaml.parent
        if any((root / split / "images").exists() for split in ("train", "valid", "val", "test")):
            candidates.append(root)
    candidates.sort(key=lambda path: len(path.parts))
    return candidates[0] if candidates else None


def compact_name(value: str) -> str:
    return "".join(char.lower() for char in str(value) if char.isalnum())


def find_dataset_root_for_source(source: dict, search_root: Path, allow_single_fallback: bool = True) -> Path | None:
    search_root = Path(search_root)
    folder_key = compact_name(source["folder"])
    project_key = compact_name(source.get("project", ""))
    dataset_key = compact_name(source.get("dataset", "").split("/")[-1] if source.get("dataset") else "")
    preferred_roots = [
        search_root / source["folder"],
        *[candidate for candidate in search_root.glob("*") if compact_name(candidate.name) in {folder_key, project_key, dataset_key}],
    ]
    for candidate in preferred_roots:
        if candidate.exists():
            root = find_dataset_root(candidate)
            if root and dataset_ready_for_source(source, root):
                return root

    roots = []
    for data_yaml in search_root.rglob("data.yaml"):
        root = data_yaml.parent
        if not dataset_ready(root):
            continue
        path_key = compact_name(str(root.relative_to(search_root)))
        score = 0
        for key in (folder_key, project_key, dataset_key):
            if key and key in path_key:
                score += 2
        if dataset_ready_for_source(source, root):
            score += 1
        if score > 0:
            roots.append((score, len(root.parts), root))
    if roots:
        roots.sort(key=lambda item: (-item[0], item[1]))
        return roots[0][2]

    all_roots = [data_yaml.parent for data_yaml in search_root.rglob("data.yaml") if dataset_ready(data_yaml.parent)]
    if allow_single_fallback and len(all_roots) == 1 and dataset_ready_for_source(source, all_roots[0]):
        return all_roots[0]
    if all_roots:
        print(
            f"Could not safely choose a dataset root for {source['folder']} under {search_root}. "
            f"Candidate roots: {[str(root) for root in all_roots[:10]]}"
        )
    return None


def materialize_dataset(source: dict, downloaded_root: Path) -> bool:
    dataset_root = find_dataset_root_for_source(source, downloaded_root, allow_single_fallback=True)
    if dataset_root is None:
        return False
    target = REPO_ROOT / source["folder"]
    if target.exists() and FORCE_DOWNLOAD_DATASETS:
        shutil.rmtree(target)
    if not dataset_ready_for_source(source, target):
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(dataset_root, target)
    return dataset_ready_for_source(source, target)


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
            if not dataset_ready_for_source(source, REPO_ROOT / source["folder"]):
                dataset_root = find_dataset_root_for_source(source, extract_root, allow_single_fallback=False)
                if dataset_root is None:
                    print(f"No safe matching folder for {source['folder']} inside raw_datasets.zip")
                    continue
                materialize_dataset(source, dataset_root)
    for source in DATASET_SOURCES:
        name = source["folder"]
        cached = RAW_CACHE_DIR / name
        target = REPO_ROOT / name
        if dataset_ready_for_source(source, cached) and not dataset_ready_for_source(source, target):
            print(f"Restoring cached dataset: {name}")
            shutil.copytree(cached, target, dirs_exist_ok=True)
        elif cached.exists() and not dataset_ready_for_source(source, target):
            print(f"Trying nested Drive cache restore for: {name}")
            materialize_dataset(source, cached)


def cache_dataset_to_drive(name: str) -> None:
    if not CACHE_RAW_DATASETS_TO_DRIVE:
        return
    source_cfg = SOURCE_BY_FOLDER.get(name, {"folder": name})
    source, target = REPO_ROOT / name, RAW_CACHE_DIR / name
    if not dataset_ready_for_source(source_cfg, source):
        print(f"Not caching {name}; source folder failed readiness/integrity checks.")
        return
    source_profile = dataset_profile(source)
    target_profile = dataset_profile(target) if target.exists() else {"ready": False}
    profile_dir = DRIVE_OUTPUT_ROOT / "summaries" / "raw_dataset_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / f"{name.replace('/', '_')}.json").write_text(
        json.dumps({"source": source_profile, "target_before": target_profile}, indent=2),
        encoding="utf-8",
    )
    if dataset_ready_for_source(source_cfg, target) and not ALLOW_OVERWRITE_DRIVE_RAW_CACHE:
        print(f"Drive raw cache already exists for {name}; not overwriting. Set ALLOW_OVERWRITE_DRIVE_RAW_CACHE=True only after checking profiles.")
        return
    if target.exists():
        backup = RAW_CACHE_BACKUP_DIR / f"{name}_{now_token()}"
        print(f"Backing up existing Drive raw cache before overwrite: {backup}")
        shutil.move(str(target), str(backup))
    print(f"Caching dataset to Drive: {name}")
    shutil.copytree(source, target)


def download_roboflow_source(source: dict, rf) -> bool:
    if dataset_ready_for_source(source, REPO_ROOT / source["folder"]) and not FORCE_DOWNLOAD_DATASETS:
        return True
    destination = LOCAL_DOWNLOAD_ROOT / source["folder"]
    if destination.exists() and FORCE_DOWNLOAD_DATASETS:
        shutil.rmtree(destination)
    if dataset_ready_for_source(source, destination):
        return materialize_dataset(source, destination)
    destination.mkdir(parents=True, exist_ok=True)
    last_error = None
    project = rf.workspace(source["workspace"]).project(source["project"])
    version_candidates = []
    for candidate in [source.get("version"), *source.get("version_candidates", [])]:
        if candidate is not None and int(candidate) not in version_candidates:
            version_candidates.append(int(candidate))
    for version_number in version_candidates:
        try:
            version = project.version(version_number)
        except Exception as exc:
            last_error = exc
            print(f"  version {version_number} unavailable for {source['folder']}: {exc}")
            continue
        for export_format in ("yolov11", "yolov8"):
            try:
                print(f"Downloading {source['folder']} v{version_number} from Roboflow as {export_format} ...")
                version.download(export_format, location=str(destination), overwrite=True)
                last_error = None
                if materialize_dataset(source, destination):
                    return True
                print(f"  downloaded v{version_number}/{export_format}, but it did not pass source readiness checks.")
            except Exception as exc:
                last_error = exc
                print(f"  v{version_number}/{export_format} failed: {exc}")
    if last_error is not None:
        raise last_error
    return False


def download_kaggle_source(source: dict) -> bool:
    if dataset_ready_for_source(source, REPO_ROOT / source["folder"]) and not FORCE_DOWNLOAD_DATASETS:
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
    if dataset_ready_for_source(source, REPO_ROOT / source["folder"]) and not FORCE_DOWNLOAD_DATASETS:
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

RAW_SOURCE_PROFILES = {
    source["folder"]: {
        **dataset_profile(REPO_ROOT / source["folder"]),
        "expected_unified": source.get("expected_unified", []),
        "min_expected_matches": source.get("min_expected_matches", 1),
        "actual_unified": sorted(unified_labels_from_profile(dataset_profile(REPO_ROOT / source["folder"]))),
        "source_ready": dataset_ready_for_source(source, REPO_ROOT / source["folder"]),
    }
    for source in DATASET_SOURCES
}
profile_path = DRIVE_OUTPUT_ROOT / "summaries" / "raw_source_profiles_current.json"
profile_path.parent.mkdir(parents=True, exist_ok=True)
profile_path.write_text(json.dumps(RAW_SOURCE_PROFILES, indent=2), encoding="utf-8")
print(f"Raw source profile saved to: {profile_path}")
print(json.dumps(RAW_SOURCE_PROFILES, indent=2)[:5000])

missing = expected_missing()
DATASETS_AVAILABLE = not missing
AVAILABLE_BASELINE_DATASETS = [
    source["folder"]
    for source in DATASET_SOURCES
    if dataset_ready_for_source(source, REPO_ROOT / source["folder"])
]
BASELINE_DATASETS_BUILDABLE = DATASETS_AVAILABLE or bool(AVAILABLE_BASELINE_DATASETS)
STATE.setdefault("decisions", {})["baseline_datasets_available"] = DATASETS_AVAILABLE
STATE["decisions"]["missing_baseline_datasets"] = missing
STATE["decisions"]["available_baseline_datasets"] = AVAILABLE_BASELINE_DATASETS
STATE["decisions"]["baseline_datasets_buildable"] = BASELINE_DATASETS_BUILDABLE
save_state()

if missing and previous_training_available():
    print("Baseline datasets are incomplete, but previous Drive training artifacts are available. Recovery/training cells will reuse those weights.")
elif missing and BASELINE_DATASETS_BUILDABLE:
    print("WARNING: Some baseline raw datasets are missing, but the notebook will build a recoverable partial baseline from available datasets.")
    print(f"Missing baseline datasets: {missing}")
    print(f"Available baseline datasets: {AVAILABLE_BASELINE_DATASETS}")
elif missing:
    raise RuntimeError(
        "Missing baseline datasets and no usable raw subset was found. "
        f"Missing: {missing}. Upload raw_datasets.zip, restore Drive cache, or provide completed Drive weights."
    )
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
    bootstrapped(materialize_project_cell()),
    md(
        """
## Cell 4: Build Baseline Datasets When Needed

This preserves source train/valid/test splits and extracts severity crops. If datasets are missing but completed baseline weights already exist, this cell skips dataset building so the notebook can recover previous training instead of blocking.
"""
    ),
    bootstrapped(
        r'''
MERGED_DATASET = REPO_ROOT / "merged_dataset"
SEVERITY_DATASET = REPO_ROOT / "severity_dataset"
MERGE_CONFIG_FOR_RUN = REPO_ROOT / "configs" / "merge_config.yaml"
DATASET_CACHE_ROOT = DRIVE_OUTPUT_ROOT / "dataset_cache"
CACHED_MERGED_DATASET = DATASET_CACHE_ROOT / "merged_dataset"
CACHED_SEVERITY_DATASET = DATASET_CACHE_ROOT / "severity_dataset"
DATASET_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def detection_labels_from_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return summary.get("counts", {}).get("labels", {})


def dataset_has_all_detection_classes(dataset_root: Path) -> bool:
    labels = detection_labels_from_summary(Path(dataset_root) / "merge_summary.json")
    return all(labels.get(name, 0) > 0 for name in ["crack", "spalling", "corrosion", "pothole", "paint_degradation"])


def restore_cached_dataset(local_root: Path, cached_root: Path, label: str) -> bool:
    if (cached_root / "data.yaml").exists() and dataset_has_all_detection_classes(cached_root):
        if local_root.exists():
            backup = REPO_ROOT / "output" / f"rejected_{label}_{now_token()}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(local_root), str(backup))
            print(f"Moved invalid local {label} aside: {backup}")
        print(f"Restoring cached {label} from Drive: {cached_root}")
        shutil.copytree(cached_root, local_root, dirs_exist_ok=True)
        return True
    return False


def cache_good_dataset(local_root: Path, cached_root: Path, label: str) -> None:
    if not dataset_has_all_detection_classes(local_root):
        print(f"Not caching {label}; it does not contain all five detection classes.")
        return
    if cached_root.exists():
        shutil.rmtree(cached_root)
    cached_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_root, cached_root)
    print(f"Cached good {label} to Drive: {cached_root}")


def reject_bad_dataset(local_root: Path, label: str) -> None:
    if not local_root.exists():
        return
    backup = REPO_ROOT / "output" / f"rejected_{label}_{now_token()}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(local_root), str(backup))
    print(f"Moved invalid {label} aside so later cells cannot train on it: {backup}")

def write_available_merge_config() -> Path:
    import yaml

    source_config = REPO_ROOT / "configs" / "merge_config.yaml"
    target_config = REPO_ROOT / "configs" / "merge_config_available.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    available = set(globals().get("AVAILABLE_BASELINE_DATASETS", []))
    config["datasets"] = [item for item in config.get("datasets", []) if item.get("name") in available]
    config["recovery_note"] = {
        "reason": "Some raw source datasets were missing in this Colab session.",
        "missing_baseline_datasets": globals().get("missing", []),
        "available_baseline_datasets": sorted(available),
    }
    target_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Using available-datasets merge config: {target_config}")
    print(f"Included datasets: {[item['name'] for item in config['datasets']]}")
    return target_config


def merged_class_counts(summary_path: Path) -> dict:
    return detection_labels_from_summary(summary_path)


def print_merge_source_audit(summary_path: Path) -> None:
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read merge source audit: {exc}")
        return
    print("Merged label counts:", summary.get("counts", {}).get("labels", {}))
    for source in summary.get("source_audit", []):
        nonzero_labels = {key: value for key, value in source.get("mapped_labels", {}).items() if value}
        print(
            f"- {source.get('name')}: classes={source.get('class_names', [])}, "
            f"mapped_images={source.get('mapped_images', 0)}, mapped_labels={nonzero_labels}"
        )


if not DATASETS_AVAILABLE and BASELINE_DATASETS_BUILDABLE:
    MERGE_CONFIG_FOR_RUN = write_available_merge_config()
elif not DATASETS_AVAILABLE and previous_training_available():
    print("Skipping baseline dataset build because previous trained weights are recoverable and no raw baseline subset is buildable.")
elif not BASELINE_DATASETS_BUILDABLE:
    raise RuntimeError("Cannot build baseline datasets because no raw baseline datasets are available.")

if BASELINE_DATASETS_BUILDABLE:
    if not DATASETS_AVAILABLE:
        print("Proceeding with partial raw baseline rebuild. The merge summary below must still contain all five classes before new baseline training is trusted.")

    if dataset_has_all_detection_classes(MERGED_DATASET) and not FORCE_REBUILD_DATASETS:
        print("Existing merged_dataset already contains all five classes; reusing it.")
        mark_done("merge_detection_dataset", {"data_yaml": str(MERGED_DATASET / "data.yaml"), "reused_existing": True})
    elif restore_cached_dataset(MERGED_DATASET, CACHED_MERGED_DATASET, "merged_dataset") and not FORCE_REBUILD_DATASETS:
        mark_done("merge_detection_dataset", {"data_yaml": str(MERGED_DATASET / "data.yaml"), "restored_from_drive_cache": True})
    elif FORCE_REBUILD_DATASETS or not dataset_has_all_detection_classes(MERGED_DATASET):
        run_process([sys.executable, "scripts/merge_datasets.py", "--config", str(MERGE_CONFIG_FOR_RUN), "--preserve-splits", "--force"], log_name="merge_detection_dataset")
        print_merge_source_audit(MERGED_DATASET / "merge_summary.json")
        class_counts = merged_class_counts(MERGED_DATASET / "merge_summary.json")
        missing_classes = [name for name in ["crack", "spalling", "corrosion", "pothole", "paint_degradation"] if class_counts.get(name, 0) == 0]
        merge_ready = False
        if missing_classes:
            reject_bad_dataset(MERGED_DATASET, "merged_dataset")
            if restore_cached_dataset(MERGED_DATASET, CACHED_MERGED_DATASET, "merged_dataset"):
                mark_done("merge_detection_dataset", {"data_yaml": str(MERGED_DATASET / "data.yaml"), "restored_after_bad_merge": True})
                merge_ready = True
            elif previous_training_available():
                print(
                    "WARNING: Fresh baseline merge was rejected because it is missing classes "
                    f"{missing_classes}. Existing Drive weights will be reused, and baseline retraining will be skipped."
                )
            else:
                raise RuntimeError(
                    "Merged baseline dataset is missing classes after recovery and no cached all-class merged dataset or Drive weights were found. "
                    f"Missing classes: {missing_classes}. "
                    "This usually means the raw folders in this Colab session restored the wrong source or only one source."
                )
        else:
            cache_good_dataset(MERGED_DATASET, CACHED_MERGED_DATASET, "merged_dataset")
            merge_ready = True
        if merge_ready:
            mark_done("merge_detection_dataset", {"data_yaml": str(MERGED_DATASET / "data.yaml"), "config": str(MERGE_CONFIG_FOR_RUN)})
    else:
        print("Merged detection dataset already exists; skipping rebuild.")

    if (SEVERITY_DATASET / "data.yaml").exists() and not FORCE_REBUILD_DATASETS:
        print("Existing severity_dataset is present; reusing it instead of replacing it with a partial extraction.")
        mark_done("extract_severity_crops", {"data_yaml": str(SEVERITY_DATASET / "data.yaml"), "reused_existing": True})
    elif (CACHED_SEVERITY_DATASET / "data.yaml").exists() and not FORCE_REBUILD_DATASETS:
        print(f"Restoring cached severity_dataset from Drive: {CACHED_SEVERITY_DATASET}")
        shutil.copytree(CACHED_SEVERITY_DATASET, SEVERITY_DATASET, dirs_exist_ok=True)
        mark_done("extract_severity_crops", {"data_yaml": str(SEVERITY_DATASET / "data.yaml"), "restored_from_drive_cache": True})
    elif FORCE_REBUILD_DATASETS or not step_done("extract_severity_crops", [SEVERITY_DATASET / "data.yaml"]):
        if (MERGED_DATASET / "data.yaml").exists():
            run_process([sys.executable, "scripts/extract_severity_crops.py", "--config", str(MERGE_CONFIG_FOR_RUN), "--output", str(SEVERITY_DATASET), "--force"], log_name="extract_severity_crops")
            if CACHED_SEVERITY_DATASET.exists():
                shutil.rmtree(CACHED_SEVERITY_DATASET)
            shutil.copytree(SEVERITY_DATASET, CACHED_SEVERITY_DATASET)
            mark_done("extract_severity_crops", {"data_yaml": str(SEVERITY_DATASET / "data.yaml"), "config": str(MERGE_CONFIG_FOR_RUN)})
        else:
            print("Skipping severity extraction because no valid merged baseline dataset is available.")
    else:
        print("Severity crop dataset already exists; skipping extraction.")

    for summary in [MERGED_DATASET / "merge_summary.json", SEVERITY_DATASET / "severity_summary.json"]:
        if summary.exists():
            mirror_path(summary, DRIVE_OUTPUT_ROOT / "summaries" / summary.name)
            print(summary.read_text(encoding="utf-8")[:2000])
else:
    print("No baseline dataset was built in this session.")

MERGED_DATASET_READY_FOR_TRAINING = dataset_has_all_detection_classes(MERGED_DATASET)
if not MERGED_DATASET_READY_FOR_TRAINING:
    print("WARNING: No all-class merged_dataset is available in this session. Baseline/balanced training will rely on recoverable weights or be skipped.")
'''
    ),
    md(
        """
## Cell 5: Recover or Train Baseline Models

This is the important recovery stage. The notebook first checks canonical Drive weights, then searches Drive run folders for completed `best.pt`, then resumes `last.pt` if data exists. It trains only when no previous usable artifact exists.
"""
    ),
    bootstrapped(
        r'''
from ultralytics import YOLO
try:
    import torch
except Exception:
    torch = None


def train_device_kwargs() -> dict:
    return {"device": 0} if torch is not None and torch.cuda.is_available() else {}


def recover_completed_weight(kind: str, final_weight: Path, local_name: str, allow_kind_search: bool = True) -> Path | None:
    if final_weight.exists() and not FORCE_INITIAL_TRAINING:
        print(f"Using existing canonical {kind} weight: {final_weight}")
        copy_weight_to_local(final_weight, local_name)
        return final_weight
    if not allow_kind_search:
        return None
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


def train_or_resume(
    kind: str,
    seed_weights: str | Path,
    data_path: Path,
    run_name: str,
    final_weight: Path,
    local_name: str,
    train_args: dict,
    force: bool = False,
    allow_kind_search: bool = True,
    allow_new_training: bool = False,
    allow_resume_training: bool = False,
) -> Path:
    recovered = recover_completed_weight(kind, final_weight, local_name, allow_kind_search)
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
        if not allow_resume_training:
            raise RuntimeError(
                f"Interrupted {kind} checkpoint exists at {last}, but SAFE_RECOVERY_MODE is on. "
                "Set ALLOW_TRAINING_RESUME=True for this run if you intentionally want to spend compute resuming it."
            )
        print(f"Resuming interrupted {kind} training from: {last}")
        YOLO(str(last)).train(resume=True)
    else:
        if not (force or allow_new_training):
            raise RuntimeError(
                f"No completed {kind} weight was recovered, and fresh training is disabled in SAFE_RECOVERY_MODE. "
                "Use existing Drive weights or set the matching ALLOW_*_TRAINING flag to True."
            )
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

trained_detector = train_or_resume(
    "detector",
    "yolo11s.pt",
    MERGED_DATASET / "data.yaml",
    "stage1_defect_detector",
    DETECTOR_WEIGHT,
    "defect_detector.pt",
    detector_args,
    FORCE_INITIAL_TRAINING,
    allow_new_training=ALLOW_BASELINE_TRAINING or FORCE_INITIAL_TRAINING,
    allow_resume_training=ALLOW_TRAINING_RESUME or FORCE_INITIAL_TRAINING,
)
trained_severity = train_or_resume(
    "severity",
    "yolo11n-cls.pt",
    SEVERITY_DATASET,
    "stage2_severity",
    SEVERITY_WEIGHT,
    "severity_cls.pt",
    severity_args,
    FORCE_INITIAL_TRAINING,
    allow_new_training=ALLOW_BASELINE_TRAINING or FORCE_INITIAL_TRAINING,
    allow_resume_training=ALLOW_TRAINING_RESUME or FORCE_INITIAL_TRAINING,
)
mark_done("baseline_model_recovery_or_training", {"detector": str(trained_detector), "severity": str(trained_severity)})
'''
    ),
    md(
        """
## Cell 6: Domain Sweep With Explicit Image Sources

This cell prints the exact sweep images from `configs/domain_sweep_manifest.csv`. It reuses cached sweep results only when the manifest fingerprint matches, so old local-path sweeps are not silently reused.
"""
    ),
    bootstrapped(
        r'''
BASELINE_SWEEP_DIR = REPO_ROOT / "output" / "domain_sweep"
DRIVE_BASELINE_SWEEP_DIR = DRIVE_OUTPUT_ROOT / "domain_sweep" / "baseline"
DRIVE_SWEEP_IMAGE_CACHE = DRIVE_OUTPUT_ROOT / "domain_sweep_image_cache"
MANIFEST_PATH = REPO_ROOT / "configs" / "domain_sweep_manifest.csv"
DRIVE_SWEEP_IMAGE_CACHE.mkdir(parents=True, exist_ok=True)


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


def seed_sweep_image_cache(output_dir: Path) -> None:
    """Reuse already downloaded sweep images so baseline/adapted comparisons test the same files."""
    output_images = Path(output_dir) / "images"
    source_dirs = [DRIVE_SWEEP_IMAGE_CACHE, BASELINE_SWEEP_DIR / "images", DRIVE_BASELINE_SWEEP_DIR / "images"]
    copied = 0
    for source_dir in source_dirs:
        if not source_dir.exists() or source_dir.resolve() == output_images.resolve():
            continue
        output_images.mkdir(parents=True, exist_ok=True)
        for image_path in source_dir.iterdir():
            if image_path.is_file():
                target = output_images / image_path.name
                if not target.exists():
                    shutil.copy2(image_path, target)
                    copied += 1
    if copied:
        print(f"Seeded adapted sweep image cache with {copied} previously downloaded images.")


def refresh_sweep_image_cache(output_dir: Path) -> None:
    image_dir = Path(output_dir) / "images"
    if not image_dir.exists():
        return
    copied = 0
    for image_path in image_dir.iterdir():
        if image_path.is_file():
            target = DRIVE_SWEEP_IMAGE_CACHE / image_path.name
            if not target.exists() or target.stat().st_size == 0:
                shutil.copy2(image_path, target)
                copied += 1
    if copied:
        print(f"Cached {copied} sweep images to Drive.")


def reuse_or_run_sweep(output_dir: Path, drive_dir: Path, detector_weight: Path, name: str, force: bool = False) -> Path | None:
    print_sweep_sources(MANIFEST_PATH)
    if sweep_ready(output_dir, MANIFEST_PATH) and not force:
        print(f"Reusing local sweep with matching manifest: {output_dir}")
        mirror_path(output_dir, drive_dir)
        return output_dir
    if sweep_ready(drive_dir, MANIFEST_PATH) and not force:
        print(f"Restoring matching sweep from Drive: {drive_dir}")
        mirror_path(drive_dir, output_dir)
        return output_dir
    if not (force or ALLOW_UNCACHED_SWEEP_RUNS):
        print(
            "No cached sweep with the current manifest was found. "
            "Skipping uncached sweep in SAFE_RECOVERY_MODE; set ALLOW_UNCACHED_SWEEP_RUNS=True to run inference checks."
        )
        return None
    if Path(output_dir).resolve() != BASELINE_SWEEP_DIR.resolve():
        seed_sweep_image_cache(output_dir)
    run_process([
        sys.executable, "scripts/domain_sweep.py",
        "--manifest", str(MANIFEST_PATH),
        "--detector", str(detector_weight),
        "--severity", str(SEVERITY_WEIGHT),
        "--output", str(output_dir),
        "--image-cache", str(DRIVE_SWEEP_IMAGE_CACHE),
        "--thresholds", "0.45,0.30,0.20,0.10",
        "--iou", "0.45",
        "--annotate-conf", "0.20",
    ], log_name=name)
    refresh_sweep_image_cache(output_dir)
    write_sweep_fingerprint(output_dir, MANIFEST_PATH)
    mirror_path(output_dir, drive_dir)
    return output_dir


baseline_sweep = reuse_or_run_sweep(BASELINE_SWEEP_DIR, DRIVE_BASELINE_SWEEP_DIR, DETECTOR_WEIGHT, "baseline_domain_sweep", FORCE_SWEEP)
if baseline_sweep is not None and (baseline_sweep / "summary.json").exists():
    print((baseline_sweep / "summary.json").read_text(encoding="utf-8"))
    mark_done("baseline_domain_sweep", {"path": str(baseline_sweep), "manifest_sha256": file_hash(MANIFEST_PATH)})
else:
    print("Baseline sweep not available in this recovery run.")
'''
    ),
    md(
        """
## Cell 7: Class-Balanced Target Recollection and Candidate Fine-Tuning

This stage actively repairs the dataset before fine-tuning: it collects targeted crack, pothole, corrosion, facade, and spalling sources, preserves the full baseline dataset, caps dominant target classes, oversamples weak classes, reserves target validation examples when needed, and refuses to train only after these repair steps still cannot preserve all five classes.
"""
    ),
    bootstrapped(
        r'''
TARGET_YOLO = REPO_ROOT / "domain_adaptation" / "target_yolo"
AUTO_RAW = REPO_ROOT / "domain_adaptation" / "auto_raw"
TARGET_SUMMARY = TARGET_YOLO / "auto_collection_summary.json"
DOMAIN_ADAPTED_DATASET = REPO_ROOT / "merged_dataset_domain_balanced"
HARD_NEGATIVES = REPO_ROOT / "domain_adaptation" / "hard_negatives"
REQUIRED_DETECTION_CLASSES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]


def count_images(root: Path) -> int:
    return sum(1 for path in Path(root).glob("**/*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}) if Path(root).exists() else 0


def target_label_count(target_root: Path) -> int:
    total = 0
    for label_path in target_root.glob("**/*.txt"):
        total += len([line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()])
    return total


def detection_class_counts(dataset_root: Path) -> dict[str, dict[str, int]]:
    counts = {split: {name: 0 for name in REQUIRED_DETECTION_CLASSES} for split in ("train", "val", "test")}
    split_aliases = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    for source_split, output_split in split_aliases.items():
        for labels_dir in (dataset_root / source_split / "labels", dataset_root / "labels" / source_split):
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
                        counts[output_split][REQUIRED_DETECTION_CLASSES[class_id]] += 1
    return counts


def missing_target_classes(target_root: Path) -> list[str]:
    counts = detection_class_counts(target_root)
    return [name for name in REQUIRED_DETECTION_CLASSES if counts["train"].get(name, 0) == 0]


def collect_or_refresh_target_domain(reason: str, force_download: bool = False) -> None:
    if not ROBOFLOW_API_KEY:
        print(f"ROBOFLOW_API_KEY is missing; target-domain recollection skipped. Reason: {reason}")
    else:
        print(f"Refreshing target-domain collection: {reason}")
        run_process([
            sys.executable, "scripts/auto_collect_domain_sources.py",
            "--api-key", ROBOFLOW_API_KEY,
            "--raw-output", str(AUTO_RAW),
            "--target-output", str(TARGET_YOLO),
        ] + (["--force-download"] if force_download else []) + ["--force-rebuild"], log_name="auto_collect_domain_sources")


if FORCE_DOMAIN_COLLECTION or not TARGET_SUMMARY.exists() or count_images(TARGET_YOLO) == 0:
    collect_or_refresh_target_domain("missing or forced target-domain cache", FORCE_DOMAIN_COLLECTION)
else:
    print(f"Using existing target-domain collection: {TARGET_YOLO}")

target_missing_before_repair = missing_target_classes(TARGET_YOLO)
if target_missing_before_repair:
    collect_or_refresh_target_domain(f"target-domain training set missing {target_missing_before_repair}", False)

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
target_class_counts = detection_class_counts(TARGET_YOLO)
target_missing_after_repair = missing_target_classes(TARGET_YOLO)
BASELINE_TRAINING_DATA_READY = bool(globals().get("MERGED_DATASET_READY_FOR_TRAINING", False)) and (MERGED_DATASET / "data.yaml").exists()
TARGET_TRAINING_DATA_READY = (
    target_images >= 10
    and target_labels >= 10
    and not target_missing_after_repair
    and (TARGET_YOLO / "data.yaml").exists()
)
CAN_BUILD_BALANCED_DATASET = BASELINE_TRAINING_DATA_READY or TARGET_TRAINING_DATA_READY
NEEDS_DOMAIN_RETRAIN = FORCE_DOMAIN_RETRAIN or (
    CAN_BUILD_BALANCED_DATASET
    and (weak_rows > 0 or not BALANCED_DETECTOR_WEIGHT.exists())
)
STATE.setdefault("decisions", {})["needs_domain_retrain"] = NEEDS_DOMAIN_RETRAIN
STATE["decisions"]["domain_retrain_reason"] = {
    "datasets_available": DATASETS_AVAILABLE,
    "baseline_training_data_ready": BASELINE_TRAINING_DATA_READY,
    "target_training_data_ready": TARGET_TRAINING_DATA_READY,
    "can_build_balanced_dataset": CAN_BUILD_BALANCED_DATASET,
    "weak_sweep_rows": weak_rows,
    "target_images": target_images,
    "target_labels": target_labels,
    "target_class_counts": target_class_counts,
    "target_missing_before_repair": target_missing_before_repair,
    "target_missing_after_repair": target_missing_after_repair,
    "force": FORCE_DOMAIN_RETRAIN,
    "allow_balanced_retrain": ALLOW_BALANCED_RETRAIN,
    "safe_recovery_mode": SAFE_RECOVERY_MODE,
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


TRAIN_BALANCED_NOW = FORCE_DOMAIN_RETRAIN or ALLOW_BALANCED_RETRAIN

if NEEDS_DOMAIN_RETRAIN:
    copied_negatives = create_hard_negatives_from_sweep(BASELINE_SWEEP_DIR / "resolved_manifest.csv", HARD_NEGATIVES)
    print(f"Hard-negative sweep images copied: {copied_negatives}")
    balanced_summary = DOMAIN_ADAPTED_DATASET / "class_balanced_summary.json"
    balance_base = MERGED_DATASET if BASELINE_TRAINING_DATA_READY else (REPO_ROOT / "domain_adaptation" / "_no_valid_baseline_dataset")
    balanced_builder_args = [
        sys.executable, "scripts/build_balanced_detection_dataset.py",
        "--base", str(balance_base),
        "--target", str(TARGET_YOLO),
        "--hard-negatives", str(HARD_NEGATIVES),
        "--output", str(DOMAIN_ADAPTED_DATASET),
        "--base-box-goal-per-class", "4500",
        "--max-base-images-per-class", "2400",
        "--target-box-goal-per-class", "2400",
        "--max-target-images-per-class", "1600",
        "--balance-goal-boxes", "2600",
        "--max-repeat-per-image", "5",
        "--val-reserve-images-per-class", "40",
        "--max-hard-negative-train", "350",
        "--max-hard-negative-val", "80",
        "--min-train-boxes-per-class", "1",
        "--min-val-boxes-per-class", "1",
        "--max-class-ratio", "18",
    ]
    if not BASELINE_TRAINING_DATA_READY:
        print("No valid all-class baseline merged dataset is available; building balanced candidate from target-domain data only.")
        balanced_builder_args.append("--allow-target-only-base")
    if FORCE_REBUILD_DATASETS:
        balanced_builder_args.append("--force")
    run_process(balanced_builder_args, log_name="build_class_balanced_detection_dataset")
    if balanced_summary.exists():
        mirror_path(balanced_summary, DRIVE_OUTPUT_ROOT / "summaries" / "class_balanced_detection_summary.json")
        print(balanced_summary.read_text(encoding="utf-8")[:4000])

    if TRAIN_BALANCED_NOW:
        seed_weight = DETECTOR_WEIGHT if (BALANCED_DETECTOR_WARM_START_FROM_CURRENT and DETECTOR_WEIGHT.exists()) else BALANCED_DETECTOR_MODEL_SEED
        balanced_run_name = f"stage1_balanced_{Path(str(seed_weight)).stem}_candidate_detector"
        domain_args = {
            "epochs": 90, "imgsz": 640, "batch": BALANCED_DETECTOR_BATCH, "patience": 20, "optimizer": "AdamW", "lr0": 0.0007,
            "cos_lr": True, "close_mosaic": 10, "mosaic": 0.35, "mixup": 0.02,
            "copy_paste": 0.04, "degrees": 2.0, "translate": 0.06, "scale": 0.25,
            "fliplr": 0.50, "hsv_s": 0.30, "hsv_v": 0.28,
        }
        trained_domain_detector = train_or_resume(
            "detector",
            seed_weight,
            DOMAIN_ADAPTED_DATASET / "data.yaml",
            balanced_run_name,
            BALANCED_DETECTOR_WEIGHT,
            "defect_detector_balanced_candidate.pt",
            domain_args,
            FORCE_DOMAIN_RETRAIN,
            allow_kind_search=False,
            allow_new_training=TRAIN_BALANCED_NOW,
            allow_resume_training=ALLOW_TRAINING_RESUME or FORCE_DOMAIN_RETRAIN,
        )
        if PROMOTE_DOMAIN_ADAPTED_MODEL and not PROMOTE_ONLY_AFTER_PRODUCTION_CHECK:
            shutil.copy2(trained_domain_detector, DETECTOR_WEIGHT)
            shutil.copy2(trained_domain_detector, DOMAIN_DETECTOR_WEIGHT)
            copy_weight_to_local(DETECTOR_WEIGHT, "defect_detector.pt")
        mark_done("balanced_detector_training", {"detector": str(trained_domain_detector), "promoted_immediately": PROMOTE_DOMAIN_ADAPTED_MODEL and not PROMOTE_ONLY_AFTER_PRODUCTION_CHECK})
    elif BALANCED_DETECTOR_WEIGHT.exists():
        print(f"Balanced candidate already exists and will be reused: {BALANCED_DETECTOR_WEIGHT}")
        copy_weight_to_local(BALANCED_DETECTOR_WEIGHT, "defect_detector_balanced_candidate.pt")
    else:
        print(
            "Balanced retraining is recommended by the sweep, but GPU training is disabled in SAFE_RECOVERY_MODE. "
            "The balanced dataset was prepared/recovered; set ALLOW_BALANCED_RETRAIN=True only when you intentionally want to spend compute."
        )
else:
    print("Balanced detector retraining skipped by decision gate.")


# Stage 2 severity adaptation is optional but kept in the same 8-section workflow.
# It uses raw target-domain downloads when they contain detailed labels that can be mapped into minor/moderate/critical.
import yaml

SEVERITY_EXTRA = REPO_ROOT / "domain_adaptation" / "severity_extra"
SEVERITY_ADAPTED_DATASET = REPO_ROOT / "severity_dataset_domain_adapted"


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
    run_process([
        sys.executable,
        "scripts/extract_severity_crops.py",
        "--config",
        str(severity_config),
        "--output",
        str(SEVERITY_EXTRA),
        "--force",
    ], log_name="extract_target_severity_crops")
elif severity_config is None:
    print("No raw target-domain folders were available for automatic severity crop extraction.")

def class_image_counts(root: Path) -> dict[str, dict[str, int]]:
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for severity in ("minor", "moderate", "critical"):
            counts[split][severity] = count_images(Path(root) / split / severity)
    return counts


def missing_required_severity_classes(counts: dict[str, dict[str, int]]) -> list[str]:
    missing = []
    for split in ("train", "val"):
        for severity in ("minor", "moderate", "critical"):
            if counts.get(split, {}).get(severity, 0) == 0:
                missing.append(f"{split}/{severity}")
    return missing


extra_crops = count_images(SEVERITY_EXTRA)
severity_candidate = FORCE_SEVERITY_RETRAIN or (extra_crops >= 10 and NEEDS_DOMAIN_RETRAIN)
if severity_candidate and (FORCE_REBUILD_DATASETS or not (SEVERITY_ADAPTED_DATASET / "severity_adaptation_summary.json").exists()):
    if FORCE_REBUILD_DATASETS or not (SEVERITY_ADAPTED_DATASET / "severity_adaptation_summary.json").exists():
        run_process([
            sys.executable,
            "scripts/build_severity_adaptation_dataset.py",
            "--base",
            str(SEVERITY_DATASET),
            "--target",
            str(SEVERITY_EXTRA),
            "--output",
            str(SEVERITY_ADAPTED_DATASET),
            "--target-repeat",
            "3",
        ], log_name="build_severity_adapted_dataset")

severity_counts = class_image_counts(SEVERITY_ADAPTED_DATASET)
missing_severity = missing_required_severity_classes(severity_counts)
NEEDS_SEVERITY_RETRAIN = severity_candidate and not missing_severity
STATE.setdefault("decisions", {})["needs_severity_retrain"] = NEEDS_SEVERITY_RETRAIN
STATE["decisions"]["severity_retrain_reason"] = {
    "extra_crops": extra_crops,
    "force": FORCE_SEVERITY_RETRAIN,
    "allow_severity_retrain": ALLOW_SEVERITY_RETRAIN,
    "safe_recovery_mode": SAFE_RECOVERY_MODE,
    "class_counts": severity_counts,
    "missing_required_classes": missing_severity,
}
save_state()

if severity_candidate and missing_severity:
    print("Severity retraining skipped because the adapted severity dataset is missing required classes:")
    for item in missing_severity:
        print(f"- {item}")
    print(json.dumps(severity_counts, indent=2))

TRAIN_SEVERITY_NOW = FORCE_SEVERITY_RETRAIN or ALLOW_SEVERITY_RETRAIN

if NEEDS_SEVERITY_RETRAIN and TRAIN_SEVERITY_NOW:
    severity_adapt_args = {
        "epochs": 40,
        "imgsz": 224,
        "patience": 10,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "cos_lr": True,
        "dropout": 0.20,
    }
    trained_domain_severity = train_or_resume(
        "severity",
        SEVERITY_WEIGHT if SEVERITY_WEIGHT.exists() else "yolo11n-cls.pt",
        SEVERITY_ADAPTED_DATASET,
        "stage2_domain_adapted_severity",
        DOMAIN_SEVERITY_WEIGHT,
        "severity_cls_domain_adapted.pt",
        severity_adapt_args,
        FORCE_SEVERITY_RETRAIN,
        allow_kind_search=False,
        allow_new_training=TRAIN_SEVERITY_NOW,
        allow_resume_training=ALLOW_TRAINING_RESUME or FORCE_SEVERITY_RETRAIN,
    )
    shutil.copy2(trained_domain_severity, SEVERITY_WEIGHT)
    copy_weight_to_local(SEVERITY_WEIGHT, "severity_cls.pt")
    mark_done("domain_severity_training", {"severity": str(trained_domain_severity)})
elif NEEDS_SEVERITY_RETRAIN:
    print(
        "Severity retraining is recommended, but GPU training is disabled in SAFE_RECOVERY_MODE. "
        "Set ALLOW_SEVERITY_RETRAIN=True only when you intentionally want to spend compute."
    )
else:
    print("Severity retraining skipped by decision gate.")
'''
    ),
    md(
        """
## Cell 8: Post-Adaptation Check and Deployment Weights

This final section runs a fixed production-style check against the current detector and the balanced candidate. The candidate is only promoted to the canonical deployment weight when it improves the check and preserves critical weak classes.
"""
    ),
    bootstrapped(
        r'''
POST_SWEEP_DIR = REPO_ROOT / "output" / "domain_sweep_domain_adapted"
DRIVE_POST_SWEEP_DIR = DRIVE_OUTPUT_ROOT / "domain_sweep" / "domain_adapted"
PRODUCTION_MANIFEST_PATH = REPO_ROOT / "configs" / "production_eval_manifest.csv"
CURRENT_PRODUCTION_DIR = REPO_ROOT / "output" / "production_eval_current"
CANDIDATE_PRODUCTION_DIR = REPO_ROOT / "output" / "production_eval_balanced_candidate"
DRIVE_CURRENT_PRODUCTION_DIR = DRIVE_OUTPUT_ROOT / "production_eval" / "current"
DRIVE_CANDIDATE_PRODUCTION_DIR = DRIVE_OUTPUT_ROOT / "production_eval" / "balanced_candidate"


def summarize_matches(summary_csv: Path, threshold: str = "0.2") -> dict:
    rows = load_sweep_rows(summary_csv)
    selected = [row for row in rows if str(row.get("threshold")) in {threshold, f"{float(threshold):.1f}"}]
    return {
        "csv_rows_total": len(rows),
        "rows": len(selected),
        "matches": sum(1 for row in selected if row.get("match") == "true"),
        "failures": sum(1 for row in selected if row.get("match") == "false"),
        "skipped": sum(1 for row in selected if row.get("match") == "skipped"),
        "no_detection": sum(1 for row in selected if row.get("detections") == "0"),
        "download_or_source_failures": sum(1 for row in rows if row.get("match") == "skipped"),
    }


def manifest_sweep_ready(path: Path, manifest_path: Path) -> bool:
    path = Path(path)
    fingerprint = sweep_fingerprint_path(path)
    return (
        (path / "domain_sweep_summary.csv").exists()
        and (path / "summary.json").exists()
        and fingerprint.exists()
        and fingerprint.read_text(encoding="utf-8").strip() == file_hash(manifest_path)
    )


def run_manifest_sweep(output_dir: Path, drive_dir: Path, detector_weight: Path, manifest_path: Path, log_name: str, force: bool = False) -> Path | None:
    if manifest_sweep_ready(output_dir, manifest_path) and not force:
        print(f"Reusing production check with matching manifest: {output_dir}")
        mirror_path(output_dir, drive_dir)
        return output_dir
    if manifest_sweep_ready(drive_dir, manifest_path) and not force:
        print(f"Restoring production check from Drive: {drive_dir}")
        mirror_path(drive_dir, output_dir)
        return output_dir
    if not (force or ALLOW_UNCACHED_SWEEP_RUNS):
        print(
            f"No cached production check found for {log_name}. "
            "Skipping uncached inference sweep in SAFE_RECOVERY_MODE; set ALLOW_UNCACHED_SWEEP_RUNS=True to run it."
        )
        return None
    seed_sweep_image_cache(output_dir)
    run_process([
        sys.executable, "scripts/domain_sweep.py",
        "--manifest", str(manifest_path),
        "--detector", str(detector_weight),
        "--severity", str(SEVERITY_WEIGHT),
        "--output", str(output_dir),
        "--image-cache", str(DRIVE_SWEEP_IMAGE_CACHE),
        "--thresholds", "0.45,0.30,0.20,0.10",
        "--iou", "0.45",
        "--annotate-conf", "0.20",
    ], log_name=log_name)
    refresh_sweep_image_cache(output_dir)
    write_sweep_fingerprint(output_dir, manifest_path)
    mirror_path(output_dir, drive_dir)
    return output_dir


def production_stats(summary_csv: Path, threshold: str = "0.2") -> dict:
    rows = load_sweep_rows(summary_csv)
    selected = [row for row in rows if str(row.get("threshold")) in {threshold, f"{float(threshold):.1f}"}]
    expected_class_matches = {name: 0 for name in REQUIRED_DETECTION_CLASSES}
    expected_class_total = {name: 0 for name in REQUIRED_DETECTION_CLASSES}
    false_positive_negatives = 0
    for row in selected:
        expected = row.get("expected_class", "")
        if expected == "none":
            if str(row.get("detections")) != "0":
                false_positive_negatives += 1
            continue
        allowed = [item.strip() for item in expected.split("|") if item.strip()]
        for class_name in allowed:
            if class_name in expected_class_total:
                expected_class_total[class_name] += 1
                if row.get("match") == "true":
                    expected_class_matches[class_name] += 1
    return {
        "csv_rows_total": len(rows),
        "rows_at_threshold": len(selected),
        "matches": sum(1 for row in selected if row.get("match") == "true"),
        "failures": sum(1 for row in selected if row.get("match") == "false"),
        "skipped": sum(1 for row in rows if row.get("match") == "skipped"),
        "no_detection": sum(1 for row in selected if row.get("detections") == "0"),
        "false_positive_negatives": false_positive_negatives,
        "expected_class_total": expected_class_total,
        "expected_class_matches": expected_class_matches,
        "score": (
            2 * sum(1 for row in selected if row.get("match") == "true")
            - 2 * sum(1 for row in selected if row.get("match") == "false")
            - sum(1 for row in selected if row.get("detections") == "0")
            - 2 * false_positive_negatives
            - sum(1 for row in rows if row.get("match") == "skipped")
        ),
    }


print("Drive run roots checked for recovery:")
for root in candidate_run_roots():
    print(f"- {root}")

recover_weight_from_completed_run(
    ("stage1_balanced", "candidate"),
    BALANCED_DETECTOR_WEIGHT,
    "defect_detector_balanced_candidate.pt",
)
recover_weight_from_completed_run(
    ("stage1_defect_detector",),
    DETECTOR_WEIGHT,
    "defect_detector.pt",
)
recover_weight_from_completed_run(
    ("stage2_severity",),
    SEVERITY_WEIGHT,
    "severity_cls.pt",
)


KNOWN_PUBLIC_WEIGHT_FILE_IDS = {
    "defect_detector_balanced_candidate.pt": "1IwxcNvdoXpUlPbEzuDq951naUmgI5WEz",
    "defect_detector.pt": "1zWMYJFG3Eon2fHpAjAzpL2Nuryk4iUKO",
    "severity_cls.pt": "1SPCbD7cqrc2_Dgkj7pm22xZkOWATrsQ_",
    "stage1_balanced_best.pt": "1TTYMb0Vd_hj5AY4707GmKmy_laJKZXGC",
}


def recover_public_drive_file(file_id: str, target_weight: Path, local_name: str) -> Path | None:
    if target_weight.exists():
        copy_weight_to_local(target_weight, local_name)
        return target_weight
    try:
        import gdown
    except Exception:
        run_process([sys.executable, "-m", "pip", "install", "-q", "gdown"], log_name="pip_gdown")
        import gdown
    target_weight.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_weight.with_name(f"{target_weight.name}.download")
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"Recovering {target_weight.name} from shared Drive file id: {file_id}")
    result = gdown.download(id=file_id, output=str(tmp_path), quiet=False)
    if result and tmp_path.exists() and tmp_path.stat().st_size > 1024 * 1024:
        shutil.move(str(tmp_path), str(target_weight))
        copy_weight_to_local(target_weight, local_name)
        return target_weight
    if tmp_path.exists():
        tmp_path.unlink()
    print(f"Shared Drive file recovery did not produce a valid checkpoint for {target_weight.name}.")
    return None


recover_public_drive_file(
    KNOWN_PUBLIC_WEIGHT_FILE_IDS["defect_detector_balanced_candidate.pt"],
    BALANCED_DETECTOR_WEIGHT,
    "defect_detector_balanced_candidate.pt",
)
recover_public_drive_file(
    KNOWN_PUBLIC_WEIGHT_FILE_IDS["defect_detector.pt"],
    DETECTOR_WEIGHT,
    "defect_detector.pt",
)
recover_public_drive_file(
    KNOWN_PUBLIC_WEIGHT_FILE_IDS["severity_cls.pt"],
    SEVERITY_WEIGHT,
    "severity_cls.pt",
)
if not DETECTOR_WEIGHT.exists() and DOMAIN_DETECTOR_WEIGHT.exists():
    shutil.copy2(DOMAIN_DETECTOR_WEIGHT, DETECTOR_WEIGHT)
    print(f"Recovered current detector from domain detector weight: {DOMAIN_DETECTOR_WEIGHT}")
    copy_weight_to_local(DETECTOR_WEIGHT, "defect_detector.pt")
if not DETECTOR_WEIGHT.exists() and BALANCED_DETECTOR_WEIGHT.exists():
    shutil.copy2(BALANCED_DETECTOR_WEIGHT, DETECTOR_WEIGHT)
    print("No separate current detector was found; using balanced candidate as canonical detector for deployment recovery.")
    copy_weight_to_local(DETECTOR_WEIGHT, "defect_detector.pt")

production_results = {}
if DETECTOR_WEIGHT.exists():
    current_eval = run_manifest_sweep(
        CURRENT_PRODUCTION_DIR,
        DRIVE_CURRENT_PRODUCTION_DIR,
        DETECTOR_WEIGHT,
        PRODUCTION_MANIFEST_PATH,
        "production_eval_current",
        FORCE_SWEEP or FORCE_PRODUCTION_CHECK,
    )
    if current_eval is not None:
        production_results["current"] = production_stats(current_eval / "domain_sweep_summary.csv", "0.2")
else:
    print(f"Current detector weight missing: {DETECTOR_WEIGHT}")

if BALANCED_DETECTOR_WEIGHT.exists():
    candidate_eval = run_manifest_sweep(
        CANDIDATE_PRODUCTION_DIR,
        DRIVE_CANDIDATE_PRODUCTION_DIR,
        BALANCED_DETECTOR_WEIGHT,
        PRODUCTION_MANIFEST_PATH,
        "production_eval_balanced_candidate",
        FORCE_SWEEP or FORCE_PRODUCTION_CHECK,
    )
    if candidate_eval is not None:
        production_results["balanced_candidate"] = production_stats(candidate_eval / "domain_sweep_summary.csv", "0.2")
else:
    print(f"Balanced candidate detector not available yet: {BALANCED_DETECTOR_WEIGHT}")

promotion_decision = {
    "promoted": False,
    "reason": "candidate_missing_or_not_checked",
    "results": production_results,
}
if not production_results and (DETECTOR_WEIGHT.exists() or BALANCED_DETECTOR_WEIGHT.exists()):
    promotion_decision["reason"] = "weights_recovered_but_production_checks_skipped_or_uncached"
if "current" in production_results and "balanced_candidate" in production_results:
    current = production_results["current"]
    candidate = production_results["balanced_candidate"]
    critical_classes_ok = (
        candidate["expected_class_matches"].get("crack", 0) >= 1
        and candidate["expected_class_matches"].get("pothole", 0) >= 1
        and candidate["expected_class_matches"].get("spalling", 0) >= 1
    )
    candidate_better = (
        candidate["rows_at_threshold"] >= current["rows_at_threshold"]
        and candidate["score"] > current["score"]
        and candidate["matches"] >= current["matches"]
        and candidate["false_positive_negatives"] <= current["false_positive_negatives"]
    )
    if candidate_better and critical_classes_ok:
        shutil.copy2(BALANCED_DETECTOR_WEIGHT, DETECTOR_WEIGHT)
        shutil.copy2(BALANCED_DETECTOR_WEIGHT, DOMAIN_DETECTOR_WEIGHT)
        copy_weight_to_local(DETECTOR_WEIGHT, "defect_detector.pt")
        copy_weight_to_local(DOMAIN_DETECTOR_WEIGHT, "defect_detector_domain_adapted.pt")
        promotion_decision.update({"promoted": True, "reason": "balanced_candidate_improved_production_check"})
        mark_done("balanced_detector_promoted", {"detector": str(BALANCED_DETECTOR_WEIGHT), "production_results": production_results})
    else:
        promotion_decision.update({
            "promoted": False,
            "reason": "candidate_not_better_or_critical_classes_not_recovered",
            "candidate_better": candidate_better,
            "critical_classes_ok": critical_classes_ok,
        })

comparison_path = DRIVE_OUTPUT_ROOT / "summaries" / "production_model_check_summary.json"
comparison_path.parent.mkdir(parents=True, exist_ok=True)
comparison_path.write_text(json.dumps(promotion_decision, indent=2), encoding="utf-8")
print(json.dumps(promotion_decision, indent=2))

for drive_weight, local_name in [
    (DETECTOR_WEIGHT, "defect_detector.pt"),
    (SEVERITY_WEIGHT, "severity_cls.pt"),
    (DOMAIN_DETECTOR_WEIGHT, "defect_detector_domain_adapted.pt"),
    (BALANCED_DETECTOR_WEIGHT, "defect_detector_balanced_candidate.pt"),
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
