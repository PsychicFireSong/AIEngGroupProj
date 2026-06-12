from __future__ import annotations

import base64
import os
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOTS = [
    APP_ROOT,
    APP_ROOT / "weights",
    APP_ROOT / "models",
    APP_ROOT / "runs",
    APP_ROOT / "output" / "severity_runs",
    Path("/content/drive/MyDrive/AIEngGroupProj_colab_outputs/weights"),
    Path("/content/drive/MyDrive/AIEngGroupProj_colab_outputs/runs"),
]

SEVERITY_WEIGHTS = {
    "minor": 1.0,
    "moderate": 2.4,
    "critical": 5.0,
    "uncertain": 1.5,
    "unknown": 1.0,
}

COLORS_BGR = {
    "minor": (191, 212, 45),
    "moderate": (21, 204, 250),
    "critical": (133, 113, 251),
    "uncertain": (250, 139, 167),
    "unknown": (184, 196, 148),
}

# Per-class confidence floor — tuned on val set (eval_threshold_fix.py).
# All defect classes land in the 0.25-0.45 confidence band; the old 0.45 global
# threshold was silently discarding valid detections for crack/spalling/pothole.
# Lowering to 0.25 recovers: crack +11% recall (+0.038 F1), pothole +7.5% recall,
# spalling +5.1% recall. Precision trades down ~10pp — acceptable for inspection use.
_PER_CLASS_CONF_OVERRIDE: dict[str, float] = {
    "crack":             0.25,  # was 0.45 global; +11% recall on val set
    "spalling":          0.25,  # was 0.45 global; +5% recall on val set
    "corrosion":         0.25,  # previously tuned; model detects but is under-confident
    "pothole":           0.25,  # was 0.45 global; +7.5% recall on val set
    "paint_degradation": 0.30,  # kept at 0.30; higher FP rate makes 0.25 too noisy
}

# Import TTA+WBF engine (provides ~+37pp corrosion recall, +11pp crack recall vs baseline)
try:
    from inference.api_tta_wbf import run_tta_wbf as _run_tta_wbf
    _TTA_AVAILABLE = True
except ImportError:
    try:
        from api_tta_wbf import run_tta_wbf as _run_tta_wbf  # fallback for legacy layout
        _TTA_AVAILABLE = True
    except ImportError:
        _TTA_AVAILABLE = False

MAX_URL_BYTES = int(os.environ.get("AIENG_MAX_URL_BYTES", 250 * 1024 * 1024))
DEFAULT_VIDEO_SCAN_FPS = float(os.environ.get("AIENG_VIDEO_SCAN_FPS", "1.0"))
MAX_VIDEO_SCAN_FPS = float(os.environ.get("AIENG_MAX_VIDEO_SCAN_FPS", "30.0"))
MAX_VIDEO_SCAN_SECONDS = float(os.environ.get("AIENG_MAX_VIDEO_SCAN_SECONDS", "600"))
MAX_VIDEO_SCAN_SAMPLES = int(os.environ.get("AIENG_MAX_VIDEO_SCAN_SAMPLES", "9000"))
MAX_RETURNED_VIDEO_FRAMES = int(os.environ.get("AIENG_MAX_RETURNED_VIDEO_FRAMES", "24"))
VIDEO_JOB_WORKERS = int(os.environ.get("AIENG_VIDEO_JOB_WORKERS", "1"))
MAX_VIDEO_JOBS = int(os.environ.get("AIENG_MAX_VIDEO_JOBS", "6"))
VIDEO_JOB_TTL_SECONDS = int(os.environ.get("AIENG_VIDEO_JOB_TTL_SECONDS", "7200"))


app = FastAPI(title="AI Facilities YOLO Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warmup_models() -> None:
    """Pre-load models at startup so the first real request is fast.
    Runs in the background to avoid blocking the health check."""
    def _load():
        try:
            paths = _discover_model_paths()
            for p in paths:
                _load_model(str(p))
        except Exception:
            pass
    threading.Thread(target=_load, daemon=True).start()


class VideoJobCancelled(Exception):
    pass


VIDEO_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, VIDEO_JOB_WORKERS), thread_name_prefix="video-scan")
VIDEO_JOBS: dict[str, dict] = {}
VIDEO_JOB_LOCK = threading.Lock()


def _is_likely_severity_model(path_value: str) -> bool:
    lowered = path_value.lower()
    return any(term in lowered for term in ("severity", "classify", "cls", "stage2", "sev_cascade_critical"))


def _is_likely_detector_model(path_value: str) -> bool:
    return not _is_likely_severity_model(path_value)


def _compact_label(model_path: Path) -> str:
    try:
        return str(model_path.relative_to(APP_ROOT))
    except ValueError:
        return str(model_path)


def _model_priority(model_path: Path, kind: str) -> tuple[int, str]:
    normalized = str(model_path).replace("\\", "/").lower()
    name = model_path.name.lower()
    if kind == "severity":
        if normalized.endswith("/weights/severity_cls.pt") and "/drive_download/" not in normalized:
            return (0, normalized)
        if name == "severity_cls.pt" and "/weights/" in normalized:
            return (1, normalized)
        if "sev_cascade_critical" in normalized and normalized.endswith("/weights/best.pt"):
            return (2, normalized)
        if "/stage2_severity/weights/best.pt" in normalized:
            return (3, normalized)
        if "/stage2_severity/weights/last.pt" in normalized:
            return (4, normalized)
    if kind == "detector":
        if "stage1_robust_later_best.pt" in normalized:
            return (0, normalized)
        if "stage1_anchor_robust_curriculum" in normalized and normalized.endswith("/weights/best.pt"):
            return (1, normalized)
        if "defect_detector_anchor_balanced_candidate.pt" in normalized:
            return (2, normalized)
        if "stage1_robust_epoch39_best.pt" in normalized:
            return (3, normalized)
        if "stage1_clean_near_done_best.pt" in normalized:
            return (4, normalized)
        if name == "defect_detector.pt" and "/weights/" in normalized:
            return (5, normalized)
        if "/stage1_defect_detector/weights/best.pt" in normalized:
            return (6, normalized)
        if normalized.endswith("/weights/best.pt"):
            return (7, normalized)
        if normalized.endswith("/best.pt"):
            return (8, normalized)
    return (9, normalized)


def _model_roots() -> list[Path]:
    roots = list(MODEL_ROOTS)
    extra = os.environ.get("AIENG_MODEL_ROOTS", "")
    for item in extra.split(os.pathsep):
        if item.strip():
            roots.append(Path(item.strip()))
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _discover_model_paths() -> list[Path]:
    candidates: set[Path] = set()
    for root in _model_roots():
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".pt":
            candidates.add(root.resolve())
            continue
        pattern = "*.pt" if root == APP_ROOT else "**/*.pt"
        for model_path in root.glob(pattern):
            if model_path.is_file():
                candidates.add(model_path.resolve())
    return sorted(candidates, key=lambda item: str(item).lower())


def _resolve_model_path(requested_path: str) -> str:
    if not requested_path:
        raise HTTPException(status_code=400, detail="Model path is required.")
    path = Path(requested_path)
    if not path.is_absolute():
        path = APP_ROOT / path
    path = path.resolve()
    discovered = {candidate.resolve() for candidate in _discover_model_paths()}
    if path not in discovered and not path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {requested_path}")
    if path.suffix.lower() != ".pt":
        raise HTTPException(status_code=400, detail="Model file must be a .pt checkpoint.")
    return str(path)


@lru_cache(maxsize=8)
def _load_model(model_path: str) -> YOLO:
    model = YOLO(model_path)
    try:
        model.fuse()
    except Exception:
        pass
    return model


def _device_arg() -> str | None:
    if torch is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return "cuda:0"
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_public_snapshot(job: dict) -> dict:
    snapshot = {
        key: value
        for key, value in job.items()
        if key not in {"future", "cancelRequested", "createdMonotonic"}
    }
    if job.get("status") == "queued":
        queued_jobs = sorted(
            (
                candidate
                for candidate in VIDEO_JOBS.values()
                if candidate.get("status") == "queued"
            ),
            key=lambda item: item.get("createdMonotonic", 0.0),
        )
        snapshot["queuePosition"] = next(
            (index + 1 for index, candidate in enumerate(queued_jobs) if candidate.get("id") == job.get("id")),
            0,
        )
    else:
        snapshot["queuePosition"] = 0
    return snapshot


def _get_job_snapshot(job_id: str) -> dict:
    with VIDEO_JOB_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Video job not found: {job_id}")
        return _job_public_snapshot(job)


def _update_video_job(job_id: str, **fields) -> None:
    with VIDEO_JOB_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updatedAt"] = _now_iso()


def _video_job_cancel_requested(job_id: str) -> bool:
    with VIDEO_JOB_LOCK:
        return bool(VIDEO_JOBS.get(job_id, {}).get("cancelRequested"))


def _prune_video_jobs_locked() -> None:
    now = time.monotonic()
    removable = []
    for job_id, job in VIDEO_JOBS.items():
        status = job.get("status")
        finished = float(job.get("finishedMonotonic") or 0.0)
        if status in {"completed", "failed", "cancelled"} and finished and now - finished > VIDEO_JOB_TTL_SECONDS:
            removable.append(job_id)
    for job_id in removable:
        VIDEO_JOBS.pop(job_id, None)


def _active_video_job_count_locked() -> int:
    _prune_video_jobs_locked()
    return sum(1 for job in VIDEO_JOBS.values() if job.get("status") in {"queued", "running"})


def _name_from_result(result, class_id: int) -> str:
    names = getattr(result, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)



def _perdefect_severity_model(defect_class: str, base_severity_path: str) -> YOLO | None:
    """Return a per-defect severity model if one exists alongside the base model."""
    try:
        base = Path(base_severity_path)
        candidate = base.parent / f"severity_{defect_class}_cls.pt"
        if candidate.exists():
            return _load_model(str(candidate))
    except Exception:
        pass
    return None


def _classify_severity(
    model: YOLO,
    crop_bgr: np.ndarray,
    min_conf: float = 0.40,
) -> tuple[str, float]:
    if crop_bgr.size == 0:
        return "unknown", 0.0
    results = model.predict(source=crop_bgr, imgsz=224, device=_device_arg(), verbose=False)
    if not results or getattr(results[0], "probs", None) is None:
        return "unknown", 0.0
    result = results[0]
    probs = result.probs
    severity = _name_from_result(result, int(probs.top1)).lower()
    confidence = float(probs.top1conf)
    if severity not in SEVERITY_WEIGHTS:
        return "unknown", confidence
    return severity, confidence


# Cascade threshold: if is_critical model confidence >= this, route to "critical".
# 0.30 is optimal for patch test set; 0.70 for full facade photos.
_CASCADE_THRESHOLD = 0.30
_cascade_minmod_cache: dict[str, "YOLO | None"] = {}


def _cascade_companion(crit_path: str) -> "YOLO | None":
    """Return the minor_or_moderate model paired with a cascade_critical model, or None."""
    if crit_path in _cascade_minmod_cache:
        return _cascade_minmod_cache[crit_path]
    companion = None
    if "cascade_critical" in crit_path.lower():
        minmod_pt = Path(crit_path).parent.parent.parent / "sev_cascade_minmod" / "weights" / "best.pt"
        if minmod_pt.exists():
            companion = _load_model(str(minmod_pt))
    _cascade_minmod_cache[crit_path] = companion
    return companion


def _classify_severity_cascade(
    model: YOLO,
    crop_bgr: np.ndarray,
    resolved_severity_path: str,
) -> tuple[str, float]:
    """Classify severity using cascade routing if model is cascade_critical, else single model."""
    companion = _cascade_companion(resolved_severity_path)
    if companion is None:
        return _classify_severity(model, crop_bgr)

    if crop_bgr.size == 0:
        return "unknown", 0.0
    r1 = model.predict(source=crop_bgr, imgsz=224, device=_device_arg(), verbose=False)
    if not r1 or getattr(r1[0], "probs", None) is None:
        return "unknown", 0.0
    probs1 = r1[0].probs.data.cpu().numpy()
    crit_idx = next((i for i, n in r1[0].names.items() if n.lower() == "critical"), 0)
    crit_conf = float(probs1[crit_idx])

    if crit_conf >= _CASCADE_THRESHOLD:
        return "critical", crit_conf

    r2 = companion.predict(source=crop_bgr, imgsz=224, device=_device_arg(), verbose=False)
    if not r2 or getattr(r2[0], "probs", None) is None:
        return "unknown", 0.0
    pred = r2[0].names[int(r2[0].probs.top1)].lower()
    conf = float(r2[0].probs.top1conf)
    return pred, conf


def _priority_for(severity: str, confidence: float, area_ratio: float) -> Literal["Immediate", "Planned", "Monitor"]:
    if severity == "critical" or (confidence >= 0.75 and area_ratio >= 0.12):
        return "Immediate"
    if severity == "moderate" or confidence >= 0.55:
        return "Planned"
    return "Monitor"


def _action_for(defect_class: str, severity: str, priority: str, confidence: float, area_ratio: float) -> str:
    defect_key = defect_class.lower().strip()
    extent = (
        "large-area"
        if area_ratio >= 0.12
        else "moderate-area"
        if area_ratio >= 0.035
        else "localized"
    )
    confidence_note = "high-confidence" if confidence >= 0.7 else "medium-confidence" if confidence >= 0.4 else "low-confidence"
    class_guidance = {
        "crack": "measure crack width/length, photograph both ends, and check whether the crack is active or moisture-stained",
        "spalling": "tap-test surrounding concrete, remove loose material, and inspect for exposed reinforcement before patch repair",
        "corrosion": "check water ingress, estimate section loss, clean/passivate the metal surface, and plan protective recoating or replacement",
        "pothole": "apply traffic control if exposed to vehicles, mark the hazard, and schedule patching with drainage check",
        "paint_degradation": "check moisture source, scrape unstable coating, repair substrate, and repaint with suitable primer/topcoat",
    }.get(defect_key, "verify the finding on site, document the location, and assign the appropriate maintenance owner")

    if priority == "Immediate":
        return (
            f"Immediate {extent} {defect_class} response: restrict access if the area is reachable or safety-sensitive, "
            f"then {class_guidance}. Evidence is {confidence_note}; confirm on site before closing the work order."
        )
    if severity == "moderate":
        return (
            f"Planned {extent} {defect_class} repair: {class_guidance}. Use the annotated image as the work-order evidence "
            f"and compare with the next inspection for progression."
        )
    if severity == "uncertain":
        return (
            f"Manual review required for {defect_class}: confidence or severity margin is weak. Recheck with a closer image, "
            f"then decide whether to log, monitor, or schedule repair."
        )
    return (
        f"Monitor {extent} {defect_class}: log the evidence, {class_guidance}, and recheck during the next inspection cycle."
    )


def _risk_score(detections: list[dict]) -> tuple[float, float]:
    if not detections:
        return 0.0, 100.0
    score = 0.0
    for detection in detections:
        severity_weight = SEVERITY_WEIGHTS.get(detection["severity"], 1.0)
        confidence_factor = max(float(detection["confidence"]), 0.2)
        area_factor = 1.0 + min(float(detection["areaRatio"]) * 12.0, 3.0)
        score += severity_weight * confidence_factor * area_factor * 4.0
    score = min(round(score, 1), 100.0)
    return score, round(max(0.0, 100.0 - score), 1)


def _decode_upload(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")
    return image


def _decode_image_or_none(payload: bytes) -> np.ndarray | None:
    if not payload:
        return None
    return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)


def _is_youtube_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "youtube.com" in host or "youtu.be" in host


def _google_drive_download_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "drive.google.com" not in host and "docs.google.com" not in host:
        return url

    query_id = parse_qs(parsed.query).get("id", [""])[0]
    if query_id:
        return f"https://drive.google.com/uc?export=download&id={query_id}"

    match = re.search(r"/(?:file/d|open\?id=|uc\?id=)/?([A-Za-z0-9_-]+)", url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"

    match = re.search(r"/d/([A-Za-z0-9_-]+)", url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"

    return url


def _download_public_url(url: str, max_bytes: int = MAX_URL_BYTES) -> tuple[bytes, str]:
    request = Request(
        _google_drive_download_url(url),
        headers={
            "User-Agent": "Mozilla/5.0 AIEngGroupProj/1.0",
            "Accept": "image/*,video/*,application/octet-stream,*/*",
        },
    )
    with urlopen(request, timeout=45) as response:
        content_type = response.headers.get("content-type", "")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Linked media is larger than the configured {max_bytes // (1024 * 1024)} MB limit.",
                )
            chunks.append(chunk)
    return b"".join(chunks), content_type


def _video_duration_seconds(source: str | Path) -> float | None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return None
    try:
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        if frame_count > 0 and fps > 0:
            return frame_count / fps
        duration_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0)
        return duration_ms / 1000.0 if duration_ms > 0 else None
    finally:
        capture.release()


def _video_scan_times(
    frame_time_sec: float,
    scan_fps: float,
    duration_sec: float | None = None,
    scan_video: bool = True,
    max_scan_seconds: float = MAX_VIDEO_SCAN_SECONDS,
) -> tuple[list[float], bool, float]:
    start = max(0.0, float(frame_time_sec or 0.0))
    if not scan_video:
        return [round(start, 2)], False, 0.0

    fps = max(0.05, min(float(scan_fps or DEFAULT_VIDEO_SCAN_FPS), MAX_VIDEO_SCAN_FPS))
    interval = 1.0 / fps
    known_duration = float(duration_sec or 0.0)
    if known_duration > 0:
        end = max(start, known_duration)
    else:
        end = start + max_scan_seconds
    if max_scan_seconds > 0:
        end = min(end, start + max_scan_seconds)
    coverage_seconds = max(0.0, end - start)
    estimated_samples = int(np.floor(coverage_seconds / interval)) + 1
    safety_capped = estimated_samples > MAX_VIDEO_SCAN_SAMPLES
    sample_count = max(1, min(estimated_samples, MAX_VIDEO_SCAN_SAMPLES))
    times = [round(start + (index * interval), 2) for index in range(sample_count)]
    if known_duration > 0:
        times = [value for value in times if value <= known_duration + 0.25]
    return times or [round(start, 2)], safety_capped, coverage_seconds


def _sample_video_frames_from_source(
    source: str | Path,
    frame_time_sec: float,
    scan_fps: float,
    scan_video: bool = True,
    duration_sec: float | None = None,
    allow_empty: bool = False,
) -> list[dict]:
    duration = duration_sec if duration_sec and duration_sec > 0 else _video_duration_seconds(source)
    times, safety_capped, coverage_seconds = _video_scan_times(frame_time_sec, scan_fps, duration, scan_video)
    frames: list[dict] = []
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        if not allow_empty:
            raise HTTPException(status_code=400, detail="Linked media is not a readable video.")
        return frames
    try:
        for seconds in times:
            if seconds > 0:
                capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames.append(
                {
                    "frameTimeSec": seconds,
                    "image": frame,
                    "mediaKind": "video",
                    "scanFps": scan_fps,
                    "safetyCapped": safety_capped,
                    "coverageSeconds": round(coverage_seconds, 2),
                }
            )
    finally:
        capture.release()

    if not frames:
        capture = cv2.VideoCapture(str(source))
        try:
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame is not None:
                    frames.append(
                        {
                            "frameTimeSec": max(0.0, frame_time_sec),
                            "image": frame,
                            "mediaKind": "video",
                            "scanFps": scan_fps,
                            "safetyCapped": safety_capped,
                            "coverageSeconds": round(coverage_seconds, 2),
                        }
                    )
        finally:
            capture.release()

    if not frames and not allow_empty:
        raise HTTPException(status_code=400, detail="Could not extract usable video frames from the linked media.")
    return frames


def _frames_from_video_file(
    video_path: Path,
    seconds: float,
    scan_fps: float = DEFAULT_VIDEO_SCAN_FPS,
    scan_video: bool = False,
) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Linked media is not a readable video.")
    finally:
        capture.release()
    return _sample_video_frames_from_source(video_path, seconds, scan_fps, scan_video)


def _frame_from_video_file(video_path: Path, seconds: float) -> np.ndarray:
    return _frames_from_video_file(video_path, seconds, scan_video=False)[0]["image"]


def _youtube_info(url: str) -> dict:
    try:
        import yt_dlp
    except ImportError as exc:
        raise HTTPException(
            status_code=400,
            detail="YouTube links require yt-dlp. Install requirements.txt or run: pip install yt-dlp",
        ) from exc

    options = {
        "quiet": True,
        "skip_download": True,
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        return downloader.extract_info(url, download=False)


def _youtube_direct_url(info: dict) -> str:
    direct_url = info.get("url")
    if not direct_url:
        formats = info.get("formats") or []
        mp4_formats = [item for item in formats if item.get("url") and item.get("ext") == "mp4"]
        if mp4_formats:
            direct_url = sorted(mp4_formats, key=lambda item: item.get("height") or 0)[-1]["url"]
    return str(direct_url or "")


def _frames_from_youtube_url(
    url: str,
    seconds: float,
    scan_fps: float = DEFAULT_VIDEO_SCAN_FPS,
    scan_video: bool = False,
) -> list[dict]:
    info = _youtube_info(url)
    direct_url = _youtube_direct_url(info)
    if direct_url:
        frames = _sample_video_frames_from_source(
            direct_url,
            seconds,
            scan_fps,
            scan_video,
            duration_sec=float(info.get("duration") or 0) or None,
            allow_empty=True,
        )
        if frames:
            return [{**item, "mediaKind": "youtube"} for item in frames]

    thumbnail = info.get("thumbnail")
    if thumbnail:
        payload, _ = _download_public_url(thumbnail, max_bytes=12 * 1024 * 1024)
        image = _decode_image_or_none(payload)
        if image is not None:
            return [{"frameTimeSec": max(0.0, seconds), "image": image, "mediaKind": "youtube_thumbnail"}]

    raise HTTPException(status_code=400, detail="Could not extract a frame from the YouTube link.")


def _frame_from_youtube_url(url: str, seconds: float) -> np.ndarray:
    return _frames_from_youtube_url(url, seconds, scan_video=False)[0]["image"]


def _frames_from_media_url(
    url: str,
    seconds: float,
    scan_fps: float = DEFAULT_VIDEO_SCAN_FPS,
    scan_video: bool = False,
) -> list[dict]:
    if _is_youtube_url(url):
        return _frames_from_youtube_url(url, seconds, scan_fps, scan_video)

    payload, content_type = _download_public_url(url)
    image = _decode_image_or_none(payload)
    if image is not None:
        return [{"frameTimeSec": max(0.0, seconds), "image": image, "mediaKind": "image"}]

    suffix = ".mp4"
    if "webm" in content_type:
        suffix = ".webm"
    elif "quicktime" in content_type or "mov" in content_type:
        suffix = ".mov"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    try:
        return _frames_from_video_file(temp_path, seconds, scan_fps, scan_video)
    finally:
        temp_path.unlink(missing_ok=True)


def _frame_from_media_url(url: str, seconds: float) -> np.ndarray:
    return _frames_from_media_url(url, seconds, 1)[0]["image"]


def _video_result_score(result: dict) -> float:
    detections = result.get("detections") or []
    if not detections:
        return 0.0
    score = float(len(detections)) * 10.0
    for detection in detections:
        severity = str(detection.get("severity", "unknown"))
        score += float(detection.get("confidence") or 0.0) * 4.0
        score += SEVERITY_WEIGHTS.get(severity, 1.0)
        score += min(float(detection.get("areaRatio") or 0.0) * 10.0, 2.5)
    return score


def _summarize_sample(result: dict, frame_time_sec: float, confidence: float) -> dict:
    detections = result.get("detections") or []
    max_confidence = max((float(item.get("confidence") or 0.0) for item in detections), default=0.0)
    classes = sorted({str(item.get("className", "unknown")) for item in detections})
    return {
        "frameTimeSec": round(float(frame_time_sec), 2),
        "detections": len(detections),
        "maxConfidence": round(max_confidence, 4),
        "classes": classes,
        "confidence": round(float(confidence), 3),
    }


def _encode_jpeg_data_url(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _draw_label(image: np.ndarray, x1: int, y1: int, label: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    top = max(0, y1 - text_height - baseline - 8)
    cv2.rectangle(image, (x1, top), (x1 + text_width + 12, top + text_height + baseline + 8), color, -1)
    cv2.putText(image, label, (x1 + 6, top + text_height + 3), font, scale, (8, 12, 18), thickness, cv2.LINE_AA)


def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = float(area_a + area_b - inter_area)
    return inter_area / union if union > 0 else 0.0


def _run_inference(
    image_bgr: np.ndarray,
    detector_path: str,
    severity_path: str,
    confidence: float,
    iou: float,
    include_image: bool = True,
    use_tta: bool = False,
    agnostic_nms: bool = False,
) -> dict:
    resolved_detector_path = _resolve_model_path(detector_path)
    resolved_severity_path = _resolve_model_path(severity_path)
    detector = _load_model(resolved_detector_path)
    severity_model = _load_model(resolved_severity_path)
    height, width = image_bgr.shape[:2]
    started = time.perf_counter()

    # ── Stage 1: detect defects ───────────────────────────────────────────────
    stage1_started = time.perf_counter()

    if use_tta and _TTA_AVAILABLE:
        # TTA+WBF path: run at 640+1280+hflip, merge with Weighted Box Fusion.
        # Improves corrosion recall by ~+37pp and crack recall by ~+11pp.
        raw_dets = _run_tta_wbf(detector, image_bgr, device=_device_arg(), base_conf=confidence)
        stage1_latency_ms = (time.perf_counter() - stage1_started) * 1000.0

        annotated = image_bgr.copy()
        detections: list[dict] = []
        stage2_latency_ms = 0.0
        kept_boxes: list[tuple[tuple[int, int, int, int], int]] = []
        for det in sorted(raw_dets, key=lambda d: d['conf'], reverse=True):
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            class_id = det['class_id']
            defect_conf = det['conf']
            class_name_lookup = det['class_name']
            box_tuple = (x1, y1, x2, y2)
            if any(ec == class_id and _bbox_iou(eb, box_tuple) >= 0.88
                   for eb, ec in kept_boxes):
                continue
            kept_boxes.append((box_tuple, class_id))
            crop = image_bgr[y1:y2, x1:x2]
            defect_class = class_name_lookup
            s2_start = time.perf_counter()
            active_sev = _perdefect_severity_model(defect_class, resolved_severity_path) or severity_model
            severity, severity_conf = _classify_severity_cascade(active_sev, crop, resolved_severity_path)
            stage2_latency_ms += (time.perf_counter() - s2_start) * 1000.0
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)
            priority = _priority_for(severity, defect_conf, area_ratio)
            color = COLORS_BGR.get(severity, COLORS_BGR["unknown"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            _draw_label(annotated, x1, y1, f"{defect_class} | {severity}", color)
            detections.append({
                "id": str(uuid.uuid4()),
                "className": defect_class,
                "confidence": round(defect_conf, 4),
                "severity": severity,
                "severityConfidence": round(severity_conf, 4),
                "priority": priority,
                "action": _action_for(defect_class, severity, priority, defect_conf, area_ratio),
                "bbox": {
                    "x": round(x1 / width, 6), "y": round(y1 / height, 6),
                    "width": round((x2 - x1) / width, 6), "height": round((y2 - y1) / height, 6),
                },
                "areaRatio": round(area_ratio, 6),
            })
    else:
        # Standard single-pass path (original logic)
        conf_floor = min(confidence, min(_PER_CLASS_CONF_OVERRIDE.values(), default=confidence))
        results = detector.predict(
            source=image_bgr,
            conf=conf_floor,
            iou=iou,
            imgsz=640,
            max_det=120,
            agnostic_nms=agnostic_nms,
            device=_device_arg(),
            verbose=False,
        )
        stage1_latency_ms = (time.perf_counter() - stage1_started) * 1000.0

        annotated = image_bgr.copy()
        detections: list[dict] = []
        stage2_latency_ms = 0.0
        if results:
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                kept_boxes: list[tuple[tuple[int, int, int, int], int]] = []
                for index in sorted(range(len(xyxy)), key=lambda item: float(confidences[item]), reverse=True):
                    coords = xyxy[index]
                    x1, y1, x2, y2 = coords.astype(int).tolist()
                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(0, min(width, x2))
                    y2 = max(0, min(height, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    class_id = int(class_ids[index])
                    defect_conf_val = float(confidences[index])
                    class_name_lookup = result.names.get(class_id, str(class_id))
                    effective_threshold = _PER_CLASS_CONF_OVERRIDE.get(class_name_lookup, confidence)
                    if defect_conf_val < effective_threshold:
                        continue  # apply per-class confidence gate in post-processing
                    box_tuple = (x1, y1, x2, y2)
                    if any(existing_class == class_id and _bbox_iou(existing_box, box_tuple) >= 0.88 for existing_box, existing_class in kept_boxes):
                        continue
                    kept_boxes.append((box_tuple, class_id))

                    crop = image_bgr[y1:y2, x1:x2]
                    defect_class = _name_from_result(result, class_id)
                    stage2_started = time.perf_counter()
                    active_sev = _perdefect_severity_model(defect_class, resolved_severity_path) or severity_model
                    severity, severity_conf = _classify_severity_cascade(active_sev, crop, resolved_severity_path)
                    stage2_latency_ms += (time.perf_counter() - stage2_started) * 1000.0
                    defect_conf = float(confidences[index])
                    area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)
                    priority = _priority_for(severity, defect_conf, area_ratio)
                    color = COLORS_BGR.get(severity, COLORS_BGR["unknown"])

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    _draw_label(annotated, x1, y1, f"{defect_class} | {severity}", color)

                    detections.append(
                        {
                            "id": str(uuid.uuid4()),
                            "className": defect_class,
                            "confidence": round(defect_conf, 4),
                            "severity": severity,
                            "severityConfidence": round(severity_conf, 4),
                            "priority": priority,
                            "action": _action_for(defect_class, severity, priority, defect_conf, area_ratio),
                            "bbox": {
                                "x": round(x1 / width, 6),
                                "y": round(y1 / height, 6),
                                "width": round((x2 - x1) / width, 6),
                                "height": round((y2 - y1) / height, 6),
                            },
                            "areaRatio": round(area_ratio, 6),
                        }
                    )

    risk, condition = _risk_score(detections)
    latency_ms = (time.perf_counter() - started) * 1000.0
    payload = {
        "detections": detections,
        "latencyMs": round(latency_ms, 2),
        "stageMetrics": {
            "stage1LatencyMs": round(stage1_latency_ms, 2),
            "stage2LatencyMs": round(stage2_latency_ms, 2),
            "cropsClassified": len(detections),
            "detectorModel": Path(resolved_detector_path).name,
            "severityModel": Path(resolved_severity_path).name,
        },
        "conditionIndex": condition,
        "riskScore": risk,
    }
    if include_image:
        payload["evidenceImage"] = _encode_jpeg_data_url(image_bgr)
        payload["annotatedImage"] = _encode_jpeg_data_url(annotated)
    return payload


def _analyze_media_url(
    media_url: str = "",
    detector_path: str = "",
    severity_path: str = "",
    confidence: float = 0.45,
    iou: float = 0.45,
    frame_time_sec: float = 0.0,
    scan_video: bool = True,
    scan_fps: float = DEFAULT_VIDEO_SCAN_FPS,
    review_confidence: float = 0.18,
    local_video_path: Optional[str] = None,
    progress_callback=None,
    cancel_requested=None,
) -> dict:
    if not local_video_path and not media_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Media URL must start with http:// or https://.")

    def update_progress(**fields) -> None:
        if progress_callback:
            progress_callback(**fields)

    def check_cancelled() -> None:
        if cancel_requested and cancel_requested():
            raise VideoJobCancelled()

    scan_started = time.perf_counter()
    requested_confidence = max(0.01, min(float(confidence), 0.99))
    review_confidence = max(0.01, min(float(review_confidence), requested_confidence))
    requested_frame_time = max(0.0, frame_time_sec)
    requested_scan_fps = max(0.05, min(float(scan_fps or DEFAULT_VIDEO_SCAN_FPS), MAX_VIDEO_SCAN_FPS))

    update_progress(progress=2, stage="resolving", message="Resolving linked media and preparing frame sampling.")
    check_cancelled()
    if local_video_path:
        frames = _frames_from_video_file(Path(local_video_path), requested_frame_time, requested_scan_fps, True)
    else:
        frames = _frames_from_media_url(media_url, requested_frame_time, requested_scan_fps, scan_video)
    total_frames = max(1, len(frames))
    update_progress(
        progress=8,
        stage="queued_frames",
        message=f"Prepared {total_frames} sampled frame{'s' if total_frames != 1 else ''}.",
        totalFrames=total_frames,
        processedFrames=0,
        positiveFrames=0,
    )

    best_result: dict | None = None
    best_frame: dict | None = None
    best_confidence = requested_confidence
    sampled_frames: list[dict] = []
    positive_frame_candidates: list[dict] = []
    confidence_passes = [requested_confidence]
    if review_confidence < requested_confidence:
        confidence_passes.append(review_confidence)

    for pass_index, pass_confidence in enumerate(confidence_passes):
        pass_samples: list[dict] = []
        pass_positive_candidates: list[dict] = []
        pass_label = "requested" if pass_index == 0 else "review"
        for frame_index, frame in enumerate(frames):
            check_cancelled()
            result = _run_inference(
                frame["image"],
                detector_path,
                severity_path,
                pass_confidence,
                iou,
                include_image=False,
            )
            sample = _summarize_sample(result, frame["frameTimeSec"], pass_confidence)
            pass_samples.append(sample)
            score = _video_result_score(result)
            if result.get("detections"):
                pass_positive_candidates.append(
                    {
                        "score": score,
                        "frame": frame,
                        "confidence": pass_confidence,
                        "sample": sample,
                    }
                )
            if best_result is None or score > _video_result_score(best_result):
                best_result = result
                best_frame = frame
                best_confidence = pass_confidence

            processed = frame_index + 1
            progress = min(84, 10 + round((processed / total_frames) * 68))
            update_progress(
                progress=progress,
                stage="scanning",
                message=f"Scanning frame {processed}/{total_frames} with {pass_label} sensitivity.",
                totalFrames=total_frames,
                processedFrames=processed,
                positiveFrames=len(pass_positive_candidates),
                currentFrameTimeSec=round(float(frame["frameTimeSec"]), 2),
                usedConfidence=round(pass_confidence, 3),
            )

        sampled_frames = pass_samples
        positive_frame_candidates = pass_positive_candidates
        if pass_positive_candidates:
            break

    if best_result is None or best_frame is None:
        raise HTTPException(status_code=400, detail="No usable media frame could be analyzed.")

    check_cancelled()
    update_progress(
        progress=88,
        stage="rendering_evidence",
        message="Rendering selected evidence and defect-frame snapshots.",
        positiveFrames=len(positive_frame_candidates),
    )
    render_confidence = min(best_confidence, review_confidence)
    result = _run_inference(best_frame["image"], detector_path, severity_path, render_confidence, iou)
    detected_frames = []
    returned_candidates = sorted(
        sorted(positive_frame_candidates, key=lambda candidate: candidate["frame"]["frameTimeSec"]),
        key=lambda candidate: candidate["score"],
        reverse=True,
    )[:MAX_RETURNED_VIDEO_FRAMES]
    returned_candidates.sort(key=lambda candidate: candidate["frame"]["frameTimeSec"])
    for index, candidate in enumerate(returned_candidates):
        check_cancelled()
        frame = candidate["frame"]
        frame_render_confidence = min(float(candidate["confidence"]), review_confidence)
        frame_result = _run_inference(
            frame["image"],
            detector_path,
            severity_path,
            frame_render_confidence,
            iou,
            include_image=True,
        )
        rendered_sample = _summarize_sample(frame_result, frame["frameTimeSec"], frame_render_confidence)
        detected_frames.append(
            {
                **rendered_sample,
                "id": f"frame-{index}-{round(float(frame['frameTimeSec']), 2)}",
                "evidenceImage": frame_result.get("evidenceImage"),
                "annotatedImage": frame_result.get("annotatedImage"),
                "detectionItems": frame_result.get("detections", []),
                "conditionIndex": frame_result.get("conditionIndex"),
                "riskScore": frame_result.get("riskScore"),
            }
        )
        update_progress(
            progress=min(96, 88 + round(((index + 1) / max(1, len(returned_candidates))) * 8)),
            stage="rendering_evidence",
            message=f"Rendering defect frame {index + 1}/{max(1, len(returned_candidates))}.",
            returnedDefectFrames=len(detected_frames),
        )

    result["latencyMs"] = round((time.perf_counter() - scan_started) * 1000.0, 2)
    result["stageMetrics"]["videoScanLatencyMs"] = result["latencyMs"]
    result["stageMetrics"]["sampledFrames"] = len(frames)
    result["stageMetrics"]["scanFps"] = round(requested_scan_fps, 3)
    result["stageMetrics"]["usedConfidence"] = round(render_confidence, 3)
    result["source"] = {
        "kind": "url",
        "mediaKind": best_frame.get("mediaKind", "unknown"),
        "frameTimeSec": requested_frame_time,
        "selectedFrameTimeSec": round(float(best_frame["frameTimeSec"]), 2),
        "sampleCount": len(frames),
        "scanFps": round(requested_scan_fps, 3),
        "coverageSeconds": best_frame.get("coverageSeconds"),
        "safetyCapped": bool(best_frame.get("safetyCapped", False)),
        "sampledFrames": sampled_frames,
        "detectedFrames": detected_frames,
        "positiveFrameCount": len(positive_frame_candidates),
        "returnedDefectFrames": len(detected_frames),
        "requestedConfidence": round(requested_confidence, 3),
        "rankingConfidence": round(best_confidence, 3),
        "usedConfidence": round(render_confidence, 3),
        "reviewMode": render_confidence < requested_confidence,
        "youtube": _is_youtube_url(media_url),
        "googleDrive": "drive.google.com" in urlparse(media_url).netloc.lower(),
    }
    update_progress(progress=100, stage="completed", message="Video analysis completed.")
    return result


def _run_video_job(job_id: str, params: dict) -> None:
    local_path = params.get("local_video_path")
    # delete_after_job=False for sample files that must not be removed after analysis
    delete_after = params.pop("delete_after_job", True)

    def _cleanup():
        if local_path and delete_after:
            Path(local_path).unlink(missing_ok=True)

    if _video_job_cancel_requested(job_id):
        _update_video_job(
            job_id,
            status="cancelled",
            progress=0,
            stage="cancelled",
            message="Video analysis was cancelled before it started.",
            finishedAt=_now_iso(),
            finishedMonotonic=time.monotonic(),
        )
        _cleanup()
        return

    _update_video_job(
        job_id,
        status="running",
        startedAt=_now_iso(),
        progress=1,
        stage="starting",
        message="Video analysis worker started.",
    )
    try:
        result = _analyze_media_url(
            **params,
            progress_callback=lambda **fields: _update_video_job(job_id, **fields),
            cancel_requested=lambda: _video_job_cancel_requested(job_id),
        )
    except VideoJobCancelled:
        _update_video_job(
            job_id,
            status="cancelled",
            progress=0,
            stage="cancelled",
            message="Video analysis was cancelled.",
            finishedAt=_now_iso(),
            finishedMonotonic=time.monotonic(),
        )
        _cleanup()
        return
    except HTTPException as exc:
        _update_video_job(
            job_id,
            status="failed",
            progress=0,
            stage="failed",
            message=str(exc.detail),
            error=str(exc.detail),
            finishedAt=_now_iso(),
            finishedMonotonic=time.monotonic(),
        )
        _cleanup()
        return
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        _update_video_job(
            job_id,
            status="failed",
            progress=0,
            stage="failed",
            message=str(exc),
            error=str(exc),
            finishedAt=_now_iso(),
            finishedMonotonic=time.monotonic(),
        )
        _cleanup()
        return

    _update_video_job(
        job_id,
        status="completed",
        progress=100,
        stage="completed",
        message="Video analysis completed.",
        result=result,
        finishedAt=_now_iso(),
        finishedMonotonic=time.monotonic(),
        totalFrames=result.get("source", {}).get("sampleCount"),
        positiveFrames=result.get("source", {}).get("positiveFrameCount"),
        returnedDefectFrames=result.get("source", {}).get("returnedDefectFrames"),
    )
    _cleanup()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "device": "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu",
        "models": len(_discover_model_paths()),
    }


@app.get("/warmup")
def warmup() -> dict:
    """Force model pre-loading. Call this once after a cold start to avoid
    the first inference request paying the model-load penalty (~5-8 s on CPU)."""
    paths = _discover_model_paths()
    loaded = []
    for p in paths:
        try:
            _load_model(str(p))
            loaded.append(Path(p).name)
        except Exception as exc:
            loaded.append(f"{Path(p).name} [error: {exc}]")
    return {"loaded": loaded, "device": "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"}


@app.get("/api/models")
def models() -> dict:
    options = []
    for model_path in _discover_model_paths():
        value = str(model_path)
        if _is_likely_severity_model(value):
            options.append(
                {
                    "label": _compact_label(model_path),
                    "value": value,
                    "kind": "severity",
                    "_priority": _model_priority(model_path, "severity"),
                }
            )
        if _is_likely_detector_model(value):
            options.append(
                {
                    "label": _compact_label(model_path),
                    "value": value,
                    "kind": "detector",
                    "_priority": _model_priority(model_path, "detector"),
                }
            )
    options.sort(key=lambda option: (option["kind"], option["_priority"]))
    for option in options:
        option.pop("_priority", None)
    return {"models": options}


@app.post("/api/infer/image")
async def infer_image(
    file: UploadFile = File(...),
    detector_path: str = Form(...),
    severity_path: str = Form(...),
    confidence: float = Form(0.45),
    iou: float = Form(0.45),
    use_tta: bool = Form(False),
    agnostic_nms: bool = Form(False),
) -> dict:
    payload = await file.read()
    image = _decode_upload(payload)
    return _run_inference(image, detector_path, severity_path, confidence, iou, use_tta=use_tta, agnostic_nms=agnostic_nms)


@app.post("/api/infer/url")
async def infer_url(
    media_url: str = Form(...),
    detector_path: str = Form(...),
    severity_path: str = Form(...),
    confidence: float = Form(0.45),
    iou: float = Form(0.45),
    frame_time_sec: float = Form(0.0),
    scan_video: bool = Form(True),
    scan_fps: float = Form(DEFAULT_VIDEO_SCAN_FPS),
    review_confidence: float = Form(0.18),
) -> dict:
    return _analyze_media_url(
        media_url=media_url,
        detector_path=detector_path,
        severity_path=severity_path,
        confidence=confidence,
        iou=iou,
        frame_time_sec=frame_time_sec,
        scan_video=scan_video,
        scan_fps=scan_fps,
        review_confidence=review_confidence,
    )


@app.post("/api/jobs/video")
async def submit_video_job(
    media_url: str = Form(...),
    detector_path: str = Form(...),
    severity_path: str = Form(...),
    confidence: float = Form(0.45),
    iou: float = Form(0.45),
    frame_time_sec: float = Form(0.0),
    scan_video: bool = Form(True),
    scan_fps: float = Form(DEFAULT_VIDEO_SCAN_FPS),
    review_confidence: float = Form(0.18),
) -> dict:
    if not media_url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Media URL must start with http:// or https://.")
    _resolve_model_path(detector_path)
    _resolve_model_path(severity_path)

    with VIDEO_JOB_LOCK:
        if _active_video_job_count_locked() >= MAX_VIDEO_JOBS:
            raise HTTPException(status_code=429, detail=f"Video queue is full. Maximum active jobs: {MAX_VIDEO_JOBS}.")

        job_id = str(uuid.uuid4())
        params = {
            "media_url": media_url,
            "detector_path": detector_path,
            "severity_path": severity_path,
            "confidence": confidence,
            "iou": iou,
            "frame_time_sec": frame_time_sec,
            "scan_video": scan_video,
            "scan_fps": scan_fps,
            "review_confidence": review_confidence,
        }
        job = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "stage": "queued",
            "message": "Video analysis queued.",
            "mediaUrl": media_url,
            "createdAt": _now_iso(),
            "updatedAt": _now_iso(),
            "createdMonotonic": time.monotonic(),
            "totalFrames": None,
            "processedFrames": 0,
            "positiveFrames": 0,
            "returnedDefectFrames": 0,
            "scanFps": round(max(0.05, min(float(scan_fps or DEFAULT_VIDEO_SCAN_FPS), MAX_VIDEO_SCAN_FPS)), 3),
            "cancelRequested": False,
        }
        VIDEO_JOBS[job_id] = job
        future = VIDEO_JOB_EXECUTOR.submit(_run_video_job, job_id, params)
        job["future"] = future
        return _job_public_snapshot(job)


@app.post("/api/jobs/video/upload")
async def submit_video_upload_job(
    video_file: UploadFile = File(...),
    detector_path: str = Form(...),
    severity_path: str = Form(...),
    confidence: float = Form(0.45),
    iou: float = Form(0.45),
    frame_time_sec: float = Form(0.0),
    scan_fps: float = Form(DEFAULT_VIDEO_SCAN_FPS),
    review_confidence: float = Form(0.18),
) -> dict:
    _resolve_model_path(detector_path)
    _resolve_model_path(severity_path)

    with VIDEO_JOB_LOCK:
        if _active_video_job_count_locked() >= MAX_VIDEO_JOBS:
            raise HTTPException(status_code=429, detail=f"Video queue is full. Maximum active jobs: {MAX_VIDEO_JOBS}.")

    content = await video_file.read()
    suffix = Path(video_file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        tmp_path = handle.name
        handle.write(content)

    with VIDEO_JOB_LOCK:
        if _active_video_job_count_locked() >= MAX_VIDEO_JOBS:
            Path(tmp_path).unlink(missing_ok=True)
            raise HTTPException(status_code=429, detail=f"Video queue is full. Maximum active jobs: {MAX_VIDEO_JOBS}.")

        job_id = str(uuid.uuid4())
        params = {
            "local_video_path": tmp_path,
            "media_url": "",
            "detector_path": detector_path,
            "severity_path": severity_path,
            "confidence": confidence,
            "iou": iou,
            "frame_time_sec": frame_time_sec,
            "scan_video": True,
            "scan_fps": scan_fps,
            "review_confidence": review_confidence,
        }
        job = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "stage": "queued",
            "message": "Video upload queued for analysis.",
            "mediaUrl": f"[uploaded: {video_file.filename}]",
            "createdAt": _now_iso(),
            "updatedAt": _now_iso(),
            "createdMonotonic": time.monotonic(),
            "totalFrames": None,
            "processedFrames": 0,
            "positiveFrames": 0,
            "returnedDefectFrames": 0,
            "scanFps": round(max(0.05, min(float(scan_fps or DEFAULT_VIDEO_SCAN_FPS), MAX_VIDEO_SCAN_FPS)), 3),
            "cancelRequested": False,
        }
        VIDEO_JOBS[job_id] = job
        future = VIDEO_JOB_EXECUTOR.submit(_run_video_job, job_id, params)
        job["future"] = future
        return _job_public_snapshot(job)


_SAMPLES_DIR = APP_ROOT / "apps" / "facility-dashboard" / "public" / "samples"
_ALLOWED_SAMPLE_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}


@app.post("/api/jobs/video/local-sample")
async def submit_video_local_sample_job(
    sample_name: str = Form(...),
    detector_path: str = Form(...),
    severity_path: str = Form(...),
    confidence: float = Form(0.45),
    iou: float = Form(0.45),
    frame_time_sec: float = Form(0.0),
    scan_fps: float = Form(DEFAULT_VIDEO_SCAN_FPS),
    review_confidence: float = Form(0.18),
) -> dict:
    safe_name = Path(sample_name).name  # strip any path traversal
    if Path(safe_name).suffix.lower() not in _ALLOWED_SAMPLE_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"'{safe_name}' is not a supported video file.")
    local_path = _SAMPLES_DIR / safe_name
    if not local_path.is_file():
        raise HTTPException(status_code=404, detail=f"Sample '{safe_name}' not found.")
    _resolve_model_path(detector_path)
    _resolve_model_path(severity_path)

    with VIDEO_JOB_LOCK:
        if _active_video_job_count_locked() >= MAX_VIDEO_JOBS:
            raise HTTPException(status_code=429, detail=f"Video queue is full. Maximum active jobs: {MAX_VIDEO_JOBS}.")

        job_id = str(uuid.uuid4())
        params = {
            "local_video_path": str(local_path),
            "delete_after_job": False,  # never delete bundled sample files
            "media_url": f"[sample: {safe_name}]",
            "detector_path": detector_path,
            "severity_path": severity_path,
            "confidence": confidence,
            "iou": iou,
            "frame_time_sec": frame_time_sec,
            "scan_video": True,
            "scan_fps": scan_fps,
            "review_confidence": review_confidence,
        }
        job = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "stage": "queued",
            "message": f"Sample '{safe_name}' queued for analysis.",
            "mediaUrl": f"[sample: {safe_name}]",
            "createdAt": _now_iso(),
            "updatedAt": _now_iso(),
            "createdMonotonic": time.monotonic(),
            "totalFrames": None,
            "processedFrames": 0,
            "positiveFrames": 0,
            "returnedDefectFrames": 0,
            "scanFps": round(max(0.05, min(float(scan_fps or DEFAULT_VIDEO_SCAN_FPS), MAX_VIDEO_SCAN_FPS)), 3),
            "cancelRequested": False,
        }
        VIDEO_JOBS[job_id] = job
        future = VIDEO_JOB_EXECUTOR.submit(_run_video_job, job_id, params)
        job["future"] = future
        return _job_public_snapshot(job)


@app.get("/api/jobs")
def list_jobs() -> dict:
    with VIDEO_JOB_LOCK:
        _prune_video_jobs_locked()
        jobs = sorted(VIDEO_JOBS.values(), key=lambda item: item.get("createdMonotonic", 0.0), reverse=True)
        return {"jobs": [_job_public_snapshot(job) for job in jobs]}


@app.get("/api/jobs/{job_id}")
def get_video_job(job_id: str) -> dict:
    return _get_job_snapshot(job_id)


@app.delete("/api/jobs/{job_id}")
def cancel_video_job(job_id: str) -> dict:
    with VIDEO_JOB_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Video job not found: {job_id}")
        if job.get("status") in {"completed", "failed", "cancelled"}:
            return _job_public_snapshot(job)

        job["cancelRequested"] = True
        future = job.get("future")
        if job.get("status") == "queued" and future is not None and future.cancel():
            job.update(
                {
                    "status": "cancelled",
                    "progress": 0,
                    "stage": "cancelled",
                    "message": "Video analysis was cancelled while queued.",
                    "finishedAt": _now_iso(),
                    "finishedMonotonic": time.monotonic(),
                    "updatedAt": _now_iso(),
                }
            )
        else:
            job["message"] = "Cancellation requested. The running frame will finish before the job stops."
            job["updatedAt"] = _now_iso()
        return _job_public_snapshot(job)
