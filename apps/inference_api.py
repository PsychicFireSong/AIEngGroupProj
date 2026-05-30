from __future__ import annotations

import base64
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    Path("/content/drive/MyDrive/AIEngGroupProj_colab_outputs/weights"),
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


app = FastAPI(title="AI Facilities YOLO Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _is_likely_severity_model(path_value: str) -> bool:
    lowered = path_value.lower()
    return any(term in lowered for term in ("severity", "classify", "cls", "stage2"))


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
        if "/stage2_severity/weights/best.pt" in normalized:
            return (2, normalized)
        if "/stage2_severity/weights/last.pt" in normalized:
            return (3, normalized)
    if kind == "detector":
        if name == "defect_detector.pt" and "/weights/" in normalized:
            return (0, normalized)
        if "/stage1_defect_detector/weights/best.pt" in normalized:
            return (1, normalized)
        if normalized.endswith("/weights/best.pt"):
            return (2, normalized)
        if normalized.endswith("/best.pt"):
            return (3, normalized)
    return (9, normalized)


def _discover_model_paths() -> list[Path]:
    candidates: set[Path] = set()
    for root in MODEL_ROOTS:
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


def _name_from_result(result, class_id: int) -> str:
    names = getattr(result, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _classification_margin(probs) -> float:
    try:
        values = probs.top5conf
        if hasattr(values, "cpu"):
            values = values.cpu().numpy()
        values = [float(value) for value in values]
    except Exception:
        return float(probs.top1conf)
    if len(values) < 2:
        return values[0] if values else 0.0
    return values[0] - values[1]


def _classify_severity(
    model: YOLO,
    crop_bgr: np.ndarray,
    min_conf: float = 0.40,
    min_margin: float = 0.08,
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
    margin = _classification_margin(probs)
    if confidence < min_conf or margin < min_margin:
        return "uncertain", confidence
    if severity not in SEVERITY_WEIGHTS:
        return "unknown", confidence
    return severity, confidence


def _priority_for(severity: str, confidence: float, area_ratio: float) -> Literal["Immediate", "Planned", "Monitor"]:
    if severity == "critical" or (confidence >= 0.75 and area_ratio >= 0.12):
        return "Immediate"
    if severity == "moderate" or confidence >= 0.55:
        return "Planned"
    return "Monitor"


def _action_for(defect_class: str, severity: str, priority: str) -> str:
    if priority == "Immediate":
        return f"Restrict area if needed and raise urgent maintenance for {defect_class}."
    if severity == "moderate":
        return f"Schedule detailed inspection and repair planning for {defect_class}."
    if severity == "uncertain":
        return "Review manually before creating a work order."
    return f"Log and monitor {defect_class} in the next inspection cycle."


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


def _run_inference(
    image_bgr: np.ndarray,
    detector_path: str,
    severity_path: str,
    confidence: float,
    iou: float,
) -> dict:
    resolved_detector_path = _resolve_model_path(detector_path)
    resolved_severity_path = _resolve_model_path(severity_path)
    detector = _load_model(resolved_detector_path)
    severity_model = _load_model(resolved_severity_path)
    height, width = image_bgr.shape[:2]
    started = time.perf_counter()

    stage1_started = time.perf_counter()
    results = detector.predict(
        source=image_bgr,
        conf=confidence,
        iou=iou,
        imgsz=640,
        max_det=120,
        agnostic_nms=True,
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
            for index, coords in enumerate(xyxy):
                x1, y1, x2, y2 = coords.astype(int).tolist()
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(0, min(width, x2))
                y2 = max(0, min(height, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image_bgr[y1:y2, x1:x2]
                stage2_started = time.perf_counter()
                severity, severity_conf = _classify_severity(severity_model, crop)
                stage2_latency_ms += (time.perf_counter() - stage2_started) * 1000.0
                defect_conf = float(confidences[index])
                defect_class = _name_from_result(result, int(class_ids[index]))
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
                        "action": _action_for(defect_class, severity, priority),
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
    return {
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
        "annotatedImage": _encode_jpeg_data_url(annotated),
    }


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "device": "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu",
        "models": len(_discover_model_paths()),
    }


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
) -> dict:
    payload = await file.read()
    image = _decode_upload(payload)
    return _run_inference(image, detector_path, severity_path, confidence, iou)
