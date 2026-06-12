from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from ultralytics import YOLO

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


APP_ROOT = Path.cwd()
HISTORY_ROOT = APP_ROOT / "inspection_history"
ASSET_ROOT = HISTORY_ROOT / "assets"
DB_PATH = HISTORY_ROOT / "inspections.db"

SEVERITY_ORDER = ["minor", "moderate", "critical", "uncertain", "unknown"]
SEVERITY_COLORS = {
    "minor": (34, 177, 76),
    "moderate": (0, 176, 240),
    "critical": (220, 53, 69),
    "uncertain": (120, 120, 120),
    "unknown": (80, 80, 80),
}
SEVERITY_WEIGHTS = {
    "minor": 1.0,
    "moderate": 2.4,
    "critical": 5.0,
    "uncertain": 1.5,
    "unknown": 1.0,
}

BALANCED_INFERENCE_DEFAULTS = {
    "speed_preset": "Balanced",
    "device_choice": "auto",
    "conf": 0.45,
    "iou": 0.45,
    "detector_imgsz": 512,
    "severity_imgsz": 224,
    "max_det": 50,
    "agnostic_nms": True,
    "severity_conf": 0.45,
    "severity_margin": 0.10,
    "frame_stride": 8,
    "preview_every": 16,
    "max_video_seconds": 180,
    "live_preview": True,
}

PRESET_CONFIGS = {
    "Near real-time": {
        **BALANCED_INFERENCE_DEFAULTS,
        "speed_preset": "Near real-time",
        "detector_imgsz": 416,
        "severity_imgsz": 192,
        "frame_stride": 6,
        "max_det": 30,
        "preview_every": 12,
        "max_video_seconds": 90,
    },
    "Balanced": BALANCED_INFERENCE_DEFAULTS,
    "High accuracy": {
        **BALANCED_INFERENCE_DEFAULTS,
        "speed_preset": "High accuracy",
        "detector_imgsz": 640,
        "frame_stride": 3,
        "max_det": 100,
        "preview_every": 10,
        "max_video_seconds": 0,
    },
}


st.set_page_config(page_title="AI Facilities Defect Dashboard", layout="wide")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #07111f;
            color: #e8edf5;
        }
        header[data-testid="stHeader"] {
            background: #f8fafc;
            border-bottom: 1px solid #d9e2ef;
        }
        [data-testid="stSidebar"] {
            background: #091525;
            border-right: 1px solid #1d3048;
        }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #d8e4f2;
        }
        [data-testid="stSidebar"] h2 {
            color: #ffffff;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #0d1726;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        .dashboard-title {
            padding: 0.5rem 0 0.15rem 0;
        }
        .dashboard-title h1 {
            margin-bottom: 0;
            font-size: 2rem;
        }
        .dashboard-title p {
            color: #91a4bd;
            margin-top: 0.25rem;
        }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem 0;
        }
        .metric-card {
            border: 1px solid #1e3a5c;
            background: #0c1b2d;
            padding: 0.85rem;
            border-radius: 8px;
        }
        .metric-card:nth-child(1) {
            border-top: 3px solid #00b0f0;
        }
        .metric-card:nth-child(2) {
            border-top: 3px solid #f7c948;
        }
        .metric-card:nth-child(3) {
            border-top: 3px solid #dc3545;
        }
        .metric-card:nth-child(4) {
            border-top: 3px solid #22b14c;
        }
        .metric-card .label {
            color: #8ea5c0;
            font-size: 0.8rem;
        }
        .metric-card .value {
            color: #f6fbff;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .metric-card .caption {
            color: #6f86a2;
            font-size: 0.75rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #123552;
            color: #dcecff;
            border: 1px solid #24577e;
            font-size: 0.78rem;
            margin-right: 0.35rem;
        }
        .warning-panel {
            border: 1px solid #5c4420;
            background: #221a0c;
            padding: 0.75rem;
            border-radius: 8px;
            color: #ffdca3;
        }
        .evidence-shell {
            border: 1px solid #244466;
            background: #091827;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .evidence-title {
            color: #f5fbff;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .evidence-subtitle {
            color: #91a4bd;
            font-size: 0.85rem;
            margin-bottom: 0.75rem;
        }
        .priority-immediate {
            color: #ff8c8c;
            font-weight: 700;
        }
        .priority-planned {
            color: #ffd166;
            font-weight: 700;
        }
        .priority-monitor {
            color: #8ee6a3;
            font-weight: 700;
        }
        @media (max-width: 900px) {
            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_storage() -> None:
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                input_name TEXT NOT NULL,
                input_type TEXT NOT NULL,
                facility TEXT,
                asset TEXT,
                zone TEXT,
                inspector TEXT,
                detector_path TEXT,
                severity_path TEXT,
                media_path TEXT,
                annotated_path TEXT,
                total_detections INTEGER,
                critical_count INTEGER,
                risk_score REAL,
                condition_index REAL,
                notes TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                inspection_id TEXT NOT NULL,
                frame_index INTEGER,
                timestamp_sec REAL,
                defect_class TEXT,
                defect_conf REAL,
                severity TEXT,
                severity_conf REAL,
                severity_margin REAL,
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                width INTEGER,
                height INTEGER,
                area_ratio REAL,
                priority TEXT,
                action TEXT,
                crop_path TEXT,
                FOREIGN KEY (inspection_id) REFERENCES inspections (id)
            )
            """
        )


@st.cache_resource(show_spinner=False)
def _load_model(model_path: str) -> YOLO:
    model = YOLO(model_path)
    try:
        model.fuse()
    except Exception:
        pass
    return model


def _first_existing(paths: list[Path]) -> str:
    for path in paths:
        if path.exists():
            return str(path)
    return ""


def _default_detector_path() -> str:
    return _first_existing(
        [
            APP_ROOT / "best.pt",
            APP_ROOT / "runs" / "detect" / "stage1_defect_detector" / "weights" / "best.pt",
            Path("/content/drive/MyDrive/AIEngGroupProj_colab_outputs/weights/best.pt"),
        ]
    )


def _default_severity_path() -> str:
    return _first_existing(
        [
            APP_ROOT / "severity_cls.pt",
            APP_ROOT / "runs" / "classify" / "stage2_severity" / "weights" / "best.pt",
            Path("/content/drive/MyDrive/AIEngGroupProj_colab_outputs/weights/severity_cls.pt"),
        ]
    )


@st.cache_data(show_spinner=False)
def _discover_model_paths() -> list[str]:
    search_roots = [
        APP_ROOT,
        APP_ROOT / "weights",
        APP_ROOT / "models",
        APP_ROOT / "runs",
        Path("/content/drive/MyDrive/AIEngGroupProj_colab_outputs/weights"),
    ]
    candidates: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".pt":
            candidates.add(str(root))
            continue
        patterns = ["*.pt"] if root == APP_ROOT else ["**/*.pt"]
        for pattern in patterns:
            for model_path in root.glob(pattern):
                if model_path.is_file():
                    candidates.add(str(model_path))
    return sorted(candidates, key=lambda item: (0 if "severity" in item.lower() else 1, item.lower()))


def _is_likely_severity_model(path_value: str) -> bool:
    lowered = path_value.lower()
    return any(term in lowered for term in ("severity", "classify", "cls", "stage2"))


def _is_likely_detector_model(path_value: str) -> bool:
    lowered = path_value.lower()
    if _is_likely_severity_model(lowered):
        return False
    return any(term in lowered for term in ("detect", "detector", "stage1", "best.pt", "yolo"))


def _model_select_options(purpose: str) -> list[str]:
    paths = _discover_model_paths()
    if purpose == "severity":
        paths = [path for path in paths if _is_likely_severity_model(path)]
    elif purpose == "detector":
        paths = [path for path in paths if _is_likely_detector_model(path)]
    return paths + ["Custom path..."]


def _compact_model_label(path_value: str) -> str:
    if path_value == "Custom path...":
        return "Custom path..."
    path = Path(path_value)
    try:
        rel_path = path.relative_to(APP_ROOT)
    except ValueError:
        rel_path = path
    return str(rel_path)


def _choose_model_option(preferred: str, purpose: str) -> str:
    options = [option for option in _model_select_options(purpose) if option != "Custom path..."]
    if preferred and preferred in options:
        return preferred
    return options[0] if options else "Custom path..."


def _resolve_model_choice(selection: str, custom_path: str) -> str:
    if selection == "Custom path...":
        return custom_path.strip()
    return selection


def _device_options() -> list[str]:
    options = ["auto", "cpu"]
    if torch is not None and torch.cuda.is_available():
        options.insert(1, "cuda:0")
    return options


def _device_arg(device_choice: str) -> str | None:
    return None if device_choice == "auto" else device_choice


def _apply_sidebar_preset(preset_name: str = "Balanced") -> None:
    for key, value in PRESET_CONFIGS[preset_name].items():
        st.session_state[key] = value
    st.session_state["half"] = bool(torch is not None and torch.cuda.is_available())


def _apply_selected_sidebar_preset() -> None:
    _apply_sidebar_preset(st.session_state.get("speed_preset", "Balanced"))


def _ensure_sidebar_defaults() -> None:
    if "speed_preset" not in st.session_state:
        _apply_sidebar_preset("Balanced")
    device_options = _device_options()
    if st.session_state.get("device_choice") not in device_options:
        st.session_state["device_choice"] = "auto"
    st.session_state.setdefault("facility", "Main Building")
    st.session_state.setdefault("asset", "External Wall")
    st.session_state.setdefault("zone", "Level 1")
    st.session_state.setdefault("inspector", "AI-assisted inspection")
    st.session_state.setdefault("notes", "")
    st.session_state.setdefault("custom_detector_path", _default_detector_path())
    st.session_state.setdefault("custom_severity_path", _default_severity_path())


def _is_model_file(value: str) -> bool:
    if not value or not value.strip():
        return False
    path = Path(value)
    return path.exists() and path.is_file() and path.suffix.lower() == ".pt"


def _result_name(result, class_id: int) -> str:
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
    device: str | None,
    min_conf: float,
    min_margin: float,
    imgsz: int,
    half: bool,
) -> tuple[str, float, float]:
    if crop_bgr.size == 0:
        return "unknown", 0.0, 0.0

    results = model.predict(source=crop_bgr, imgsz=imgsz, half=half, device=device, verbose=False)
    if not results or getattr(results[0], "probs", None) is None:
        return "unknown", 0.0, 0.0

    result = results[0]
    probs = result.probs
    top1 = int(probs.top1)
    confidence = float(probs.top1conf)
    margin = _classification_margin(probs)
    severity = _result_name(result, top1)
    if confidence < min_conf or margin < min_margin:
        return "uncertain", confidence, margin
    return severity, confidence, margin


def _priority_for(severity: str, confidence: float, area_ratio: float) -> str:
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
    for det in detections:
        severity_weight = SEVERITY_WEIGHTS.get(det["severity"], 1.0)
        confidence_factor = max(float(det["defect_conf"]), 0.2)
        area_factor = 1.0 + min(float(det["area_ratio"]) * 12.0, 3.0)
        score += severity_weight * confidence_factor * area_factor * 4.0
    score = min(round(score, 1), 100.0)
    return score, round(max(0.0, 100.0 - score), 1)


def _run_detection(
    detector: YOLO,
    severity_model: YOLO,
    frame_bgr: np.ndarray,
    conf: float,
    iou: float,
    max_det: int,
    agnostic_nms: bool,
    severity_conf: float,
    severity_margin: float,
    detector_imgsz: int,
    severity_imgsz: int,
    half: bool,
    device: str | None,
    frame_index: int = 0,
    timestamp_sec: float = 0.0,
) -> list[dict]:
    results = detector.predict(
        source=frame_bgr,
        conf=conf,
        iou=iou,
        max_det=max_det,
        agnostic_nms=agnostic_nms,
        imgsz=detector_imgsz,
        half=half,
        device=device,
        verbose=False,
    )
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    height, width = frame_bgr.shape[:2]
    xyxy = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    detections: list[dict] = []

    for index, coords in enumerate(xyxy):
        x1, y1, x2, y2 = coords.astype(int).tolist()
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame_bgr[y1:y2, x1:x2]
        severity, severity_score, severity_delta = _classify_severity(
            severity_model, crop, device, severity_conf, severity_margin, severity_imgsz, half
        )
        defect_class = _result_name(result, int(class_ids[index]))
        defect_conf = float(confidences[index])
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)
        priority = _priority_for(severity, defect_conf, area_ratio)

        detections.append(
            {
                "id": str(uuid.uuid4()),
                "frame_index": frame_index,
                "timestamp_sec": round(timestamp_sec, 3),
                "defect_class": defect_class,
                "defect_conf": round(defect_conf, 4),
                "severity": severity,
                "severity_conf": round(severity_score, 4),
                "severity_margin": round(severity_delta, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "area_ratio": round(area_ratio, 6),
                "priority": priority,
                "action": _action_for(defect_class, severity, priority),
                "crop_bgr": crop.copy(),
            }
        )
    return detections


def _draw_detections(frame_bgr: np.ndarray, detections: list[dict], selected_id: str | None = None) -> np.ndarray:
    annotated = frame_bgr.copy()
    for det in detections:
        color = SEVERITY_COLORS.get(det["severity"], SEVERITY_COLORS["unknown"])
        thickness = 4 if det["id"] == selected_id else 2
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{det['defect_class']} {det['defect_conf']:.2f} | {det['severity']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        text_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        label_y1 = max(0, y1 - text_size[1] - 10)
        label_y2 = min(annotated.shape[0] - 1, label_y1 + text_size[1] + 8)
        label_x2 = min(annotated.shape[1] - 1, x1 + text_size[0] + 8)
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y2 - 5),
            font,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _detections_from_frame(detections: pd.DataFrame) -> list[dict]:
    if detections.empty:
        return []
    return detections.to_dict(orient="records")


def _heatmap_overlay(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    heat = np.zeros(frame_bgr.shape[:2], dtype=np.float32)
    for det in detections:
        weight = SEVERITY_WEIGHTS.get(det["severity"], 1.0) * max(det["defect_conf"], 0.2)
        cv2.rectangle(heat, (det["x1"], det["y1"]), (det["x2"], det["y2"]), weight, -1)
    if heat.max() <= 0:
        return frame_bgr.copy()
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=21, sigmaY=21)
    heat = np.uint8(255 * heat / heat.max())
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 0.62, colored, 0.38, 0)


def _stable_inspection_id(name: str, payload: bytes) -> str:
    digest = hashlib.sha1(payload + datetime.utcnow().isoformat().encode("utf-8")).hexdigest()[:12]
    stem = Path(name).stem[:32].replace(" ", "_")
    return f"{stem}_{digest}"


def _save_image(path: Path, image_bgr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)
    return str(path)


def _find_heatmap_path(inspection_id: str) -> str | None:
    asset_dir = ASSET_ROOT / inspection_id
    matches = sorted(asset_dir.glob("risk_heatmap_*.jpg"))
    if not matches:
        return None
    return str(matches[0])


def _save_detection_crops(inspection_id: str, detections: list[dict]) -> None:
    crop_dir = ASSET_ROOT / inspection_id / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for index, det in enumerate(detections, start=1):
        crop_path = crop_dir / f"{index:03d}_{det['defect_class']}_{det['severity']}.jpg"
        cv2.imwrite(str(crop_path), det["crop_bgr"])
        det["crop_path"] = str(crop_path)


def _record_inspection(
    inspection_id: str,
    input_name: str,
    input_type: str,
    facility: str,
    asset: str,
    zone: str,
    inspector: str,
    detector_path: str,
    severity_path: str,
    media_path: str,
    annotated_path: str,
    detections: list[dict],
    notes: str,
) -> None:
    risk, condition = _risk_score(detections)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO inspections (
                id, created_at, input_name, input_type, facility, asset, zone, inspector,
                detector_path, severity_path, media_path, annotated_path, total_detections,
                critical_count, risk_score, condition_index, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inspection_id,
                datetime.utcnow().isoformat(timespec="seconds"),
                input_name,
                input_type,
                facility,
                asset,
                zone,
                inspector,
                detector_path,
                severity_path,
                media_path,
                annotated_path,
                len(detections),
                sum(1 for det in detections if det["severity"] == "critical"),
                risk,
                condition,
                notes,
            ),
        )
        conn.execute("DELETE FROM detections WHERE inspection_id = ?", (inspection_id,))
        for det in detections:
            conn.execute(
                """
                INSERT INTO detections (
                    id, inspection_id, frame_index, timestamp_sec, defect_class, defect_conf,
                    severity, severity_conf, severity_margin, x1, y1, x2, y2, width, height,
                    area_ratio, priority, action, crop_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    det["id"],
                    inspection_id,
                    det["frame_index"],
                    det["timestamp_sec"],
                    det["defect_class"],
                    det["defect_conf"],
                    det["severity"],
                    det["severity_conf"],
                    det["severity_margin"],
                    det["x1"],
                    det["y1"],
                    det["x2"],
                    det["y2"],
                    det["width"],
                    det["height"],
                    det["area_ratio"],
                    det["priority"],
                    det["action"],
                    det.get("crop_path", ""),
                ),
            )


def _load_inspections() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM inspections ORDER BY created_at DESC", conn)


def _load_detections(inspection_id: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM detections"
    params: tuple[str, ...] = ()
    if inspection_id:
        query += " WHERE inspection_id = ?"
        params = (inspection_id,)
    query += " ORDER BY frame_index, timestamp_sec, severity DESC"
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)


def _render_metric_strip(inspections: pd.DataFrame, detections: pd.DataFrame) -> None:
    total_inspections = len(inspections)
    total_detections = len(detections)
    critical_count = int((detections["severity"] == "critical").sum()) if not detections.empty else 0
    avg_condition = inspections["condition_index"].mean() if not inspections.empty else 100.0
    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-card"><div class="label">Inspections</div><div class="value">{total_inspections}</div><div class="caption">historical sessions</div></div>
            <div class="metric-card"><div class="label">Detected Defects</div><div class="value">{total_detections}</div><div class="caption">image and video records</div></div>
            <div class="metric-card"><div class="label">Critical Findings</div><div class="value">{critical_count}</div><div class="caption">immediate-priority candidates</div></div>
            <div class="metric-card"><div class="label">Avg Condition Index</div><div class="value">{avg_condition:.1f}</div><div class="caption">100 means no detected risk</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _detection_table(detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty:
        return detections
    columns = [
        "id",
        "frame_index",
        "timestamp_sec",
        "defect_class",
        "defect_conf",
        "severity",
        "severity_conf",
        "severity_margin",
        "area_ratio",
        "priority",
        "action",
    ]
    return detections[[column for column in columns if column in detections.columns]].copy()


def _render_analytics(inspections: pd.DataFrame, detections: pd.DataFrame) -> None:
    if inspections.empty:
        st.info("No inspection history yet. Run an image or video inspection first.")
        return

    left, right = st.columns(2)
    if not detections.empty:
        severity_counts = detections["severity"].value_counts().reindex(SEVERITY_ORDER).dropna().reset_index()
        severity_counts.columns = ["severity", "count"]
        fig = px.pie(
            severity_counts,
            names="severity",
            values="count",
            hole=0.55,
            color="severity",
            color_discrete_map={
                "minor": "#22b14c",
                "moderate": "#00b0f0",
                "critical": "#dc3545",
                "uncertain": "#777777",
                "unknown": "#444444",
            },
        )
        fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), height=320)
        left.plotly_chart(fig, use_container_width=True)

        defect_counts = detections["defect_class"].value_counts().reset_index()
        defect_counts.columns = ["defect_class", "count"]
        fig = px.bar(defect_counts, x="defect_class", y="count", color="defect_class")
        fig.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=30, b=40), height=320)
        right.plotly_chart(fig, use_container_width=True)
    else:
        left.info("No detections recorded yet.")
        right.info("No detections recorded yet.")

    trend = inspections.copy()
    trend["created_at"] = pd.to_datetime(trend["created_at"], errors="coerce")
    trend = trend.dropna(subset=["created_at"]).sort_values("created_at")
    if not trend.empty:
        fig = px.line(
            trend,
            x="created_at",
            y=["risk_score", "condition_index"],
            markers=True,
            labels={"value": "score", "created_at": "inspection time", "variable": "metric"},
        )
        fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=40), height=340)
        st.plotly_chart(fig, use_container_width=True)


def _render_detection_review(detections: pd.DataFrame, annotated_path: str | None = None) -> None:
    if detections.empty:
        st.info("No detections available for review.")
        return

    table = _detection_table(detections)
    selected = st.selectbox(
        "Detection record",
        table["id"].tolist(),
        format_func=lambda det_id: _format_detection_option(table, det_id),
    )
    det_row = table[table["id"] == selected].iloc[0].to_dict()

    media_col, detail_col = st.columns([1.25, 1])
    with media_col:
        if annotated_path and Path(annotated_path).exists():
            suffix = Path(annotated_path).suffix.lower()
            if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
                st.video(Path(annotated_path).read_bytes())
            else:
                st.image(str(annotated_path), caption="Annotated inspection result", use_container_width=True)
        crop_path = detections[detections["id"] == selected]["crop_path"].iloc[0]
        if crop_path and Path(crop_path).exists():
            st.image(str(crop_path), caption="Selected defect crop", use_container_width=True)

    with detail_col:
        st.markdown(f"**Defect:** `{det_row['defect_class']}`")
        st.markdown(f"**Severity:** `{det_row['severity']}`")
        st.markdown(f"**Priority:** `{det_row['priority']}`")
        st.markdown(f"**Detector confidence:** `{det_row['defect_conf']:.3f}`")
        st.markdown(f"**Severity confidence:** `{det_row['severity_conf']:.3f}`")
        st.markdown(f"**Image area affected:** `{det_row['area_ratio'] * 100:.2f}%`")
        st.markdown(f"**Recommended action:** {det_row['action']}")
        st.json(
            {
                "frame_index": int(det_row["frame_index"]),
                "timestamp_sec": float(det_row["timestamp_sec"]),
                "bbox_xyxy": [int(det_row["x1"]), int(det_row["y1"]), int(det_row["x2"]), int(det_row["y2"])],
            },
            expanded=False,
        )

    st.dataframe(table.drop(columns=["id"]), use_container_width=True, hide_index=True)
    st.download_button(
        "Download detection records CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="detection_records.csv",
        mime="text/csv",
    )


def _render_visual_evidence(inspection: pd.Series, detections: pd.DataFrame) -> None:
    inspection_id = str(inspection["id"])
    media_path = str(inspection.get("media_path", "") or "")
    annotated_path = str(inspection.get("annotated_path", "") or "")
    heatmap_path = _find_heatmap_path(inspection_id)
    input_type = str(inspection.get("input_type", "") or "")

    st.markdown(
        f"""
        <div class="evidence-shell">
          <div class="evidence-title">Visual Evidence: {inspection.get('input_name', 'Inspection')}</div>
          <div class="evidence-subtitle">
            {inspection.get('facility', '')} / {inspection.get('asset', '')} / {inspection.get('zone', '')}
            &nbsp; | &nbsp; Risk {float(inspection.get('risk_score', 0.0)):.1f}
            &nbsp; | &nbsp; Condition {float(inspection.get('condition_index', 100.0)):.1f}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    visual_col, detail_col = st.columns([1.55, 1.0])
    with visual_col:
        if input_type == "video":
            modes = ["Annotated Video", "Original Video"]
            mode = st.radio("Evidence view", modes, horizontal=True, label_visibility="collapsed")
            selected_path = annotated_path if mode == "Annotated Video" else media_path
            if selected_path and Path(selected_path).exists():
                st.video(Path(selected_path).read_bytes())
            else:
                st.warning("Video evidence file is missing from local history storage.")
        else:
            modes = ["Annotated", "Original", "Risk Heatmap"]
            if not detections.empty:
                modes.append("Focused Defect")
            mode = st.radio("Evidence view", modes, horizontal=True, label_visibility="collapsed")

            selected_id = None
            if mode == "Focused Defect" and not detections.empty:
                selected_id = st.session_state.get("selected_detection_id")
                if not selected_id or selected_id not in set(detections["id"].tolist()):
                    selected_id = detections.iloc[0]["id"]
                image = cv2.imread(media_path) if media_path and Path(media_path).exists() else None
                if image is not None:
                    focused = _draw_detections(image, _detections_from_frame(detections), selected_id=selected_id)
                    st.image(focused, channels="BGR", caption="Focused selected defect", use_container_width=True)
                else:
                    st.warning("Original image file is missing from local history storage.")
            elif mode == "Original":
                if media_path and Path(media_path).exists():
                    st.image(media_path, caption="Original visual evidence", use_container_width=True)
                else:
                    st.warning("Original image file is missing from local history storage.")
            elif mode == "Risk Heatmap":
                if heatmap_path and Path(heatmap_path).exists():
                    st.image(heatmap_path, caption="Risk heatmap overlay", use_container_width=True)
                else:
                    st.info("Risk heatmap is available for image inspections after detections are generated.")
            else:
                if annotated_path and Path(annotated_path).exists():
                    st.image(annotated_path, caption="Annotated visual evidence", use_container_width=True)
                else:
                    st.warning("Annotated image file is missing from local history storage.")

    with detail_col:
        if detections.empty:
            st.info("No defects detected for this inspection at the current thresholds.")
            return

        table = _detection_table(detections)
        selected_id = st.selectbox(
            "Select defect evidence",
            table["id"].tolist(),
            key=f"evidence_select_{inspection_id}",
            format_func=lambda det_id: _format_detection_option(table, det_id),
        )
        st.session_state["selected_detection_id"] = selected_id
        selected = detections[detections["id"] == selected_id].iloc[0]

        crop_path = selected.get("crop_path", "")
        if crop_path and Path(str(crop_path)).exists():
            st.image(str(crop_path), caption="Defect crop sent to severity classifier", use_container_width=True)

        priority_class = {
            "Immediate": "priority-immediate",
            "Planned": "priority-planned",
            "Monitor": "priority-monitor",
        }.get(str(selected["priority"]), "priority-monitor")
        st.markdown(
            f"""
            <div>
              <div><strong>Defect:</strong> {selected['defect_class']}</div>
              <div><strong>Severity:</strong> {selected['severity']} ({float(selected['severity_conf']):.2f})</div>
              <div><strong>Detector confidence:</strong> {float(selected['defect_conf']):.2f}</div>
              <div><strong>Area affected:</strong> {float(selected['area_ratio']) * 100:.2f}%</div>
              <div><strong>Priority:</strong> <span class="{priority_class}">{selected['priority']}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"**Recommended action:** {selected['action']}")
        st.dataframe(table.drop(columns=["id"]), use_container_width=True, hide_index=True, height=260)
        st.download_button(
            "Download current detection CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"{inspection_id}_detections.csv",
            mime="text/csv",
        )


def _format_detection_option(table: pd.DataFrame, det_id: str) -> str:
    row = table[table["id"] == det_id].iloc[0]
    return (
        f"{row['frame_index']} | {row['defect_class']} | {row['severity']} | "
        f"{row['priority']} | conf {row['defect_conf']:.2f}"
    )


def _process_image_upload(
    uploaded_file,
    detector: YOLO,
    severity_model: YOLO,
    settings: dict,
    context: dict,
) -> str:
    payload = uploaded_file.getvalue()
    inspection_id = _stable_inspection_id(uploaded_file.name, payload)
    image = Image.open(uploaded_file).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    detections = _run_detection(
        detector,
        severity_model,
        frame_bgr,
        settings["conf"],
        settings["iou"],
        settings["max_det"],
        settings["agnostic_nms"],
        settings["severity_conf"],
        settings["severity_margin"],
        settings["detector_imgsz"],
        settings["severity_imgsz"],
        settings["half"],
        settings["device"],
    )
    _save_detection_crops(inspection_id, detections)
    annotated = _draw_detections(frame_bgr, detections)
    heatmap = _heatmap_overlay(frame_bgr, detections)

    asset_dir = ASSET_ROOT / inspection_id
    media_path = _save_image(asset_dir / f"input_{uploaded_file.name}", frame_bgr)
    annotated_path = _save_image(asset_dir / f"annotated_{Path(uploaded_file.name).stem}.jpg", annotated)
    _save_image(asset_dir / f"risk_heatmap_{Path(uploaded_file.name).stem}.jpg", heatmap)

    _record_inspection(
        inspection_id,
        uploaded_file.name,
        "image",
        context["facility"],
        context["asset"],
        context["zone"],
        context["inspector"],
        settings["detector_path"],
        settings["severity_path"],
        media_path,
        annotated_path,
        detections,
        context["notes"],
    )
    return inspection_id


def _process_video_upload(
    uploaded_file,
    detector: YOLO,
    severity_model: YOLO,
    settings: dict,
    context: dict,
) -> str:
    payload = uploaded_file.getvalue()
    inspection_id = _stable_inspection_id(uploaded_file.name, payload)
    asset_dir = ASSET_ROOT / inspection_id
    asset_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / uploaded_file.name
        input_path.write_bytes(payload)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Unable to open uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        annotated_path = asset_dir / f"annotated_{Path(uploaded_file.name).stem}.mp4"
        input_archive_path = asset_dir / uploaded_file.name
        input_archive_path.write_bytes(payload)

        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        detections: list[dict] = []
        progress = st.progress(0)
        preview_slot = st.empty()
        status_slot = st.empty()
        frame_index = 0
        inference_frame_count = 0
        latest_frame_detections: list[dict] = []
        started_at = time.perf_counter()
        max_frames = total
        if settings["max_video_seconds"] > 0:
            max_frames = min(total or int(settings["max_video_seconds"] * fps), int(settings["max_video_seconds"] * fps))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_index >= max_frames:
                break

            if frame_index % settings["frame_stride"] == 0:
                inference_started = time.perf_counter()
                frame_detections = _run_detection(
                    detector,
                    severity_model,
                    frame,
                    settings["conf"],
                    settings["iou"],
                    settings["max_det"],
                    settings["agnostic_nms"],
                    settings["severity_conf"],
                    settings["severity_margin"],
                    settings["detector_imgsz"],
                    settings["severity_imgsz"],
                    settings["half"],
                    settings["device"],
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                )
                latest_frame_detections = frame_detections
                detections.extend(frame_detections)
                inference_frame_count += 1
                last_latency_ms = (time.perf_counter() - inference_started) * 1000.0
            else:
                last_latency_ms = None

            annotated_frame = _draw_detections(frame, latest_frame_detections)
            writer.write(annotated_frame)
            frame_index += 1

            denominator = max_frames or total
            if denominator:
                progress.progress(min(frame_index / denominator, 1.0))
            if settings["live_preview"] and frame_index % settings["preview_every"] == 0:
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                processing_fps = frame_index / elapsed
                status_text = (
                    f"Processed {frame_index} frames"
                    f" | inference frames {inference_frame_count}"
                    f" | processing FPS {processing_fps:.1f}"
                    f" | detections {len(detections)}"
                )
                if last_latency_ms is not None:
                    status_text += f" | last inference {last_latency_ms:.0f} ms"
                status_slot.info(status_text)
                preview_slot.image(
                    annotated_frame,
                    channels="BGR",
                    caption="Live processing preview",
                    use_container_width=True,
                )

        cap.release()
        writer.release()
        progress.progress(1.0)
        elapsed = max(time.perf_counter() - started_at, 1e-6)
        status_slot.success(
            f"Video processed in {elapsed:.1f}s at {frame_index / elapsed:.1f} FPS "
            f"with {inference_frame_count} inference frames."
        )

    _save_detection_crops(inspection_id, detections)
    _record_inspection(
        inspection_id,
        uploaded_file.name,
        "video",
        context["facility"],
        context["asset"],
        context["zone"],
        context["inspector"],
        settings["detector_path"],
        settings["severity_path"],
        str(input_archive_path),
        str(annotated_path),
        detections,
        context["notes"],
    )
    return inspection_id


def _load_ready_models(settings: dict) -> tuple[YOLO | None, YOLO | None]:
    if not _is_model_file(settings["detector_path"]) or not _is_model_file(settings["severity_path"]):
        return None, None
    if torch is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    return _load_model(settings["detector_path"]), _load_model(settings["severity_path"])


def _sidebar() -> tuple[dict, dict]:
    _ensure_sidebar_defaults()
    detector_options = _model_select_options("detector")
    severity_options = _model_select_options("severity")
    default_detector = _choose_model_option(_default_detector_path(), "detector")
    default_severity = _choose_model_option(_default_severity_path(), "severity")
    if st.session_state.get("detector_model_choice") not in detector_options:
        st.session_state["detector_model_choice"] = default_detector
    if st.session_state.get("severity_model_choice") not in severity_options:
        st.session_state["severity_model_choice"] = default_severity

    with st.sidebar:
        st.header("Inspection Controls")

        detector_choice = st.selectbox(
            "Detector model",
            detector_options,
            key="detector_model_choice",
            format_func=_compact_model_label,
            help="Available .pt models found in the project, runs, models, weights, and Colab Drive output folders.",
        )
        severity_choice = st.selectbox(
            "Severity model",
            severity_options,
            key="severity_model_choice",
            format_func=_compact_model_label,
            help="Available .pt models found in the same known locations.",
        )

        with st.expander("Custom model paths", expanded=False):
            custom_detector_path = st.text_input(
                "Detector .pt path",
                key="custom_detector_path",
                placeholder="best.pt",
            )
            custom_severity_path = st.text_input(
                "Severity .pt path",
                key="custom_severity_path",
                placeholder="severity_cls.pt",
            )

        detector_path = _resolve_model_choice(detector_choice, custom_detector_path)
        severity_path = _resolve_model_choice(severity_choice, custom_severity_path)

        st.divider()
        if st.button("Recommended balanced settings", type="primary", use_container_width=True):
            _apply_sidebar_preset("Balanced")
            st.rerun()

        st.selectbox(
            "Inference mode",
            ["Near real-time", "Balanced", "High accuracy"],
            key="speed_preset",
            on_change=_apply_selected_sidebar_preset,
            help="Presets tune input size, video stride, and preview cadence.",
        )

        with st.expander("Advanced inference tuning", expanded=False):
            device_choice = st.selectbox("Device", _device_options(), key="device_choice")
            conf = st.slider("Detector confidence", 0.05, 0.95, step=0.05, key="conf")
            iou = st.slider("NMS IoU", 0.10, 0.90, step=0.05, key="iou")
            detector_imgsz = st.select_slider(
                "Detector image size",
                options=[320, 416, 512, 640],
                key="detector_imgsz",
            )
            severity_imgsz = st.select_slider(
                "Severity image size",
                options=[160, 192, 224],
                key="severity_imgsz",
            )
            max_det = st.slider("Max detections", 1, 200, step=1, key="max_det")
            agnostic_nms = st.toggle("Class-agnostic NMS", key="agnostic_nms")
            severity_conf = st.slider("Severity confidence", 0.05, 0.95, step=0.05, key="severity_conf")
            severity_margin = st.slider("Severity top-1 margin", 0.00, 0.50, step=0.05, key="severity_margin")
            frame_stride = st.slider("Video frame stride", 1, 30, step=1, key="frame_stride")
            preview_every = st.slider("Live preview cadence", 1, 60, step=1, key="preview_every")
            max_video_seconds = st.slider(
                "Max video seconds",
                0,
                600,
                step=10,
                key="max_video_seconds",
                help="Use 0 to process the full video. Limit this for quick field testing.",
            )
            live_preview = st.toggle("Live video preview", key="live_preview")
            half = st.toggle(
                "GPU half precision",
                key="half",
                help="Uses FP16 on CUDA for faster inference. It is ignored by Ultralytics on unsupported devices.",
            )

        with st.expander("Inspection details", expanded=False):
            facility = st.text_input("Facility", key="facility")
            asset = st.text_input("Asset / Block", key="asset")
            zone = st.text_input("Zone / Level", key="zone")
            inspector = st.text_input("Inspector", key="inspector")
            notes = st.text_area("Notes", key="notes", height=80)

    settings = {
        "detector_path": detector_path,
        "severity_path": severity_path,
        "device": _device_arg(st.session_state["device_choice"]),
        "conf": st.session_state["conf"],
        "iou": st.session_state["iou"],
        "detector_imgsz": st.session_state["detector_imgsz"],
        "severity_imgsz": st.session_state["severity_imgsz"],
        "max_det": st.session_state["max_det"],
        "agnostic_nms": st.session_state["agnostic_nms"],
        "severity_conf": st.session_state["severity_conf"],
        "severity_margin": st.session_state["severity_margin"],
        "frame_stride": st.session_state["frame_stride"],
        "preview_every": st.session_state["preview_every"],
        "max_video_seconds": st.session_state["max_video_seconds"],
        "live_preview": st.session_state["live_preview"],
        "half": st.session_state["half"] and _device_arg(st.session_state["device_choice"]) != "cpu",
    }
    context = {
        "facility": st.session_state["facility"],
        "asset": st.session_state["asset"],
        "zone": st.session_state["zone"],
        "inspector": st.session_state["inspector"],
        "notes": st.session_state["notes"],
    }
    return settings, context


def _render_header() -> None:
    st.markdown(
        """
        <div class="dashboard-title">
          <h1>AI Facilities Management Dashboard</h1>
          <p>Structural condition monitoring, defect detection, severity triage, and inspection history.</p>
        </div>
        <span class="status-pill">Object Detection</span>
        <span class="status-pill">Severity Classification</span>
        <span class="status-pill">Historical Analytics</span>
        <span class="status-pill">Inspection Review</span>
        """,
        unsafe_allow_html=True,
    )


def _show_model_status(settings: dict) -> None:
    detector_exists = _is_model_file(settings["detector_path"])
    severity_exists = _is_model_file(settings["severity_path"])
    if detector_exists and severity_exists:
        return
    st.markdown(
        """
        <div class="warning-panel">
        Model files are not both available yet. The dashboard and history can still open, but live inference requires
        both the Stage 1 detector and Stage 2 severity classifier selected in the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_css()
    _ensure_storage()
    settings, context = _sidebar()
    _render_header()
    _show_model_status(settings)

    inspections = _load_inspections()
    all_detections = _load_detections()
    _render_metric_strip(inspections, all_detections)

    current_id = st.session_state.get("current_inspection_id")
    if not current_id and not inspections.empty:
        current_id = inspections.iloc[0]["id"]

    if current_id and current_id in inspections["id"].tolist():
        current_row = inspections[inspections["id"] == current_id].iloc[0]
        current_detections = _load_detections(current_id)
        _render_visual_evidence(current_row, current_detections)
    else:
        st.markdown(
            """
            <div class="evidence-shell">
              <div class="evidence-title">Visual Evidence Appears Here</div>
              <div class="evidence-subtitle">
                Upload an image or video in Live Inspection to generate annotated evidence,
                defect crops, severity labels, and historical records.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    tab_live, tab_review, tab_analytics, tab_history = st.tabs(
        ["Live Inspection", "Evidence Archive", "Analytics", "History"]
    )

    with tab_live:
        detector, severity_model = _load_ready_models(settings)
        if detector is None or severity_model is None:
            st.info("Add valid model paths in the sidebar to run live image or video inspection.")
        image_files = st.file_uploader(
            "Image inspection",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
        run_images = st.button(
            "Run image inspection",
            disabled=not image_files or detector is None or severity_model is None,
        )

        if run_images and detector is not None and severity_model is not None:
            created_ids = []
            with st.spinner("Inspecting uploaded images..."):
                for image_file in image_files:
                    created_ids.append(_process_image_upload(image_file, detector, severity_model, settings, context))
            st.session_state["current_inspection_id"] = created_ids[-1]
            st.success(f"Saved {len(created_ids)} image inspection record(s).")
            st.rerun()

        video_file = st.file_uploader(
            "Video inspection",
            type=["mp4", "mov", "avi", "mkv"],
            accept_multiple_files=False,
        )
        run_video = st.button(
            "Run video inspection",
            disabled=video_file is None or detector is None or severity_model is None,
        )
        if run_video and detector is not None and severity_model is not None and video_file is not None:
            with st.spinner("Inspecting video frames..."):
                inspection_id = _process_video_upload(video_file, detector, severity_model, settings, context)
            st.session_state["current_inspection_id"] = inspection_id
            st.success("Saved video inspection record.")
            st.rerun()

    inspections = _load_inspections()
    current_id = st.session_state.get("current_inspection_id")
    if not current_id and not inspections.empty:
        current_id = inspections.iloc[0]["id"]

    with tab_review:
        if current_id:
            selected_inspection = st.selectbox(
                "Inspection session",
                inspections["id"].tolist(),
                index=inspections["id"].tolist().index(current_id) if current_id in inspections["id"].tolist() else 0,
                format_func=lambda item: _format_inspection_option(inspections, item),
            )
            st.session_state["current_inspection_id"] = selected_inspection
            inspection_row = inspections[inspections["id"] == selected_inspection].iloc[0]
            detections = _load_detections(selected_inspection)
            _render_visual_evidence(inspection_row, detections)
        else:
            st.info("No inspection selected yet.")

    with tab_analytics:
        _render_analytics(inspections, all_detections)

    with tab_history:
        if inspections.empty:
            st.info("No historical inspections saved yet.")
        else:
            st.dataframe(inspections.drop(columns=["detector_path", "severity_path"]), use_container_width=True, hide_index=True)
            export = {
                "inspections": inspections.to_dict(orient="records"),
                "detections": all_detections.to_dict(orient="records"),
            }
            st.download_button(
                "Download full inspection history JSON",
                data=json.dumps(export, indent=2).encode("utf-8"),
                file_name="inspection_history.json",
                mime="application/json",
            )


def _format_inspection_option(inspections: pd.DataFrame, inspection_id: str) -> str:
    row = inspections[inspections["id"] == inspection_id].iloc[0]
    return (
        f"{row['created_at']} | {row['input_name']} | defects {row['total_detections']} | "
        f"risk {row['risk_score']:.1f}"
    )


if __name__ == "__main__":
    main()
