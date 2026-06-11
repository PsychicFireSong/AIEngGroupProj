"""
Local production sweep using val dataset images (no internet required).
Produces domain_sweep_summary.csv and comparison_summary.json compatible
with Cell 23 gate and Cell 25 report.
"""
from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
# threshold used as "primary" by gate
PRIMARY_THRESHOLD = 0.20
ALL_THRESHOLDS = [0.45, 0.30, 0.20, 0.10]

VAL_IMAGES = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\images\val")
VAL_LABELS = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\labels\val")

MODELS = {
    "current": Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\weights\defect_detector.pt"),
    "hn_weak": Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt"),
}
SEVERITY_MODEL = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls.pt")

OUTPUT_DIR = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\production_sweep")

SAMPLES_PER_CLASS = 12


def get_label_classes(label_path: Path) -> set[int]:
    classes = set()
    if not label_path.exists():
        return classes
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if parts:
            classes.add(int(parts[0]))
    return classes


def build_local_manifest(samples_per_class: int) -> list[dict]:
    per_class: dict[int, list[Path]] = {i: [] for i in range(len(CLASS_NAMES))}
    for img in sorted(VAL_IMAGES.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = VAL_LABELS / (img.stem + ".txt")
        classes = get_label_classes(lbl)
        for c in classes:
            if c < len(CLASS_NAMES):
                per_class[c].append(img)

    rng = random.Random(42)
    rows = []
    for class_id, images in per_class.items():
        sampled = rng.sample(images, min(samples_per_class, len(images)))
        for img in sampled:
            rows.append({
                "id": f"local_{CLASS_NAMES[class_id]}_{img.stem[:40]}",
                "domain_group": f"{CLASS_NAMES[class_id]}_local_val",
                "expected_class": CLASS_NAMES[class_id],
                "local_path": str(img),
                "image_url": "",
                "source_url": str(img),
                "notes": "local_val_image",
                "scenario_variant": "original",
            })
    return rows


def classify_crop(severity_model: YOLO, crop, device: str) -> tuple[str, float]:
    import numpy as np
    if crop is None or (hasattr(crop, "size") and crop.size == 0):
        return "unknown", 0.0
    results = severity_model.predict(source=crop, imgsz=224, device=device, verbose=False)
    if not results or getattr(results[0], "probs", None) is None:
        return "unknown", 0.0
    probs = results[0].probs
    idx = int(probs.top1)
    conf = float(probs.top1conf)
    names = results[0].names
    return names.get(idx, str(idx)), conf


def run_model_sweep(model_key: str, detector_path: Path, rows: list[dict], output_dir: Path) -> dict:
    import cv2
    print(f"\n=== Evaluating model: {model_key} ({detector_path.name}) ===")
    detector = YOLO(str(detector_path))
    severity_model = YOLO(str(SEVERITY_MODEL))
    device = "0" if __import__("torch").cuda.is_available() else "cpu"

    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    conf_floor = min(ALL_THRESHOLDS)
    result_rows = []
    total = len(rows)

    for idx, row in enumerate(rows, 1):
        print(f"[{model_key}] {idx}/{total} {row['id']}", flush=True)
        local_path = row.get("local_path", "").strip()
        if not local_path or not Path(local_path).exists():
            for threshold in ALL_THRESHOLDS:
                result_rows.append({**_skip_row(row, threshold), "notes": "missing_local_path"})
            continue

        t0 = time.perf_counter()
        try:
            results = detector.predict(
                source=local_path,
                conf=conf_floor,
                iou=0.45,
                imgsz=640,
                max_det=100,
                agnostic_nms=True,
                device=device,
                verbose=False,
            )
        except Exception as exc:
            for threshold in ALL_THRESHOLDS:
                result_rows.append({**_skip_row(row, threshold), "notes": f"error:{exc}"})
            continue

        latency_ms = (time.perf_counter() - t0) * 1000.0
        detections = []
        if results and getattr(results[0], "boxes", None) is not None and len(results[0].boxes) > 0:
            res = results[0]
            img = cv2.imread(local_path)
            h, w = (img.shape[:2] if img is not None else (640, 640))
            for coords, class_id, confidence in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int),
                res.boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = [int(v) for v in coords]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2] if img is not None else None
                severity, severity_conf = classify_crop(severity_model, crop, device)
                name = res.names.get(int(class_id), str(class_id))
                detections.append({
                    "class_name": name,
                    "confidence": float(confidence),
                    "severity": severity,
                    "severity_confidence": severity_conf,
                })

        for threshold in ALL_THRESHOLDS:
            filtered = [d for d in detections if d["confidence"] >= threshold]
            top = max(filtered, key=lambda d: d["confidence"], default=None)
            top_class = top["class_name"] if top else ""
            expected = row["expected_class"]
            is_match = expected in (top_class,) if top_class else False
            # handle multi-label
            expected_set = {e.strip() for e in expected.split("|")}
            is_match = bool(expected_set & {top_class}) if top_class else False
            if expected == "none":
                status = "false_positive" if filtered else "true_negative"
            elif is_match:
                status = "match"
            elif not filtered:
                status = "no_detection"
            else:
                status = f"wrong_{top_class}"
            result_rows.append({
                "id": row["id"],
                "domain_group": row["domain_group"],
                "expected_class": expected,
                "threshold": threshold,
                "detections": len(filtered),
                "top_class": top_class,
                "top_confidence": round(top["confidence"], 4) if top else "",
                "top_severity": top["severity"] if top else "",
                "top_severity_confidence": round(top["severity_confidence"], 4) if top else "",
                "match": str(is_match).lower(),
                "status": status,
                "scenario_variant": row.get("scenario_variant", "original"),
                "latency_ms": round(latency_ms, 2),
                "class_counts": json.dumps(dict(Counter(d["class_name"] for d in filtered)), sort_keys=True),
                "severity_counts": json.dumps(dict(Counter(d["severity"] for d in filtered)), sort_keys=True),
                "source_url": row.get("source_url", ""),
                "notes": row.get("notes", ""),
            })

    _write_csv(model_dir / "domain_sweep_summary.csv", result_rows)

    primary_rows = [r for r in result_rows if abs(float(r["threshold"]) - PRIMARY_THRESHOLD) < 1e-9 and r.get("match") != "skipped"]
    matches = sum(1 for r in primary_rows if r["status"] in {"match", "true_negative"})
    pass_rate = round(matches / len(primary_rows), 4) if primary_rows else 0
    summary = {
        "detector": str(detector_path),
        "severity": str(SEVERITY_MODEL),
        "rows": len(rows),
        "thresholds": ALL_THRESHOLDS,
        "primary_threshold": PRIMARY_THRESHOLD,
        "primary_rows": len(primary_rows),
        "matches_or_true_negatives": matches,
        "failure_rows": len(primary_rows) - matches,
        "pass_rate": pass_rate,
        "summary_csv": str(model_dir / "domain_sweep_summary.csv"),
    }
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{model_key}] pass_rate={pass_rate:.2%} ({matches}/{len(primary_rows)} primary rows)")
    return summary


def _skip_row(row: dict, threshold: float) -> dict:
    return {
        "id": row["id"],
        "domain_group": row["domain_group"],
        "expected_class": row["expected_class"],
        "threshold": threshold,
        "detections": 0,
        "top_class": "",
        "top_confidence": "",
        "top_severity": "",
        "top_severity_confidence": "",
        "match": "skipped",
        "status": "skipped",
        "scenario_variant": row.get("scenario_variant", "original"),
        "latency_ms": "",
        "class_counts": "{}",
        "severity_counts": "{}",
        "source_url": row.get("source_url", ""),
        "notes": row.get("notes", ""),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building local manifest from val images...")
    rows = build_local_manifest(SAMPLES_PER_CLASS)
    print(f"Manifest: {len(rows)} rows across {len(CLASS_NAMES)} classes")

    summaries = {}
    for model_key, detector_path in MODELS.items():
        if not detector_path.exists():
            print(f"Skipping {model_key}: missing {detector_path}")
            continue
        summaries[model_key] = run_model_sweep(model_key, detector_path, rows, OUTPUT_DIR)

    comparison_path = OUTPUT_DIR / "comparison_summary.json"
    comparison_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nComparison saved to: {comparison_path}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
