from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import cv2
import numpy as np
import requests
from ultralytics import YOLO

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def device_arg() -> str | None:
    if torch is not None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return "cuda:0"
    return None


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_name(value: str) -> str:
    cleaned = []
    for char in value:
        cleaned.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(cleaned).strip("_").lower()


def filename_from_url(url: str, fallback: str) -> str:
    parsed = urlparse(url)
    name = unquote(Path(parsed.path).name)
    if not name or "." not in name:
        name = f"{fallback}.jpg"
    return name


def wikimedia_thumbnail_url(url: str, width: int = 1280) -> str:
    """Use Wikimedia's thumbnail redirect to avoid repeatedly requesting full-size originals."""
    parsed = urlparse(url)
    if "wikimedia.org" not in parsed.netloc.lower():
        return url
    marker = "/wiki/Special:FilePath/"
    if marker not in parsed.path:
        return url
    filename = unquote(parsed.path.split(marker, 1)[1])
    if not filename:
        return url
    return f"https://commons.wikimedia.org/wiki/Special:Redirect/file/{quote(filename, safe='')}?width={width}"


def download_candidates(url: str) -> list[str]:
    thumbnail_url = wikimedia_thumbnail_url(url)
    candidates = [thumbnail_url, url] if thumbnail_url != url else [url]
    return list(dict.fromkeys(candidates))


def valid_image_bytes(content: bytes) -> bool:
    if not content:
        return False
    array = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR) is not None


def ensure_image(row: dict, image_dir: Path, cache_dir: Path | None = None) -> Path:
    local_path = row.get("local_path", "").strip()
    if local_path:
        path = Path(local_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Local image does not exist for {row['id']}: {path}")

    image_url = row.get("image_url", "").strip()
    if not image_url:
        raise ValueError(f"No local_path or image_url configured for {row['id']}")

    image_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = image_dir / f"{safe_name(row['id'])}_{filename_from_url(image_url, row['id'])}"
    cache_path = (cache_dir / output_path.name) if cache_dir is not None else None
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    if cache_path is not None and cache_path.exists() and cache_path.stat().st_size > 0:
        output_path.write_bytes(cache_path.read_bytes())
        return output_path

    last_error: Exception | None = None
    headers = {
        "User-Agent": "AIEngGroupProj-domain-sweep/1.1 (student evaluation; cached; contact: local-notebook)",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    for candidate_url in download_candidates(image_url):
        for attempt in range(5):
            try:
                response = requests.get(
                    candidate_url,
                    timeout=60,
                    headers=headers,
                    allow_redirects=True,
                )
                response.raise_for_status()
                if not valid_image_bytes(response.content):
                    raise RuntimeError(f"Downloaded content is not a readable image: {candidate_url}")
                output_path.write_bytes(response.content)
                if cache_path is not None:
                    cache_path.write_bytes(response.content)
                time.sleep(2.0)
                return output_path
            except requests.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code not in {429, 500, 502, 503, 504}:
                    raise
                retry_after = 0
                if exc.response is not None:
                    try:
                        retry_after = int(exc.response.headers.get("Retry-After", "0"))
                    except ValueError:
                        retry_after = 0
                sleep_seconds = max(retry_after, min(90, 8 * (2**attempt)))
                time.sleep(sleep_seconds)
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
                time.sleep(min(60, 6 * (attempt + 1)))
    if last_error is not None:
        raise last_error
    return output_path


def model_name(result, class_id: int) -> str:
    names = getattr(result, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def classify_crop(model: YOLO, crop: np.ndarray, device: str | None) -> tuple[str, float]:
    if crop.size == 0:
        return "unknown", 0.0
    results = model.predict(source=crop, imgsz=224, device=device, verbose=False)
    if not results or getattr(results[0], "probs", None) is None:
        return "unknown", 0.0
    result = results[0]
    probs = result.probs
    return model_name(result, int(probs.top1)), float(probs.top1conf)


def expected_match(expected: str, predicted: str | None) -> bool:
    expected = (expected or "").strip().lower()
    if expected == "none":
        return predicted is None
    allowed = {item.strip().lower() for item in expected.split("|") if item.strip()}
    return bool(predicted and predicted.lower() in allowed)


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["xyxy"]
        label = f"{detection['class_name']} {detection['confidence']:.2f} | {detection['severity']}"
        color = (20, 220, 240)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(output, (x1, max(0, y1 - 24)), (x1 + min(360, 8 + len(label) * 8), y1), color, -1)
        cv2.putText(output, label, (x1 + 4, max(14, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (8, 12, 18), 1, cv2.LINE_AA)
    return output


def run_one(
    detector: YOLO,
    severity_model: YOLO,
    image_path: Path,
    conf: float,
    iou: float,
    device: str | None,
) -> tuple[list[dict], float]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    started = time.perf_counter()
    results = detector.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=640,
        max_det=100,
        agnostic_nms=True,
        device=device,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    detections: list[dict] = []
    if not results:
        return detections, latency_ms
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections, latency_ms

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
        crop = image[y1:y2, x1:x2]
        severity, severity_conf = classify_crop(severity_model, crop, device)
        detections.append(
            {
                "class_name": model_name(result, int(class_ids[index])),
                "confidence": float(confidences[index]),
                "severity": severity,
                "severity_confidence": severity_conf,
                "xyxy": [x1, y1, x2, y2],
                "area_ratio": ((x2 - x1) * (y2 - y1)) / float(width * height),
            }
        )
    return detections, latency_ms


def summarize_rows(rows: list[dict], primary_conf: float) -> dict:
    primary = []
    for row in rows:
        try:
            threshold = float(row["threshold"])
        except (TypeError, ValueError):
            continue
        if abs(threshold - primary_conf) < 1e-9:
            primary.append(row)
    by_domain: dict[str, Counter] = {}
    for row in primary:
        domain = row["domain_group"]
        by_domain.setdefault(domain, Counter())
        by_domain[domain]["total"] += 1
        if row["match"] == "true":
            by_domain[domain]["match"] += 1
        if int(row["detections"]) == 0:
            by_domain[domain]["no_detection"] += 1
        if row["top_class"] and row["match"] != "true":
            by_domain[domain][f"wrong_{row['top_class']}"] += 1
    return {domain: dict(counter) for domain, counter in by_domain.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external domain sweep against the two-stage YOLO pipeline.")
    parser.add_argument("--manifest", default="configs/domain_sweep_manifest.csv")
    parser.add_argument("--detector", default="weights/defect_detector.pt")
    parser.add_argument("--severity", default="weights/severity_cls.pt")
    parser.add_argument("--output", default="output/domain_sweep")
    parser.add_argument("--image-cache", default="", help="Optional shared cache directory for downloaded sweep images.")
    parser.add_argument("--thresholds", default="0.45,0.30,0.20,0.10")
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--annotate-conf", type=float, default=0.20)
    args = parser.parse_args()

    output_dir = Path(args.output)
    image_dir = output_dir / "images"
    cache_dir = Path(args.image_cache) if args.image_cache else None
    annotation_dir = output_dir / "annotated"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    rows = read_manifest(Path(args.manifest))
    detector = YOLO(args.detector)
    severity_model = YOLO(args.severity)
    device = device_arg()

    result_rows: list[dict] = []
    detail_rows: list[dict] = []
    downloaded_rows: list[dict] = []

    for row in rows:
        try:
            image_path = ensure_image(row, image_dir, cache_dir)
        except Exception as exc:
            result_rows.append(
                {
                    "id": row["id"],
                    "domain_group": row["domain_group"],
                    "expected_class": row["expected_class"],
                    "threshold": "",
                    "detections": 0,
                    "top_class": "",
                    "top_confidence": "",
                    "top_severity": "",
                    "top_severity_confidence": "",
                    "match": "skipped",
                    "latency_ms": "",
                    "class_counts": "{}",
                    "severity_counts": "{}",
                    "source_url": row.get("source_url", ""),
                    "notes": f"download_failed: {exc}",
                }
            )
            continue
        downloaded_rows.append({**row, "resolved_path": str(image_path)})
        for threshold in thresholds:
            detections, latency_ms = run_one(detector, severity_model, image_path, threshold, args.iou, device)
            top_detection = max(detections, key=lambda item: item["confidence"], default=None)
            top_class = top_detection["class_name"] if top_detection else ""
            match = expected_match(row["expected_class"], top_class if top_detection else None)
            class_counts = Counter(detection["class_name"] for detection in detections)
            severity_counts = Counter(detection["severity"] for detection in detections)
            result_rows.append(
                {
                    "id": row["id"],
                    "domain_group": row["domain_group"],
                    "expected_class": row["expected_class"],
                    "threshold": threshold,
                    "detections": len(detections),
                    "top_class": top_class,
                    "top_confidence": round(top_detection["confidence"], 4) if top_detection else "",
                    "top_severity": top_detection["severity"] if top_detection else "",
                    "top_severity_confidence": round(top_detection["severity_confidence"], 4) if top_detection else "",
                    "match": str(match).lower(),
                    "latency_ms": round(latency_ms, 2),
                    "class_counts": json.dumps(class_counts, sort_keys=True),
                    "severity_counts": json.dumps(severity_counts, sort_keys=True),
                    "source_url": row.get("source_url", ""),
                    "notes": row.get("notes", ""),
                }
            )
            for detection in detections:
                detail_rows.append(
                    {
                        "id": row["id"],
                        "threshold": threshold,
                        "class_name": detection["class_name"],
                        "confidence": round(detection["confidence"], 4),
                        "severity": detection["severity"],
                        "severity_confidence": round(detection["severity_confidence"], 4),
                        "area_ratio": round(detection["area_ratio"], 6),
                        "xyxy": json.dumps(detection["xyxy"]),
                    }
                )
            if abs(threshold - args.annotate_conf) < 1e-9:
                image = cv2.imread(str(image_path))
                if image is not None:
                    cv2.imwrite(str(annotation_dir / f"{safe_name(row['id'])}.jpg"), draw_detections(image, detections))

    result_path = output_dir / "domain_sweep_summary.csv"
    detail_path = output_dir / "domain_sweep_detections.csv"
    manifest_path = output_dir / "resolved_manifest.csv"
    for path, table in [(result_path, result_rows), (detail_path, detail_rows), (manifest_path, downloaded_rows)]:
        if not table:
            continue
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)

    summary = {
        "thresholds": thresholds,
        "primary_threshold": args.annotate_conf,
        "rows": len(rows),
        "result_rows": len(result_rows),
        "primary_threshold_rows": sum(
            1
            for row in result_rows
            if str(row.get("threshold", "")) and abs(float(row["threshold"]) - args.annotate_conf) < 1e-9
        ),
        "skipped_rows": sum(1 for row in result_rows if row.get("match") == "skipped"),
        "domain_summary_at_primary_threshold": summarize_rows(result_rows, args.annotate_conf),
        "summary_csv": str(result_path),
        "detections_csv": str(detail_path),
        "resolved_manifest_csv": str(manifest_path),
        "annotated_dir": str(annotation_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
