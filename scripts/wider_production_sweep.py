from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from domain_sweep import device_arg, draw_detections, ensure_image, expected_match, model_name, safe_name


CLASS_ORDER = ["crack", "spalling", "corrosion", "pothole", "paint_degradation", "none"]

COMMONS_GROUPS = [
    {
        "expected_class": "crack",
        "domain_group": "crack_external",
        "queries": [
            "concrete crack",
            "wall crack building",
            "cracked concrete wall",
            "crack in plaster wall",
            "structural crack concrete",
        ],
        "must_any": ["crack", "cracks", "cracked"],
        "avoid": ["micrograph", "microscope", "diagram", "map", "logo", "gif"],
    },
    {
        "expected_class": "spalling",
        "domain_group": "spalling_external",
        "queries": [
            "concrete spalling exposed rebar",
            "spalled concrete column",
            "spalling concrete",
            "damaged concrete column rebar",
            "concrete delamination spalling",
        ],
        "must_any": ["spalling", "spalled", "rebar", "reinforced concrete"],
        "avoid": ["diagram", "map", "logo", "gif"],
    },
    {
        "expected_class": "corrosion",
        "domain_group": "corrosion_external",
        "queries": [
            "rust corrosion metal",
            "corroded steel beam",
            "rusted metal surface",
            "corrosion bridge steel",
            "rust on iron",
        ],
        "must_any": ["rust", "rusted", "corrosion", "corroded"],
        "avoid": ["diagram", "map", "logo", "gif", "stainless"],
    },
    {
        "expected_class": "pothole",
        "domain_group": "pothole_external",
        "queries": [
            "pothole asphalt road",
            "road pothole damage",
            "asphalt pavement pothole",
            "potholes road",
            "pothole street",
        ],
        "must_any": ["pothole", "potholes"],
        "avoid": ["diagram", "map", "logo", "gif"],
    },
    {
        "expected_class": "paint_degradation",
        "domain_group": "paint_external",
        "queries": [
            "peeling paint wall",
            "flaking paint wall",
            "paint peeling building",
            "damaged paint wall",
            "peeling paint facade",
        ],
        "must_any": ["paint", "peeling", "flaking"],
        "avoid": ["diagram", "map", "logo", "gif"],
    },
    {
        "expected_class": "none",
        "domain_group": "negative_external",
        "queries": [
            "clean concrete wall",
            "brick wall texture",
            "asphalt road surface",
            "building facade clean",
            "concrete surface texture",
        ],
        "must_any": ["wall", "asphalt", "concrete", "facade", "brick"],
        "avoid": [
            "crack",
            "cracked",
            "spalling",
            "spalled",
            "rust",
            "corrosion",
            "corroded",
            "pothole",
            "peeling",
            "flaking",
            "damaged",
            "diagram",
            "map",
            "logo",
            "gif",
        ],
    },
]


def parse_models(value: str) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Model entry must be name=path: {item}")
        name, path = item.split("=", 1)
        models[safe_name(name)] = Path(path)
    return models


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def commons_file_url(title: str) -> str:
    title = title.removeprefix("File:")
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{quote(title, safe='')}"


def commons_source_url(title: str) -> str:
    title = title.removeprefix("File:")
    return f"https://commons.wikimedia.org/wiki/File:{quote(title.replace(' ', '_'), safe=':/_().,%')}"


def title_matches(title: str, must_any: list[str], avoid: list[str]) -> bool:
    lowered = title.lower()
    if any(term in lowered for term in avoid):
        return False
    return any(term in lowered for term in must_any)


def search_commons(group: dict, limit_per_group: int, max_per_query: int, session: requests.Session) -> list[dict]:
    rows: list[dict] = []
    seen_titles: set[str] = set()
    for query in group["queries"]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrsearch": query,
            "gsrlimit": max_per_query,
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
        }
        response = session.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=45)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            if not title or title in seen_titles:
                continue
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            width = int(info.get("width", 0) or 0)
            height = int(info.get("height", 0) or 0)
            if not mime.startswith("image/") or mime in {"image/svg+xml", "image/gif", "image/tiff"}:
                continue
            if min(width, height) < 220:
                continue
            if not title_matches(title, group["must_any"], group["avoid"]):
                continue
            seen_titles.add(title)
            row_id = f"wm_{safe_name(group['domain_group'])}_{safe_name(title.removeprefix('File:'))[:60]}"
            rows.append(
                {
                    "id": row_id,
                    "domain_group": group["domain_group"],
                    "expected_class": group["expected_class"],
                    "local_path": "",
                    "image_url": commons_file_url(title),
                    "source_url": commons_source_url(title),
                    "notes": f"commons_search:{query}",
                }
            )
            if len(rows) >= limit_per_group:
                return rows
        time.sleep(0.4)
    return rows


def add_user_local_rows(rows: list[dict]) -> None:
    candidates = [
        (
            Path(r"C:\Users\User\Downloads\download (1).jpg"),
            "user_wall_damage_download_1",
            "user_real_input",
            "paint_degradation|spalling",
            "Previously uploaded facade/wall-damage test image",
        )
    ]
    existing_ids = {row["id"] for row in rows}
    for path, row_id, domain, expected, note in candidates:
        if path.exists() and row_id not in existing_ids:
            rows.append(
                {
                    "id": row_id,
                    "domain_group": domain,
                    "expected_class": expected,
                    "local_path": str(path),
                    "image_url": "",
                    "source_url": str(path),
                    "notes": note,
                }
            )


def build_manifest(base_manifest: Path, output_path: Path, per_group: int, max_per_query: int) -> list[dict]:
    rows = read_csv(base_manifest)
    add_user_local_rows(rows)
    if per_group <= 0:
        write_csv(output_path, rows)
        return rows
    seen_keys = {(row.get("image_url", ""), row.get("local_path", "")) for row in rows}
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "AIEngGroupProj-wider-production-sweep/1.0 (student evaluation; local cached)",
            "Accept": "application/json",
        }
    )
    for group in COMMONS_GROUPS:
        added = 0
        for row in search_commons(group, per_group, max_per_query, session):
            key = (row.get("image_url", ""), row.get("local_path", ""))
            if key in seen_keys:
                continue
            rows.append(row)
            seen_keys.add(key)
            added += 1
            if added >= per_group:
                break
    write_csv(output_path, rows)
    return rows


def classify_crop(severity_model: YOLO, crop: np.ndarray, device: str | None) -> tuple[str, float]:
    if crop.size == 0:
        return "unknown", 0.0
    results = severity_model.predict(source=crop, imgsz=224, device=device, verbose=False)
    if not results or getattr(results[0], "probs", None) is None:
        return "unknown", 0.0
    result = results[0]
    probs = result.probs
    return model_name(result, int(probs.top1)), float(probs.top1conf)


def run_detector_once(
    detector: YOLO,
    severity_model: YOLO,
    image_path: Path,
    conf_floor: float,
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
        conf=conf_floor,
        iou=iou,
        imgsz=640,
        max_det=100,
        agnostic_nms=True,
        device=device,
        verbose=False,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    detections: list[dict] = []
    if not results or getattr(results[0], "boxes", None) is None or len(results[0].boxes) == 0:
        return detections, latency_ms
    result = results[0]
    boxes = result.boxes
    for coords, class_id, confidence in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int), boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = coords.astype(int).tolist()
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        severity, severity_conf = classify_crop(severity_model, image[y1:y2, x1:x2], device)
        detections.append(
            {
                "class_name": model_name(result, int(class_id)),
                "confidence": float(confidence),
                "severity": severity,
                "severity_confidence": severity_conf,
                "xyxy": [x1, y1, x2, y2],
                "area_ratio": ((x2 - x1) * (y2 - y1)) / float(width * height),
            }
        )
    return detections, latency_ms


def status_for(expected: str, top_class: str, match: bool, detections: int) -> str:
    if expected == "none":
        return "false_positive" if detections else "true_negative"
    if match:
        return "match"
    if detections == 0:
        return "no_detection"
    return f"wrong_{top_class}"


def apply_scenario_variant(image: np.ndarray, variant: str) -> np.ndarray:
    if variant == "original":
        return image
    if variant == "low_light":
        dark = cv2.convertScaleAbs(image, alpha=0.72, beta=-24)
        table = np.array([(i / 255.0) ** (1.0 / 0.78) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(dark, table)
    if variant == "overexposed":
        bright = cv2.convertScaleAbs(image, alpha=1.22, beta=24)
        table = np.array([(i / 255.0) ** (1.0 / 1.18) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(bright, table)
    if variant == "blur_distance":
        height, width = image.shape[:2]
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        small = cv2.resize(blurred, (max(8, int(width * 0.45)), max(8, int(height * 0.45))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if variant == "shadow":
        output = cv2.convertScaleAbs(image, alpha=0.86, beta=-10)
        height, width = output.shape[:2]
        overlay = output.copy()
        pts = np.array([[0, 0], [width, height // 5], [width, height], [width // 4, height]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (18, 18, 18))
        return cv2.addWeighted(overlay, 0.22, output, 0.78, 0)
    if variant == "occlusion":
        output = image.copy()
        height, width = output.shape[:2]
        blocks = [
            (width // 12, height // 12, width // 5, height // 5),
            (width * 7 // 10, height // 6, width * 9 // 10, height // 3),
        ]
        for x1, y1, x2, y2 in blocks:
            cv2.rectangle(output, (x1, y1), (x2, y2), (70, 70, 70), -1)
        return output
    raise ValueError(f"Unknown scenario variant: {variant}")


def expand_scenario_rows(
    rows: list[dict],
    variants: list[str],
    image_dir: Path,
    cache_dir: Path,
    scenario_dir: Path,
) -> list[dict]:
    if variants == ["original"]:
        return [{**row, "scenario_variant": "original"} for row in rows]
    scenario_dir.mkdir(parents=True, exist_ok=True)
    expanded: list[dict] = []
    for row in rows:
        try:
            image_path = ensure_image(row, image_dir, cache_dir)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                expanded.append({**row, "scenario_variant": "original"})
                continue
        except Exception:
            expanded.append({**row, "scenario_variant": "original"})
            continue
        for variant in variants:
            if variant == "original":
                expanded.append({**row, "scenario_variant": "original"})
                continue
            variant_image = apply_scenario_variant(image, variant)
            output_name = f"{safe_name(row['id'])}__{safe_name(variant)}.jpg"
            output_path = scenario_dir / output_name
            cv2.imwrite(str(output_path), variant_image)
            expanded.append(
                {
                    **row,
                    "id": f"{row['id']}__{variant}",
                    "domain_group": f"{row['domain_group']}__{variant}",
                    "local_path": str(output_path),
                    "image_url": "",
                    "scenario_variant": variant,
                    "notes": f"{row.get('notes', '')}; scenario:{variant}",
                }
            )
    return expanded


def evaluate_model(
    model_name_key: str,
    detector_path: Path,
    severity_path: Path,
    rows: list[dict],
    output_dir: Path,
    thresholds: list[float],
    annotate_conf: float,
    iou: float,
    image_dir: Path,
    cache_dir: Path,
) -> dict:
    detector = YOLO(str(detector_path))
    severity_model = YOLO(str(severity_path))
    device = device_arg()
    model_dir = output_dir / model_name_key
    annotation_dir = model_dir / "annotated"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []
    detail_rows: list[dict] = []
    resolved_rows: list[dict] = []
    conf_floor = min(thresholds)

    for index, row in enumerate(rows, start=1):
        print(f"[{model_name_key}] {index}/{len(rows)} {row['id']}", flush=True)
        try:
            image_path = ensure_image(row, image_dir, cache_dir)
            detections, latency_ms = run_detector_once(detector, severity_model, image_path, conf_floor, iou, device)
            resolved_rows.append({**row, "resolved_path": str(image_path)})
        except Exception as exc:
            for threshold in thresholds:
                result_rows.append(
                    {
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
                        "notes": f"{row.get('notes', '')}; error:{exc}",
                    }
                )
            continue

        for detection in detections:
            detail_rows.append(
                {
                    "id": row["id"],
                    "class_name": detection["class_name"],
                    "confidence": round(detection["confidence"], 4),
                    "severity": detection["severity"],
                    "severity_confidence": round(detection["severity_confidence"], 4),
                    "area_ratio": round(detection["area_ratio"], 6),
                    "xyxy": json.dumps(detection["xyxy"]),
                }
            )

        for threshold in thresholds:
            filtered = [detection for detection in detections if detection["confidence"] >= threshold]
            top_detection = max(filtered, key=lambda item: item["confidence"], default=None)
            top_class = top_detection["class_name"] if top_detection else ""
            is_match = expected_match(row["expected_class"], top_class if top_detection else None)
            class_counts = Counter(detection["class_name"] for detection in filtered)
            severity_counts = Counter(detection["severity"] for detection in filtered)
            result_rows.append(
                {
                    "id": row["id"],
                    "domain_group": row["domain_group"],
                    "expected_class": row["expected_class"],
                    "threshold": threshold,
                    "detections": len(filtered),
                    "top_class": top_class,
                    "top_confidence": round(top_detection["confidence"], 4) if top_detection else "",
                    "top_severity": top_detection["severity"] if top_detection else "",
                    "top_severity_confidence": round(top_detection["severity_confidence"], 4) if top_detection else "",
                    "match": str(is_match).lower(),
                    "status": status_for(row["expected_class"], top_class, is_match, len(filtered)),
                    "scenario_variant": row.get("scenario_variant", "original"),
                    "latency_ms": round(latency_ms, 2),
                    "class_counts": json.dumps(class_counts, sort_keys=True),
                    "severity_counts": json.dumps(severity_counts, sort_keys=True),
                    "source_url": row.get("source_url", ""),
                    "notes": row.get("notes", ""),
                }
            )

        filtered_for_annotation = [detection for detection in detections if detection["confidence"] >= annotate_conf]
        image = cv2.imread(str(image_path))
        if image is not None:
            cv2.imwrite(str(annotation_dir / f"{safe_name(row['id'])}.jpg"), draw_detections(image, filtered_for_annotation))

    write_csv(model_dir / "domain_sweep_summary.csv", result_rows)
    write_csv(model_dir / "domain_sweep_detections.csv", detail_rows)
    write_csv(model_dir / "resolved_manifest.csv", resolved_rows)
    summary = summarize_result_rows(result_rows, annotate_conf)
    summary.update(
        {
            "detector": str(detector_path),
            "severity": str(severity_path),
            "rows": len(rows),
            "thresholds": thresholds,
            "primary_threshold": annotate_conf,
            "summary_csv": str(model_dir / "domain_sweep_summary.csv"),
            "detections_csv": str(model_dir / "domain_sweep_detections.csv"),
            "annotated_dir": str(annotation_dir),
        }
    )
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_contact_sheets(result_rows, resolved_rows, annotation_dir, model_dir / "contact_sheets", annotate_conf, model_name_key)
    return summary


def summarize_result_rows(rows: list[dict], threshold: float) -> dict:
    primary_rows = [row for row in rows if abs(float(row["threshold"]) - threshold) < 1e-9 and row["match"] != "skipped"]
    by_expected: dict[str, Counter] = defaultdict(Counter)
    by_domain: dict[str, Counter] = defaultdict(Counter)
    confusion: dict[str, Counter] = defaultdict(Counter)
    latency_values = []
    for row in primary_rows:
        expected = row["expected_class"]
        status = row["status"]
        top_class = row["top_class"] or "none"
        by_expected[expected]["total"] += 1
        by_expected[expected][status] += 1
        by_domain[row["domain_group"]]["total"] += 1
        by_domain[row["domain_group"]][status] += 1
        confusion[expected][top_class] += 1
        if row["latency_ms"] != "":
            latency_values.append(float(row["latency_ms"]))
    matches = sum(1 for row in primary_rows if row["status"] in {"match", "true_negative"})
    return {
        "primary_rows": len(primary_rows),
        "matches_or_true_negatives": matches,
        "failure_rows": len(primary_rows) - matches,
        "pass_rate": round(matches / len(primary_rows), 4) if primary_rows else 0,
        "by_expected": {key: dict(value) for key, value in by_expected.items()},
        "by_domain": {key: dict(value) for key, value in by_domain.items()},
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "latency_ms_mean": round(sum(latency_values) / len(latency_values), 2) if latency_values else None,
    }


def load_image_for_tile(path: Path, tile_width: int, image_height: int) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        image = np.zeros((image_height, tile_width, 3), dtype=np.uint8)
    height, width = image.shape[:2]
    scale = min(tile_width / max(width, 1), image_height / max(height, 1))
    resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((image_height, tile_width, 3), 245, dtype=np.uint8)
    y = (image_height - resized.shape[0]) // 2
    x = (tile_width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def put_wrapped_text(image: np.ndarray, text: str, x: int, y: int, max_width: int, color: tuple[int, int, int]) -> int:
    words = text.split()
    line = ""
    line_height = 18
    for word in words:
        candidate = f"{line} {word}".strip()
        width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        if width > max_width and line:
            cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y += line_height
            line = word
        else:
            line = candidate
    if line:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += line_height
    return y


def make_contact_sheets(
    result_rows: list[dict],
    resolved_rows: list[dict],
    annotation_dir: Path,
    output_dir: Path,
    threshold: float,
    model_name_key: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_rows = [row for row in result_rows if abs(float(row["threshold"]) - threshold) < 1e-9 and row["match"] != "skipped"]
    resolved = {row["id"]: row for row in resolved_rows}

    def sort_key(row: dict) -> tuple[int, str, str]:
        status = row["status"]
        rank = 0 if status not in {"match", "true_negative"} else 1
        return (rank, row["expected_class"], row["id"])

    primary_rows.sort(key=sort_key)
    columns = 3
    rows_per_page = 4
    tile_width = 460
    image_height = 300
    caption_height = 112
    margin = 16
    page_capacity = columns * rows_per_page
    page_count = max(1, math.ceil(len(primary_rows) / page_capacity))
    for page_index in range(page_count):
        page_rows = primary_rows[page_index * page_capacity : (page_index + 1) * page_capacity]
        sheet = np.full(
            (
                rows_per_page * (image_height + caption_height + margin) + margin,
                columns * (tile_width + margin) + margin,
                3,
            ),
            250,
            dtype=np.uint8,
        )
        for tile_index, row in enumerate(page_rows):
            grid_y = tile_index // columns
            grid_x = tile_index % columns
            x = margin + grid_x * (tile_width + margin)
            y = margin + grid_y * (image_height + caption_height + margin)
            image_path = annotation_dir / f"{safe_name(row['id'])}.jpg"
            tile = load_image_for_tile(image_path, tile_width, image_height)
            sheet[y : y + image_height, x : x + tile_width] = tile
            status = row["status"]
            good = status in {"match", "true_negative"}
            color = (37, 137, 61) if good else (35, 45, 210)
            cv2.rectangle(sheet, (x, y + image_height), (x + tile_width, y + image_height + caption_height), (255, 255, 255), -1)
            cv2.putText(sheet, f"{model_name_key} | {status}", (x + 10, y + image_height + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            summary = (
                f"{row['id']} | exp: {row['expected_class']} | pred: {row['top_class'] or 'none'} "
                f"{row['top_confidence']} | sev: {row['top_severity'] or 'n/a'}"
            )
            next_y = put_wrapped_text(sheet, summary, x + 10, y + image_height + 48, tile_width - 20, (30, 35, 45))
            source_note = resolved.get(row["id"], {}).get("notes", "")
            put_wrapped_text(sheet, source_note, x + 10, next_y + 2, tile_width - 20, (85, 90, 100))
        cv2.imwrite(str(output_dir / f"{model_name_key}_contact_sheet_{page_index + 1:02d}.jpg"), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a wider production-style sweep and create visual contact sheets.")
    parser.add_argument("--base-manifest", default="configs/production_eval_manifest.csv")
    parser.add_argument("--output", default="output/wider_production_sweep")
    parser.add_argument("--models", default="current=weights/drive_production_check/defect_detector.pt,balanced=weights/drive_production_check/defect_detector_balanced_candidate.pt")
    parser.add_argument("--severity", default="weights/drive_production_check/severity_cls.pt")
    parser.add_argument("--per-group", type=int, default=7)
    parser.add_argument("--max-per-query", type=int, default=12)
    parser.add_argument("--thresholds", default="0.45,0.30,0.20,0.10")
    parser.add_argument("--annotate-conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--scenario-variants", default="original,low_light,overexposed,blur_distance,shadow,occlusion")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "wider_production_manifest.csv"
    rows = build_manifest(Path(args.base_manifest), manifest_path, args.per_group, args.max_per_query)
    print(f"Manifest rows: {len(rows)}")
    print(f"Manifest: {manifest_path}")

    image_dir = output_dir / "images"
    cache_dir = output_dir / "image_cache"
    scenario_dir = output_dir / "scenario_images"
    scenario_variants = [item.strip() for item in args.scenario_variants.split(",") if item.strip()]
    rows = expand_scenario_rows(rows, scenario_variants, image_dir, cache_dir, scenario_dir)
    write_csv(output_dir / "wider_production_manifest_with_scenarios.csv", rows)
    print(f"Scenario rows: {len(rows)} using variants: {scenario_variants}")
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    severity_path = Path(args.severity)
    summaries: dict[str, dict] = {}
    for name, detector_path in parse_models(args.models).items():
        if not detector_path.exists():
            print(f"Skipping {name}: missing detector {detector_path}")
            continue
        summaries[name] = evaluate_model(
            name,
            detector_path,
            severity_path,
            rows,
            output_dir,
            thresholds,
            args.annotate_conf,
            args.iou,
            image_dir,
            cache_dir,
        )

    comparison_path = output_dir / "comparison_summary.json"
    comparison_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "comparison": str(comparison_path), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
