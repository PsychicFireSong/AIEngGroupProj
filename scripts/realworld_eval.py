"""
Real-world facility evaluation: tests hn_weak detector + severity classifier on
actual building/facility images including clean facades (FP test), multi-defect,
and wide-angle perspectives.
"""
from __future__ import annotations
import json, time, sys, io, urllib.request, urllib.error
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DETECTOR  = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt")
SEVERITY  = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls.pt")
OUT_DIR   = Path(r"C:\Users\User\AIEngGroupProj\output\realworld_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "0" if torch.cuda.is_available() else "cpu"
CONF   = 0.45   # production default
IOU    = 0.45

# ── Test images ─────────────────────────────────────────────────────────────
# Format: (label, expected_class, url)
TEST_IMAGES = [
    # ── Clean facades / hard negatives ──────────────────────────────────────
    ("clean_facade_brick",   "none", "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Brick_wall_close-up_view.jpg/800px-Brick_wall_close-up_view.jpg"),
    ("clean_concrete_wall",  "none", "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Concrete_wall_texture.jpg/800px-Concrete_wall_texture.jpg"),
    ("clean_building_wide",  "none", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/New_building_%283%29.jpg/800px-New_building_%283%29.jpg"),
    # ── Crack — various angles ───────────────────────────────────────────────
    ("crack_closeup",        "crack", "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Cracked_wall.jpg/800px-Cracked_wall.jpg"),
    ("crack_wide_facade",    "crack", "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Crack_in_the_wall.jpg/800px-Crack_in_the_wall.jpg"),
    ("crack_concrete",       "crack", "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Concrete_cracked.jpg/800px-Concrete_cracked.jpg"),
    # ── Spalling ─────────────────────────────────────────────────────────────
    ("spalling_concrete",    "spalling", "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Concrete_spalling.jpg/800px-Concrete_spalling.jpg"),
    # ── Corrosion / rust ─────────────────────────────────────────────────────
    ("corrosion_metal",      "corrosion", "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Rust_and_dirt.jpg/800px-Rust_and_dirt.jpg"),
    ("corrosion_beam",       "corrosion", "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Rust_never_sleeps.jpg/800px-Rust_never_sleeps.jpg"),
    # ── Paint degradation ────────────────────────────────────────────────────
    ("paint_peeling",        "paint_degradation", "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Peeling_paint_1.jpg/800px-Peeling_paint_1.jpg"),
    ("paint_flaking_wall",   "paint_degradation", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Flaking_paint.jpg/800px-Flaking_paint.jpg"),
    # ── Pothole ──────────────────────────────────────────────────────────────
    ("pothole_road",         "pothole", "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Pothole_in_road.jpg/800px-Pothole_in_road.jpg"),
    # ── Multi-defect / mixed context ─────────────────────────────────────────
    ("multi_damaged_building","crack|paint_degradation", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Abandoned_building_with_peeling_paint.jpg/800px-Abandoned_building_with_peeling_paint.jpg"),
    ("multi_deteriorated",   "spalling|corrosion", "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Deteriorated_concrete.jpg/800px-Deteriorated_concrete.jpg"),
]

HEADERS = {"User-Agent": "AIEngGroupProj-realworld-eval/1.0 (student project)"}

def fetch_image(url: str, timeout: int = 20) -> np.ndarray | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        print(f"  [fetch] FAILED: {exc}")
        return None

def classify_severity(sev_model: YOLO, crop: np.ndarray) -> tuple[str, float]:
    if crop is None or crop.size == 0:
        return "unknown", 0.0
    res = sev_model.predict(source=crop, imgsz=224, device=DEVICE, verbose=False)
    if not res or res[0].probs is None:
        return "unknown", 0.0
    probs = res[0].probs
    top1_conf = float(probs.top1conf)
    # margin gate matching app logic
    top2 = sorted(probs.data.tolist(), reverse=True)
    margin = top2[0] - top2[1] if len(top2) > 1 else top2[0]
    if top1_conf < 0.40 or margin < 0.08:
        return "uncertain", top1_conf
    return res[0].names.get(int(probs.top1), "unknown"), top1_conf

def run_eval():
    print(f"Loading models on device={DEVICE}...")
    det = YOLO(str(DETECTOR))
    sev = YOLO(str(SEVERITY))
    print("Models loaded.\n")

    results = []
    by_expected: dict[str, dict] = defaultdict(lambda: {"correct": 0, "fp": 0, "missed": 0, "total": 0})

    for label, expected, url in TEST_IMAGES:
        print(f"── {label} (expected: {expected})")
        img = fetch_image(url)
        if img is None:
            print(f"  SKIP (download failed)\n")
            results.append({"label": label, "expected": expected, "status": "download_failed"})
            continue

        h, w = img.shape[:2]
        t0 = time.perf_counter()
        preds = det.predict(source=img, conf=CONF, iou=IOU, imgsz=640,
                            max_det=120, agnostic_nms=False, device=DEVICE, verbose=False)
        latency = (time.perf_counter() - t0) * 1000

        boxes = preds[0].boxes if preds else None
        detections = []
        if boxes is not None and len(boxes) > 0:
            for coords, cls_id, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy().astype(int),
                boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = [int(v) for v in coords]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
                sev_label, sev_conf = classify_severity(sev, crop) if crop is not None else ("unknown", 0.0)
                name = preds[0].names.get(cls_id, str(cls_id))
                area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                detections.append({
                    "class": name, "conf": round(float(conf), 3),
                    "severity": sev_label, "sev_conf": round(sev_conf, 3),
                    "area_ratio": round(area_ratio, 4),
                })
                print(f"  [{name}] conf={conf:.2f}  severity={sev_label}({sev_conf:.2f})  area={area_ratio:.3f}")

        if not detections:
            print(f"  (no detections above conf={CONF})")

        expected_set = {e.strip() for e in expected.split("|")}
        detected_classes = {d["class"] for d in detections}

        if expected == "none":
            status = "FP" if detections else "TN"
            by_expected["none"]["total"] += 1
            if detections:
                by_expected["none"]["fp"] += 1
                print(f"  ⚠ FALSE POSITIVE — detected {detected_classes} on clean image!")
            else:
                print(f"  ✓ True Negative (clean image, no detection)")
        else:
            hits = expected_set & detected_classes
            if hits:
                status = "match"
                by_expected[expected]["correct"] += 1
                print(f"  ✓ Match: {hits}")
            else:
                status = "miss"
                by_expected[expected]["missed"] += 1
                print(f"  ✗ Miss (detected: {detected_classes or 'nothing'})")
            by_expected[expected]["total"] += 1

        print(f"  latency={latency:.0f}ms\n")

        # Save annotated image
        ann = img.copy()
        for det in detections:
            pass  # bboxes already drawn by YOLO if needed
        results.append({
            "label": label, "expected": expected, "status": status,
            "detections": detections, "latency_ms": round(latency, 1),
        })

    # Summary
    print("\n" + "="*60)
    print("REAL-WORLD EVALUATION SUMMARY")
    print("="*60)
    fp_images = [r for r in results if r.get("status") == "FP"]
    tn_images = [r for r in results if r.get("status") == "TN"]
    match_images = [r for r in results if r.get("status") == "match"]
    miss_images  = [r for r in results if r.get("status") == "miss"]
    skipped = [r for r in results if r.get("status") == "download_failed"]

    print(f"\nFalse Positive Check (clean images): {len(fp_images)} FP / {len(fp_images)+len(tn_images)} tested")
    if fp_images:
        for r in fp_images:
            print(f"  ⚠ {r['label']}: detected {[d['class'] for d in r['detections']]}")
    else:
        print("  ✓ No false positives on clean images")

    print(f"\nDefect Detection: {len(match_images)} match, {len(miss_images)} miss (excl. {len(skipped)} download failures)")
    for r in miss_images:
        print(f"  ✗ {r['label']} (expected {r['expected']}, got {[d['class'] for d in r['detections']]})")

    print(f"\nSeverity Distribution on Detections:")
    all_dets = [d for r in results for d in r.get("detections", [])]
    sev_counts = defaultdict(int)
    for d in all_dets:
        sev_counts[d["severity"]] += 1
    for sev, cnt in sorted(sev_counts.items()):
        print(f"  {sev}: {cnt}")

    (OUT_DIR / "realworld_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results: {OUT_DIR / 'realworld_results.json'}")

if __name__ == "__main__":
    run_eval()
