"""
Real-world facility evaluation using:
1. Test-set images (unseen during training) for defect classes
2. Images with EMPTY labels as clean/negative (FP test)
3. Images with multiple class bounding boxes (multi-defect test)
4. Augmented wide-angle perspective variants
Reports per-class accuracy, FP rate, severity distribution.
"""
from __future__ import annotations
import json, random, time
from collections import defaultdict, Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

DETECTOR = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt")
SEVERITY = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls.pt")
OUT_DIR  = Path(r"C:\Users\User\AIEngGroupProj\output\realworld_eval2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET  = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue")
TEST_IMG = DATASET / "images" / "test"
TEST_LBL = DATASET / "labels" / "test"
VAL_IMG  = DATASET / "images" / "val"
VAL_LBL  = DATASET / "labels" / "val"

DEVICE = "0" if torch.cuda.is_available() else "cpu"
CONF   = 0.45
IOU    = 0.45
CLASS_NAMES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
RNG = random.Random(99)

def get_label_classes(lbl_path: Path) -> list[int]:
    if not lbl_path.exists():
        return []
    classes = []
    for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if parts:
            classes.append(int(parts[0]))
    return classes

def build_test_sets(n_per_class=15, n_neg=20, n_multi=10):
    """Build test image lists from test + val splits."""
    # Per-class images (single dominant class)
    per_class: dict[int, list[Path]] = {i: [] for i in range(5)}
    multi_defect: list[Path] = []
    negatives: list[Path] = []

    for img_dir, lbl_dir in [(TEST_IMG, TEST_LBL), (VAL_IMG, VAL_LBL)]:
        if not img_dir.exists():
            continue
        for img in img_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            classes = get_label_classes(lbl)
            unique = list(set(classes))
            if len(classes) == 0:
                negatives.append(img)
            elif len(unique) == 1:
                per_class[unique[0]].append(img)
            elif len(unique) >= 2:
                multi_defect.append(img)

    selected: dict[str, list[Path]] = {}
    for cls_id, images in per_class.items():
        RNG.shuffle(images)
        selected[CLASS_NAMES[cls_id]] = images[:n_per_class]

    RNG.shuffle(negatives)
    selected["none"] = negatives[:n_neg]
    RNG.shuffle(multi_defect)
    selected["multi"] = multi_defect[:n_multi]
    return selected

def wide_angle_perspective(img: np.ndarray, scale: float = 0.55) -> np.ndarray:
    """Simulate a wider perspective by shrinking the defect image into a larger blank canvas."""
    h, w = img.shape[:2]
    canvas_h, canvas_w = int(h / scale), int(w / scale)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 180  # grey building wall
    y_off = (canvas_h - h) // 2
    x_off = (canvas_w - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = img
    return canvas

def classify_severity(sev_model: YOLO, crop: np.ndarray) -> tuple[str, float]:
    if crop is None or crop.size == 0:
        return "unknown", 0.0
    res = sev_model.predict(source=crop, imgsz=224, device=DEVICE, verbose=False)
    if not res or res[0].probs is None:
        return "unknown", 0.0
    probs = res[0].probs
    top1_conf = float(probs.top1conf)
    top_vals = sorted(probs.data.tolist(), reverse=True)
    margin = top_vals[0] - top_vals[1] if len(top_vals) > 1 else top_vals[0]
    if top1_conf < 0.40 or margin < 0.08:
        return "uncertain", top1_conf
    return res[0].names.get(int(probs.top1), "unknown"), top1_conf

def infer(det_model: YOLO, sev_model: YOLO, img: np.ndarray):
    h, w = img.shape[:2]
    t0 = time.perf_counter()
    preds = det_model.predict(source=img, conf=CONF, iou=IOU, imgsz=640,
                               max_det=120, agnostic_nms=False, device=DEVICE, verbose=False)
    latency = (time.perf_counter() - t0) * 1000
    detections = []
    if preds and preds[0].boxes is not None and len(preds[0].boxes) > 0:
        for coords, cls_id, conf in zip(
            preds[0].boxes.xyxy.cpu().numpy(),
            preds[0].boxes.cls.cpu().numpy().astype(int),
            preds[0].boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = [int(v) for v in coords]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
            sev_label, sev_conf = classify_severity(sev_model, crop) if crop is not None else ("unknown", 0.0)
            name = preds[0].names.get(cls_id, str(cls_id))
            detections.append({
                "class": name, "conf": round(float(conf), 3),
                "severity": sev_label, "sev_conf": round(sev_conf, 3),
                "area_ratio": round(((x2-x1)*(y2-y1))/(w*h), 4),
            })
    return detections, round(latency, 1)

def main():
    print("Loading models...")
    det = YOLO(str(DETECTOR))
    sev = YOLO(str(SEVERITY))
    print(f"Models loaded. Device={DEVICE}, conf={CONF}\n")

    test_sets = build_test_sets()

    stats = {
        "per_class": {},
        "false_positive": {"fp": 0, "tn": 0, "total": 0},
        "multi_defect": {"any_detected": 0, "multi_detected": 0, "total": 0},
        "wide_angle": {},
        "severity_dist": Counter(),
        "uncertain_rate": 0,
    }
    all_results = []

    # ── 1. Per-class detection test ──────────────────────────────────────────
    print("="*60)
    print("1. PER-CLASS DETECTION (test+val images, conf=0.45)")
    print("="*60)
    for cls_name in CLASS_NAMES:
        images = test_sets.get(cls_name, [])
        if not images:
            print(f"  {cls_name}: no test images found")
            continue
        correct = miss = wrong = fp_count = 0
        sev_dist: Counter = Counter()
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            dets, _ = infer(det, sev, img)
            detected = {d["class"] for d in dets}
            for d in dets:
                sev_dist[d["severity"]] += 1
                stats["severity_dist"][d["severity"]] += 1
            if cls_name in detected:
                correct += 1
            elif not dets:
                miss += 1
            else:
                wrong += 1
        total = correct + miss + wrong
        acc = correct / total if total else 0
        print(f"  {cls_name:20s}: {correct}/{total} ({acc:.0%})  "
              f"miss={miss} wrong_class={wrong}  severity={dict(sev_dist.most_common(3))}")
        stats["per_class"][cls_name] = {"correct": correct, "miss": miss, "wrong": wrong,
                                         "total": total, "accuracy": round(acc, 3)}

    # ── 2. False positive test (clean/negative images) ───────────────────────
    print(f"\n{'='*60}")
    print(f"2. FALSE POSITIVE TEST (clean images, expected=none)")
    print("="*60)
    fp_imgs = test_sets.get("none", [])
    print(f"  Testing {len(fp_imgs)} clean images...")
    fp_by_class: Counter = Counter()
    for img_path in fp_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dets, _ = infer(det, sev, img)
        stats["false_positive"]["total"] += 1
        if dets:
            stats["false_positive"]["fp"] += 1
            for d in dets:
                fp_by_class[d["class"]] += 1
        else:
            stats["false_positive"]["tn"] += 1
    fp_total = stats["false_positive"]["total"]
    fp_count = stats["false_positive"]["fp"]
    fp_rate = fp_count / fp_total if fp_total else 0
    print(f"  FP rate: {fp_count}/{fp_total} ({fp_rate:.0%})")
    if fp_by_class:
        print(f"  Most common FP classes: {dict(fp_by_class.most_common(5))}")
    else:
        print(f"  ✓ No false positives")

    # ── 3. Multi-defect detection test ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"3. MULTI-DEFECT DETECTION (images with 2+ defect classes)")
    print("="*60)
    multi_imgs = test_sets.get("multi", [])
    print(f"  Testing {len(multi_imgs)} multi-defect images...")
    for img_path in multi_imgs[:10]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dets, latency = infer(det, sev, img)
        detected_cls = [d["class"] for d in dets]
        unique_cls = set(detected_cls)
        gt_classes = [CLASS_NAMES[c] for c in get_label_classes(VAL_LBL / (img_path.stem + ".txt")) if c < 5]
        if not gt_classes:
            gt_classes = [CLASS_NAMES[c] for c in get_label_classes(TEST_LBL / (img_path.stem + ".txt")) if c < 5]
        stats["multi_defect"]["total"] += 1
        if dets:
            stats["multi_defect"]["any_detected"] += 1
        if len(unique_cls) >= 2:
            stats["multi_defect"]["multi_detected"] += 1
        print(f"  {img_path.stem[:40]:40s}: GT={set(gt_classes)} → detected={unique_cls} ({len(dets)} boxes, {latency:.0f}ms)")

    # ── 4. Wide-angle perspective test ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"4. WIDE-ANGLE PERSPECTIVE TEST (defect in wider canvas)")
    print("="*60)
    wide_results = {}
    for cls_name in CLASS_NAMES:
        images = test_sets.get(cls_name, [])[:5]
        correct = total = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            wide = wide_angle_perspective(img, scale=0.50)  # defect fills 50% of frame
            dets, _ = infer(det, sev, wide)
            total += 1
            if any(d["class"] == cls_name for d in dets):
                correct += 1
        acc = correct / total if total else 0
        print(f"  {cls_name:20s}: {correct}/{total} ({acc:.0%}) at 50% scale (wide angle)")
        wide_results[cls_name] = {"correct": correct, "total": total, "accuracy": round(acc, 3)}
    stats["wide_angle"] = wide_results

    # ── 5. Severity quality ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"5. SEVERITY DISTRIBUTION ON REAL DETECTIONS")
    print("="*60)
    sev_total = sum(stats["severity_dist"].values())
    uncertain = stats["severity_dist"].get("uncertain", 0)
    stats["uncertain_rate"] = round(uncertain / sev_total, 3) if sev_total else 0
    for sev, cnt in sorted(stats["severity_dist"].items()):
        pct = cnt / sev_total if sev_total else 0
        print(f"  {sev:12s}: {cnt:4d} ({pct:.0%})")
    print(f"  → Uncertain rate: {stats['uncertain_rate']:.0%}  (uncertain = low confidence, unreliable)")

    # ── Final verdict ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"VERDICT")
    print("="*60)
    avg_acc = sum(v["accuracy"] for v in stats["per_class"].values()) / max(len(stats["per_class"]), 1)
    avg_wide = sum(v["accuracy"] for v in wide_results.values()) / max(len(wide_results), 1)
    print(f"  Avg per-class detection accuracy:  {avg_acc:.0%}")
    print(f"  Avg wide-angle detection accuracy: {avg_wide:.0%}")
    print(f"  False positive rate:               {fp_rate:.0%}")
    print(f"  Multi-defect any-detection rate:   {stats['multi_defect']['any_detected']}/{stats['multi_defect']['total']}")
    print(f"  Severity uncertain rate:           {stats['uncertain_rate']:.0%}")

    (OUT_DIR / "eval2_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nFull stats: {OUT_DIR / 'eval2_stats.json'}")

if __name__ == "__main__":
    main()
