"""
diagnose_val.py - Systematic per-class weakness analysis on the val set.

Reports:
  1. Per-class AP50, precision, recall at standard conf
  2. Confidence score histograms per class (detected vs missed)
  3. False-positive breakdown: which backgrounds fire as which class
  4. Miss-vs-calibration split: are misses truly not detected, or detected below conf?

Run after stopping training so GPU is free.
Usage: python scripts/diagnose_val.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

WEIGHTS    = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs"
                  r"\continued_hn_weak_finetune\runs\current\weights"
                  r"\defect_detector_hn_weak_candidate.pt")
VAL_YAML   = Path(r"C:\Users\User\AIEngGroupProj\output\v7_lean_dataset\data.yaml")
OUT_DIR    = Path(r"C:\Users\User\AIEngGroupProj\output\diagnosis")
CLASSES    = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
CONF_SWEEP = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]


def _iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def xywhn_to_xyxy(cx, cy, w, h, W, H):
    return ((cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    # ── 1. Official val() at low conf to capture full score range ────────────
    print("\n[1/4] Running model.val() (conf=0.001, iou=0.5)...")
    val_results = model.val(
        data=str(VAL_YAML),
        conf=0.001,
        iou=0.5,
        workers=0,
        device="0",
        verbose=False,
    )

    print("\n=== PER-CLASS METRICS (conf=0.001) ===")
    print(f"{'Class':<22} {'AP50':>7} {'Precision':>10} {'Recall':>8}")
    print("-" * 52)
    nc = len(CLASSES)
    ap50_per_class    = val_results.box.ap50      # shape [nc]
    prec_per_class    = val_results.box.p         # shape [nc]
    rec_per_class     = val_results.box.r         # shape [nc]
    for i, cls in enumerate(CLASSES):
        ap   = float(ap50_per_class[i]) if i < len(ap50_per_class) else float("nan")
        prec = float(prec_per_class[i]) if i < len(prec_per_class) else float("nan")
        rec  = float(rec_per_class[i])  if i < len(rec_per_class)  else float("nan")
        bar  = "#" * int(ap * 40)
        print(f"  {cls:<20} {ap:>7.3f} {prec:>10.3f} {rec:>8.3f}  {bar}")
    print(f"\n  mAP50 overall : {float(val_results.box.map50):.4f}")
    print(f"  mAP50-95      : {float(val_results.box.map):.4f}")

    # ── 2. Confidence sweep — recall vs threshold per class ──────────────────
    print("\n[2/4] Confidence threshold sweep...")
    sweep_rows: list[dict] = []
    for conf in CONF_SWEEP:
        r = model.val(data=str(VAL_YAML), conf=conf, iou=0.5, workers=0,
                      device="0", verbose=False)
        row = {"conf": conf, "mAP50": float(r.box.map50)}
        for i, cls in enumerate(CLASSES):
            row[f"rec_{cls}"] = float(r.box.r[i]) if i < len(r.box.r) else 0.0
            row[f"prec_{cls}"] = float(r.box.p[i]) if i < len(r.box.p) else 0.0
        sweep_rows.append(row)

    print(f"\n{'Conf':>6}  {'mAP50':>7}  " + "  ".join(f"{c[:5]:>7}" for c in CLASSES))
    print("-" * 75)
    for row in sweep_rows:
        recs = "  ".join(f"{row[f'rec_{c}']:>7.3f}" for c in CLASSES)
        print(f"  {row['conf']:>4.2f}   {row['mAP50']:>7.4f}  {recs}")

    # ── 3. FP analysis — what does the model fire on in background-only images?
    print("\n[3/4] False-positive analysis (images with no GT labels)...")
    import yaml, cv2
    data_cfg = yaml.safe_load(VAL_YAML.read_text())
    val_img_root = Path(data_cfg["path"]) / data_cfg.get("val", "images/val")

    fp_counter: dict[int, int] = collections.defaultdict(int)
    fp_conf_sums: dict[int, list] = collections.defaultdict(list)
    total_bg_images = 0

    val_lbl_root = val_img_root.parent.parent / "labels" / "val"
    for img_path in sorted(val_img_root.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = val_lbl_root / (img_path.stem + ".txt")
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            continue  # skip images that DO have GT labels
        total_bg_images += 1
        preds = model.predict(str(img_path), conf=0.25, verbose=False, device="0")[0]
        for box in preds.boxes:
            cls_id = int(box.cls)
            conf_  = float(box.conf)
            fp_counter[cls_id] += 1
            fp_conf_sums[cls_id].append(conf_)

    print(f"\n  Background-only val images: {total_bg_images}")
    print(f"  {'Class':<22} {'FP count':>10} {'Avg conf':>10} {'Max conf':>10}")
    print("  " + "-" * 56)
    for i, cls in enumerate(CLASSES):
        count = fp_counter.get(i, 0)
        confs = fp_conf_sums.get(i, [])
        avg_c = np.mean(confs) if confs else 0.0
        max_c = np.max(confs) if confs else 0.0
        print(f"  {cls:<22} {count:>10} {avg_c:>10.3f} {max_c:>10.3f}")

    # ── 4. Miss analysis — for GT boxes that were missed, what was max IOU pred?
    print("\n[4/4] Miss analysis — how badly are misses missed?")
    miss_stats: dict[int, list] = collections.defaultdict(list)  # class -> list of best-iou-with-any-pred

    for img_path in sorted(val_img_root.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = val_lbl_root / (img_path.stem + ".txt")
        if not lbl_path.exists() or lbl_path.stat().st_size == 0:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        gt_boxes: list[tuple] = []
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            box = xywhn_to_xyxy(float(parts[1]), float(parts[2]),
                                 float(parts[3]), float(parts[4]), W, H)
            gt_boxes.append((cls_id, box))

        if not gt_boxes:
            continue

        # Low-conf predictions to capture near-misses
        preds = model.predict(str(img_path), conf=0.01, verbose=False, device="0")[0]
        pred_boxes = []
        for box in preds.boxes:
            xyxy = box.xyxy[0].tolist()
            pred_boxes.append((int(box.cls), float(box.conf), xyxy))

        for cls_id, gt_box in gt_boxes:
            if pred_boxes:
                best_iou = max(_iou(gt_box, pb[2]) for pb in pred_boxes)
                best_same_cls = max(
                    (_iou(gt_box, pb[2]) for pb in pred_boxes if pb[0] == cls_id),
                    default=0.0,
                )
            else:
                best_iou = best_same_cls = 0.0
            miss_stats[cls_id].append((best_iou, best_same_cls))

    print(f"\n  {'Class':<22} {'GT boxes':>9} {'Avg best IoU':>13} {'IoU>0.5 %':>10} {'Same-cls IoU':>13}")
    print("  " + "-" * 72)
    for i, cls in enumerate(CLASSES):
        records = miss_stats.get(i, [])
        if not records:
            continue
        all_iou  = [r[0] for r in records]
        same_iou = [r[1] for r in records]
        above_half = sum(1 for v in all_iou if v >= 0.5) / len(all_iou)
        print(f"  {cls:<22} {len(records):>9} {np.mean(all_iou):>13.3f} "
              f"{above_half:>10.1%} {np.mean(same_iou):>13.3f}")
    print("\n  Interpretation: low avg-best-IoU = model truly ignores the region.")
    print("  High avg-best-IoU but wrong class = confusion, not miss.")

    # ── Save summary JSON ────────────────────────────────────────────────────
    summary = {
        "per_class_ap50": {CLASSES[i]: float(ap50_per_class[i])
                           for i in range(min(nc, len(ap50_per_class)))},
        "per_class_recall": {CLASSES[i]: float(rec_per_class[i])
                             for i in range(min(nc, len(rec_per_class)))},
        "conf_sweep": sweep_rows,
        "fp_bg_count": {CLASSES[k]: v for k, v in fp_counter.items() if k < nc},
    }
    (OUT_DIR / "diagnosis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nSummary saved to: {OUT_DIR / 'diagnosis.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
