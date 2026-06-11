"""
eval_threshold_fix.py  —  measure impact of fixing per-class conf thresholds.

Current API behaviour:
  YOLO runs at conf_floor = min(0.25, 0.30) = 0.25 (due to existing overrides)
  Post-processing gate: crack/spalling/pothole use global conf=0.45 → drops 0.25-0.44 detections

Proposed fix:
  Add crack=0.25, spalling=0.25, pothole=0.25 overrides
  Post-processing gate: all classes at 0.25, paint_degradation at 0.30

This script simulates both behaviours on the val set and reports per-class P/R/F1.
"""
from __future__ import annotations
import collections
import json
from pathlib import Path
import numpy as np

WEIGHTS  = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs"
                r"\continued_hn_weak_finetune\runs\current\weights"
                r"\defect_detector_hn_weak_candidate.pt")
VAL_YAML = Path(r"C:\Users\User\AIEngGroupProj\output\v7_lean_dataset\data.yaml")
OUT_DIR  = Path(r"C:\Users\User\AIEngGroupProj\output\threshold_eval")
CLASSES  = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
IOU_THRESH = 0.5

# Threshold configs to compare
CONFIGS = {
    "current_api  (crack/spalling/pothole=0.45)": {
        "crack": 0.45, "spalling": 0.45, "corrosion": 0.25,
        "pothole": 0.45, "paint_degradation": 0.30,
    },
    "proposed_fix (all=0.25, paint_deg=0.30)": {
        "crack": 0.25, "spalling": 0.25, "corrosion": 0.25,
        "pothole": 0.25, "paint_degradation": 0.30,
    },
}


def iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def xywhn_to_xyxy(cx, cy, w, h, W, H):
    return ((cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H)


def evaluate_config(model, val_img_root: Path, val_lbl_root: Path,
                    thresholds: dict[str, float]) -> dict:
    run_conf = min(thresholds.values())  # YOLO runs at lowest threshold
    tp = collections.defaultdict(int)
    fp = collections.defaultdict(int)
    fn = collections.defaultdict(int)
    import cv2

    for img_path in sorted(val_img_root.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = val_lbl_root / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        # GT boxes
        gt: list[tuple[int, tuple]] = []
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                box = xywhn_to_xyxy(float(parts[1]), float(parts[2]),
                                     float(parts[3]), float(parts[4]), W, H)
                gt.append((cid, box))

        # Predictions filtered by per-class threshold
        preds_raw = model.predict(str(img_path), conf=run_conf,
                                  verbose=False, device="0")[0]
        preds: list[tuple[int, float, tuple]] = []
        for box in preds_raw.boxes:
            cid = int(box.cls)
            conf_ = float(box.conf)
            cname = CLASSES[cid] if cid < len(CLASSES) else str(cid)
            thr = thresholds.get(cname, 0.45)
            if conf_ < thr:
                continue
            xyxy = tuple(box.xyxy[0].tolist())
            preds.append((cid, conf_, xyxy))

        # Match preds to GT (greedy by conf desc)
        matched_gt = set()
        matched_pred = set()
        for pi, (pcid, pconf, pbox) in enumerate(
                sorted(preds, key=lambda x: -x[1])):
            best_iou, best_gi = 0.0, -1
            for gi, (gcid, gbox) in enumerate(gt):
                if gi in matched_gt or gcid != pcid:
                    continue
                v = iou(pbox, gbox)
                if v > best_iou:
                    best_iou, best_gi = v, gi
            if best_iou >= IOU_THRESH and best_gi >= 0:
                tp[pcid] += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                fp[pcid] += 1

        for gi, (gcid, _) in enumerate(gt):
            if gi not in matched_gt:
                fn[gcid] += 1

    results = {}
    for i, cls in enumerate(CLASSES):
        t = tp[i]; f = fp[i]; n = fn[i]
        prec = t / (t + f + 1e-9)
        rec  = t / (t + n + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        results[cls] = {"TP": t, "FP": f, "FN": n,
                        "precision": round(prec, 3),
                        "recall":    round(rec, 3),
                        "F1":        round(f1, 3)}
    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import yaml, cv2
    data_cfg = yaml.safe_load(VAL_YAML.read_text())
    base = Path(data_cfg["path"])
    val_img = base / data_cfg.get("val", "images/val")
    val_lbl = val_img.parent.parent / "labels" / "val"

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    all_results = {}
    for config_name, thresholds in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"{'='*60}")
        res = evaluate_config(model, val_img, val_lbl, thresholds)
        all_results[config_name] = res
        print(f"  {'Class':<22} {'P':>6} {'R':>6} {'F1':>6} {'TP':>6} {'FP':>6} {'FN':>6}")
        print(f"  {'-'*58}")
        for cls, m in res.items():
            print(f"  {cls:<22} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['F1']:>6.3f} {m['TP']:>6} {m['FP']:>6} {m['FN']:>6}")
        avg_r = np.mean([v['recall'] for v in res.values()])
        avg_p = np.mean([v['precision'] for v in res.values()])
        avg_f = np.mean([v['F1'] for v in res.values()])
        print(f"  {'MEAN':<22} {avg_p:>6.3f} {avg_r:>6.3f} {avg_f:>6.3f}")

    # Delta summary
    print(f"\n{'='*60}")
    print("DELTA: proposed_fix vs current_api")
    print(f"{'='*60}")
    curr = all_results[list(CONFIGS.keys())[0]]
    prop = all_results[list(CONFIGS.keys())[1]]
    print(f"  {'Class':<22} {'dRecall':>8} {'dPrecision':>11} {'dF1':>6}")
    print(f"  {'-'*50}")
    for cls in CLASSES:
        dr = prop[cls]['recall']    - curr[cls]['recall']
        dp = prop[cls]['precision'] - curr[cls]['precision']
        df = prop[cls]['F1']        - curr[cls]['F1']
        print(f"  {cls:<22} {dr:>+8.3f} {dp:>+11.3f} {df:>+6.3f}")

    out = OUT_DIR / "threshold_eval.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
