"""
eval_tta_enhanced.py — Benchmark mAP50 for 3 TTA configs on the val set.

Configs:
  A) Baseline      : single pass at imgsz=640, no flip/rotate
  B) Current TTA   : 640 + 1280 + hflip@640  (already in inference_tta_wbf.py)
  C) Enhanced TTA  : 640 + 1280 + hflip + vflip + rotate90 (4 orientations)

For each config, per-class AP50 and overall mAP50 are computed using the full
val set (3,855 images) via standard VOC AP (area under PR curve, interpolated).

Expected improvement from enhanced TTA vs baseline:
  - Vertical flip: helps horizontal surfaces (floors, roads, walls with gravity defects)
  - Rotate90: thin cracks can appear at any orientation; 4-view ensemble captures all
  - Improvement estimate from literature: +1-2 mAP50 points on top of existing TTA
"""
from __future__ import annotations
import collections
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
WEIGHTS  = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs"
                r"\continued_hn_weak_finetune\runs\current\weights"
                r"\defect_detector_hn_weak_candidate.pt")
VAL_IMG  = Path(r"C:\Users\User\AIEngGroupProj\output\v7_lean_dataset\images\val")
VAL_LBL  = Path(r"C:\Users\User\AIEngGroupProj\output\v7_lean_dataset\labels\val")
OUT_DIR  = Path(r"C:\Users\User\AIEngGroupProj\output\tta_enhanced_eval")

CLASSES  = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
CONF_FLOOR   = 0.10   # run all configs at this floor; mAP sweeps above it
IOU_THRESH   = 0.50
WBF_IOU_THR  = 0.55
DEVICE       = "0"    # CUDA device


# ── WBF (same algorithm as inference_tta_wbf.py) ─────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def wbf(boxes_list, scores_list, labels_list, iou_thr=WBF_IOU_THR):
    all_b, all_s, all_l = [], [], []
    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        for box, score, label in zip(boxes, scores, labels):
            all_b.append(np.asarray(box, dtype=np.float32))
            all_s.append(float(score))
            all_l.append(int(label))
    if not all_b:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    order = np.argsort(all_s)[::-1]
    all_b = [all_b[i] for i in order]
    all_s = [all_s[i] for i in order]
    all_l = [all_l[i] for i in order]
    clusters: list[dict] = []
    for box, score, label in zip(all_b, all_s, all_l):
        matched = False
        for cl in clusters:
            if cl['label'] != label:
                continue
            if _iou(box, cl['rep']) >= iou_thr:
                cl['boxes'].append(box); cl['scores'].append(score)
                w = np.array(cl['scores'])
                cl['rep'] = (np.stack(cl['boxes']) * w[:, None]).sum(0) / w.sum()
                matched = True; break
        if not matched:
            clusters.append({'label': label, 'boxes': [box],
                              'scores': [score], 'rep': box.copy()})
    out_b, out_s, out_l = [], [], []
    for cl in clusters:
        w = np.array(cl['scores'])
        out_b.append((np.stack(cl['boxes']) * w[:, None]).sum(0) / w.sum())
        out_s.append(float(np.max(w)))
        out_l.append(cl['label'])
    return np.stack(out_b), np.array(out_s), np.array(out_l, dtype=int)


# ── Single YOLO pass ──────────────────────────────────────────────────────────

def predict_one(model, img_bgr: np.ndarray, imgsz: int):
    """Returns (boxes_norm, scores, labels) as numpy arrays."""
    results = model.predict(source=img_bgr, conf=CONF_FLOOR, iou=0.45,
                            imgsz=imgsz, max_det=150, verbose=False, device=DEVICE)
    if not results:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    h, w = img_bgr.shape[:2]
    xyxy   = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    labels = r.boxes.cls.cpu().numpy().astype(int)
    norm   = xyxy.copy()
    norm[:, [0, 2]] /= w
    norm[:, [1, 3]] /= h
    return np.clip(norm, 0, 1), scores, labels


# ── Augmentation passes per config ────────────────────────────────────────────

def get_passes_A(model, img):
    """Config A: single 640 pass."""
    b, s, l = predict_one(model, img, 640)
    return [b], [s], [l]


def get_passes_B(model, img):
    """Config B: 640 + 1280 + hflip@640 (current TTA)."""
    boxes_list, scores_list, labels_list = [], [], []
    for sz in [640, 1280]:
        b, s, l = predict_one(model, img, sz)
        boxes_list.append(b); scores_list.append(s); labels_list.append(l)
    flipped = cv2.flip(img, 1)
    bf, sf, lf = predict_one(model, flipped, 640)
    if len(bf) > 0:
        bfc = bf.copy()
        bfc[:, 0] = 1.0 - bf[:, 2]
        bfc[:, 2] = 1.0 - bf[:, 0]
        boxes_list.append(bfc); scores_list.append(sf); labels_list.append(lf)
    return boxes_list, scores_list, labels_list


def get_passes_C(model, img):
    """Config C: 640 + 1280 + hflip + vflip + 4×rotate90."""
    boxes_list, scores_list, labels_list = get_passes_B(model, img)  # start from B

    # Vertical flip
    vflipped = cv2.flip(img, 0)
    bv, sv, lv = predict_one(model, vflipped, 640)
    if len(bv) > 0:
        bvc = bv.copy()
        bvc[:, 1] = 1.0 - bv[:, 3]   # y1_orig = 1 - y2_flip
        bvc[:, 3] = 1.0 - bv[:, 1]   # y2_orig = 1 - y1_flip
        boxes_list.append(bvc); scores_list.append(sv); labels_list.append(lv)

    # 4 rotations: 90, 180, 270 degrees
    for k in [1, 2, 3]:  # cv2.ROTATE_90_CLOCKWISE, etc.
        rot_code = [cv2.ROTATE_90_CLOCKWISE,
                    cv2.ROTATE_180,
                    cv2.ROTATE_90_COUNTERCLOCKWISE][k - 1]
        rotated = cv2.rotate(img, rot_code)
        br, sr, lr = predict_one(model, rotated, 640)
        if len(br) == 0:
            boxes_list.append(br); scores_list.append(sr); labels_list.append(lr)
            continue
        # Un-rotate normalized boxes back to original coords
        brc = br.copy()
        if k == 1:   # 90° clockwise: (x,y) → (1-y, x)
            brc[:, 0] = 1.0 - br[:, 3]   # x1 = 1 - y2
            brc[:, 1] = br[:, 0]          # y1 = x1_rot
            brc[:, 2] = 1.0 - br[:, 1]   # x2 = 1 - y1
            brc[:, 3] = br[:, 2]          # y2 = x2_rot
        elif k == 2: # 180°: (x,y) → (1-x, 1-y)
            brc[:, 0] = 1.0 - br[:, 2]
            brc[:, 1] = 1.0 - br[:, 3]
            brc[:, 2] = 1.0 - br[:, 0]
            brc[:, 3] = 1.0 - br[:, 1]
        elif k == 3: # 270° clockwise (= 90° CCW): (x,y) → (y, 1-x)
            brc[:, 0] = br[:, 1]
            brc[:, 1] = 1.0 - br[:, 2]
            brc[:, 2] = br[:, 3]
            brc[:, 3] = 1.0 - br[:, 0]
        # Ensure x1<x2, y1<y2
        brc[:, [0, 2]] = np.sort(brc[:, [0, 2]], axis=1)
        brc[:, [1, 3]] = np.sort(brc[:, [1, 3]], axis=1)
        boxes_list.append(brc); scores_list.append(sr); labels_list.append(lr)

    return boxes_list, scores_list, labels_list


# ── AP computation (VOC 11-point interpolation) ───────────────────────────────

def compute_ap50(all_preds: list[tuple[float, bool]], n_gt: int) -> float:
    """all_preds: list of (conf, is_tp) sorted by conf desc."""
    if n_gt == 0:
        return 0.0
    all_preds.sort(key=lambda x: -x[0])
    tp_cum = 0; fp_cum = 0
    precs, recs = [], []
    for _, is_tp in all_preds:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precs.append(tp_cum / (tp_cum + fp_cum))
        recs.append(tp_cum / n_gt)
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_at_t = [p for p, r in zip(precs, recs) if r >= t]
        ap += max(p_at_t) / 11.0 if p_at_t else 0.0
    return ap


def iou_boxes(pred_xyxy, gt_xyxy) -> float:
    ix1 = max(pred_xyxy[0], gt_xyxy[0]); iy1 = max(pred_xyxy[1], gt_xyxy[1])
    ix2 = min(pred_xyxy[2], gt_xyxy[2]); iy2 = min(pred_xyxy[3], gt_xyxy[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1 = (pred_xyxy[2]-pred_xyxy[0])*(pred_xyxy[3]-pred_xyxy[1])
    a2 = (gt_xyxy[2]-gt_xyxy[0])*(gt_xyxy[3]-gt_xyxy[1])
    return inter / (a1 + a2 - inter + 1e-9)


# ── Evaluate one config on the full val set ───────────────────────────────────

def evaluate_config(model, config_fn, config_name: str) -> dict:
    """Returns per-class AP50 and mAP50."""
    img_paths = sorted(VAL_IMG.glob("*.jpg"))
    # per-class: list of (conf, is_tp) for AP computation
    pred_records: dict[int, list] = {i: [] for i in range(5)}
    gt_counts:    dict[int, int]  = {i: 0  for i in range(5)}

    t0 = time.time()
    for idx, img_path in enumerate(img_paths):
        lbl_path = VAL_LBL / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        # Load GT
        gt_per_cls: dict[int, list] = collections.defaultdict(list)
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (cx - bw/2) * W; y1 = (cy - bh/2) * H
                x2 = (cx + bw/2) * W; y2 = (cy + bh/2) * H
                gt_per_cls[cid].append([x1, y1, x2, y2])
                gt_counts[cid] += 1

        # Run inference passes + WBF
        bl, sl, ll = config_fn(model, img)
        merged_boxes, merged_scores, merged_labels = wbf(bl, sl, ll)

        # Denormalize merged boxes to pixel coords
        if len(merged_boxes) > 0:
            merged_boxes_px = merged_boxes.copy()
            merged_boxes_px[:, [0, 2]] *= W
            merged_boxes_px[:, [1, 3]] *= H
        else:
            merged_boxes_px = merged_boxes

        # Match preds to GT per class
        # Greedy matching: sort by conf desc within each class
        for cid in range(5):
            cls_mask = (merged_labels == cid)
            if not cls_mask.any():
                continue
            cls_boxes  = merged_boxes_px[cls_mask]
            cls_scores = merged_scores[cls_mask]
            cls_gt     = [np.array(b) for b in gt_per_cls.get(cid, [])]
            matched_gt = set()
            order = np.argsort(cls_scores)[::-1]
            for pi in order:
                best_iou, best_gi = 0.0, -1
                for gi, gt_box in enumerate(cls_gt):
                    if gi in matched_gt:
                        continue
                    v = iou_boxes(cls_boxes[pi], gt_box)
                    if v > best_iou:
                        best_iou, best_gi = v, gi
                is_tp = best_iou >= IOU_THRESH and best_gi >= 0
                if is_tp:
                    matched_gt.add(best_gi)
                pred_records[cid].append((float(cls_scores[pi]), is_tp))

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(img_paths) - idx - 1)
            print(f"  [{config_name}] {idx+1}/{len(img_paths)}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    per_class_ap = {}
    for cid, cls_name in enumerate(CLASSES):
        ap = compute_ap50(pred_records[cid], gt_counts[cid])
        per_class_ap[cls_name] = round(ap, 4)
    map50 = round(float(np.mean(list(per_class_ap.values()))), 4)
    elapsed = round(time.time() - t0, 1)
    print(f"  [{config_name}] Done in {elapsed}s  mAP50={map50:.4f}")
    return {'map50': map50, 'per_class': per_class_ap, 'elapsed_s': elapsed}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    print("Loading model...")
    model = YOLO(str(WEIGHTS))
    print("  Done.\n")

    configs = [
        ("A_baseline",     get_passes_A),
        ("B_current_TTA",  get_passes_B),
        ("C_enhanced_TTA", get_passes_C),
    ]

    results = {}
    for name, fn in configs:
        print(f"\n{'='*60}")
        print(f"Config {name}")
        print(f"{'='*60}")
        results[name] = evaluate_config(model, fn, name)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Class':<24}", end="")
    for name in results:
        print(f"  {name:<22}", end="")
    print()
    print("-" * 70)
    for cls in CLASSES:
        print(f"  {cls:<22}", end="")
        for name, res in results.items():
            print(f"  {res['per_class'][cls]:.4f}              ", end="")
        print()
    print("-" * 70)
    print(f"  {'mAP50':<22}", end="")
    for name, res in results.items():
        print(f"  {res['map50']:.4f}              ", end="")
    print()

    # Deltas vs baseline
    base = results["A_baseline"]
    print(f"\n{'='*70}")
    print("DELTA vs A_baseline")
    print(f"{'='*70}")
    for name, res in results.items():
        if name == "A_baseline":
            continue
        delta_map = res['map50'] - base['map50']
        print(f"\n  {name}  (mAP50 delta = {delta_map:+.4f})")
        for cls in CLASSES:
            d = res['per_class'][cls] - base['per_class'][cls]
            print(f"    {cls:<22} {d:+.4f}")

    # Save JSON
    out_json = OUT_DIR / "tta_enhanced_eval.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
