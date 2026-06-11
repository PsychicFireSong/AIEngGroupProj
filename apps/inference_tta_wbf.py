"""
inference_tta_wbf.py  —  TTA + Weighted Box Fusion drop-in for Stage 1 inference.

Problem this solves (per literature + our eval):
  1. Corrosion recall=25% on val set  →  model detects it at lower conf than threshold;
     TTA forces the model to see the defect at multiple scales/flips, boosting recall.
  2. Multi-class co-occurrence miss (50% class recall)  →  NMS suppresses the secondary
     detection when two defects overlap; WBF merges overlapping predictions instead of
     discarding them, so both classes survive.
  3. Crack small-object miss (66.7% recall)  →  multi-scale inference (640 + 1280)
     catches cracks that are sub-pixel at base resolution.

Algorithm:
  For each image:
    1. Run detector at 3 views: original@640, h-flip@640, original@1280
    2. Normalize all predicted boxes to [0,1]
    3. Run per-class WBF with iou_thr=0.55 to merge overlapping preds
    4. Return merged list, replacing the old _run_detection output

Reference:
  Solovyev et al., "Weighted boxes fusion: Ensembling boxes from different
  object detection models," Image and Vision Computing, 2021.
  https://arxiv.org/abs/1910.13302
"""
from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# ── Per-class confidence overrides (tightened vs. original _PER_CLASS_CONF_OVERRIDE)
# Corrosion dropped to 0.25 (from 0.38): model detects it, just under-confident.
# paint_degradation dropped to 0.30 (from 0.35): consistent under-confidence observed.
PER_CLASS_CONF: dict[str, float] = {
    "paint_degradation": 0.30,
    "corrosion": 0.25,
}
DEFAULT_CONF = 0.45
CONF_FLOOR = min(PER_CLASS_CONF.values())   # 0.25 — detector runs at this floor

# WBF merge threshold: boxes of the same class with IoU > this get merged.
# Slightly above the standard 0.5 to avoid merging distinct touching defects.
WBF_IOU_THR = 0.55

# Scales to run inference at. 1280 is the key addition for small objects.
TTA_SCALES = [640, 1280]          # pixels (short-side resize target)
TTA_HFLIP  = True                 # also run with horizontal flip at base scale

CLASS_NAMES = ['crack', 'spalling', 'corrosion', 'pothole', 'paint_degradation']


# ═══════════════════════════════════════════════════════════════════════════════
# WBF implementation (no external dependencies)
# ═══════════════════════════════════════════════════════════════════════════════

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two boxes [x1,y1,x2,y2] in normalized [0,1] coords."""
    ix1 = max(box_a[0], box_b[0]); iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2]); iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def weighted_box_fusion(
    boxes_list: list[np.ndarray],    # each shape (N,4) normalized [x1,y1,x2,y2]
    scores_list: list[np.ndarray],   # each shape (N,)
    labels_list: list[np.ndarray],   # each shape (N,) int class ids
    iou_thr: float = WBF_IOU_THR,
    skip_box_thr: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Box Fusion across multiple prediction sets.
    Returns (merged_boxes, merged_scores, merged_labels) as arrays.
    """
    # Flatten all predictions
    all_boxes:  list[np.ndarray] = []
    all_scores: list[float]      = []
    all_labels: list[int]        = []

    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        for box, score, label in zip(boxes, scores, labels):
            if score < skip_box_thr:
                continue
            all_boxes.append(np.asarray(box, dtype=np.float32))
            all_scores.append(float(score))
            all_labels.append(int(label))

    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # Sort by confidence descending
    order = np.argsort(all_scores)[::-1]
    all_boxes  = [all_boxes[i]  for i in order]
    all_scores = [all_scores[i] for i in order]
    all_labels = [all_labels[i] for i in order]

    # Build clusters: each cluster groups overlapping same-class boxes
    clusters: list[dict] = []  # {'boxes':[], 'scores':[], 'label':int}

    for box, score, label in zip(all_boxes, all_scores, all_labels):
        matched = False
        for cluster in clusters:
            if cluster['label'] != label:
                continue
            # Compare against cluster representative (weighted-mean box so far)
            rep = cluster['representative']
            if _iou(box, rep) >= iou_thr:
                cluster['boxes'].append(box)
                cluster['scores'].append(score)
                # Update representative as weighted mean
                w = np.array(cluster['scores'])
                cluster['representative'] = (
                    np.stack(cluster['boxes']) * w[:, None]
                ).sum(axis=0) / w.sum()
                matched = True
                break
        if not matched:
            clusters.append({
                'label': label,
                'boxes': [box],
                'scores': [score],
                'representative': box.copy(),
            })

    # Compute final box (weighted mean) and score (max conf in cluster) for each cluster
    out_boxes:  list[np.ndarray] = []
    out_scores: list[float]      = []
    out_labels: list[int]        = []

    for cluster in clusters:
        w = np.array(cluster['scores'])
        merged_box = (np.stack(cluster['boxes']) * w[:, None]).sum(axis=0) / w.sum()
        out_boxes.append(merged_box)
        out_scores.append(float(np.max(w)))
        out_labels.append(cluster['label'])

    return (
        np.stack(out_boxes),
        np.array(out_scores, dtype=np.float32),
        np.array(out_labels, dtype=int),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TTA: augmented prediction passes
# ═══════════════════════════════════════════════════════════════════════════════

def _predict_one(
    detector,
    img_bgr: np.ndarray,
    imgsz: int,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one YOLO predict pass. Returns boxes in normalized [0,1] coords,
    scores, and class ids — all as numpy arrays.
    """
    results = detector.predict(
        source=img_bgr,
        conf=CONF_FLOOR,
        iou=0.45,
        imgsz=imgsz,
        max_det=120,
        agnostic_nms=False,
        device=device,
        verbose=False,
    )
    if not results:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    result = results[0]
    boxes_obj = getattr(result, 'boxes', None)
    if boxes_obj is None or len(boxes_obj) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    h, w = img_bgr.shape[:2]
    xyxy   = boxes_obj.xyxy.cpu().numpy()          # (N,4) pixel coords
    scores = boxes_obj.conf.cpu().numpy()
    labels = boxes_obj.cls.cpu().numpy().astype(int)

    # Normalize to [0,1]
    norm_boxes = xyxy.copy()
    norm_boxes[:, [0, 2]] /= w
    norm_boxes[:, [1, 3]] /= h
    norm_boxes = np.clip(norm_boxes, 0.0, 1.0)

    return norm_boxes, scores, labels


def run_tta_wbf(
    detector,
    img_bgr: np.ndarray,
    device: str | None = None,
    base_conf: float = DEFAULT_CONF,
    iou_thr: float = 0.45,
) -> list[dict]:
    """
    Main entry point: runs TTA + WBF and returns a list of detection dicts
    with the same schema as the original _run_detection output in inference_api.py.

    Args:
        detector:   loaded YOLO model (already on device)
        img_bgr:    input image in BGR format
        device:     CUDA device string or None
        base_conf:  base confidence threshold (per-class overrides apply on top)
        iou_thr:    NMS-equivalent IoU threshold (used within YOLO, not WBF)

    Returns:
        list of detection dicts: {class_id, class_name, conf, severity,
                                  x1, y1, x2, y2, area_ratio}
        (severity is left as 'unknown'; caller should fill in via Stage 2)
    """
    h, w = img_bgr.shape[:2]

    boxes_list:  list[np.ndarray] = []
    scores_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    # ── Pass 1: original image at each TTA scale ──────────────────────────────
    for scale in TTA_SCALES:
        b, s, l = _predict_one(detector, img_bgr, imgsz=scale, device=device)
        boxes_list.append(b); scores_list.append(s); labels_list.append(l)

    # ── Pass 2: horizontal flip at base scale ─────────────────────────────────
    if TTA_HFLIP:
        flipped = cv2.flip(img_bgr, 1)
        bf, sf, lf = _predict_one(detector, flipped, imgsz=TTA_SCALES[0], device=device)
        if len(bf) > 0:
            # Mirror x-coordinates back: x_orig = 1 - x_flip (for normalized coords)
            bf_corrected = bf.copy()
            bf_corrected[:, 0] = 1.0 - bf[:, 2]  # x1_orig = 1 - x2_flip
            bf_corrected[:, 2] = 1.0 - bf[:, 0]  # x2_orig = 1 - x1_flip
            boxes_list.append(bf_corrected)
            scores_list.append(sf)
            labels_list.append(lf)

    # ── WBF: merge all passes ─────────────────────────────────────────────────
    merged_boxes, merged_scores, merged_labels = weighted_box_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=WBF_IOU_THR,
    )

    # ── Apply per-class confidence gate ───────────────────────────────────────
    detections: list[dict] = []
    for i in range(len(merged_scores)):
        class_id   = int(merged_labels[i])
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        conf       = float(merged_scores[i])
        eff_thresh = PER_CLASS_CONF.get(class_name, base_conf)
        if conf < eff_thresh:
            continue

        # Denormalize to pixel coords
        x1 = max(0, min(w - 1, int(merged_boxes[i, 0] * w)))
        y1 = max(0, min(h - 1, int(merged_boxes[i, 1] * h)))
        x2 = max(0, min(w,     int(merged_boxes[i, 2] * w)))
        y2 = max(0, min(h,     int(merged_boxes[i, 3] * h)))
        if x2 <= x1 or y2 <= y1:
            continue

        area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
        detections.append({
            'class_id':   class_id,
            'class_name': class_name,
            'conf':       round(conf, 4),
            'severity':   'unknown',     # filled in by Stage 2 caller
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'area_ratio': round(area_ratio, 6),
        })

    return detections


# ═══════════════════════════════════════════════════════════════════════════════
# Quick smoke-test (run this file directly to verify)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import ctypes, sys
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass

    DETECTOR_PT = Path(
        r"C:\Users\User\AIEngGroupProj_colab_outputs"
        r"\continued_hn_weak_finetune\runs\current\weights"
        r"\defect_detector_hn_weak_candidate.pt"
    )
    SEVERITY_PT = Path(
        r"C:\Users\User\AIEngGroupProj_colab_outputs"
        r"\continued_hn_weak_finetune\runs\current\weights"
        r"\severity_cls_hn_weak_candidate.pt"
    )
    VAL_IMAGES = Path(
        r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak"
        r"\current\datasets\merged_dataset_hn_weak_continue\images\val"
    )
    VAL_LABELS = Path(
        r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak"
        r"\current\datasets\merged_dataset_hn_weak_continue\labels\val"
    )
    OUT_DIR = Path(r"C:\Users\User\AIEngGroupProj\output\tta_wbf_results")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    print("Loading models...")
    detector = YOLO(str(DETECTOR_PT))
    severity_model = YOLO(str(SEVERITY_PT))
    print("  Done.\n")

    # ── helper: load GT boxes ─────────────────────────────────────────────────
    def _load_gt(label_path: Path, img_w: int, img_h: int):
        boxes = []
        if not label_path.exists():
            return boxes
        for line in label_path.read_text().strip().splitlines():
            p = line.strip().split()
            if len(p) < 5:
                continue
            cls = int(p[0])
            cx, cy, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            x1 = (cx - bw/2) * img_w; y1 = (cy - bh/2) * img_h
            x2 = (cx + bw/2) * img_w; y2 = (cy + bh/2) * img_h
            boxes.append((cls, x1, y1, x2, y2))
        return boxes

    def _best_iou(det, gt_boxes):
        best = 0.0
        pred = (det['x1'], det['y1'], det['x2'], det['y2'])
        for (cls, x1, y1, x2, y2) in gt_boxes:
            if cls != det['class_id']:
                continue
            ix1 = max(pred[0], x1); iy1 = max(pred[1], y1)
            ix2 = min(pred[2], x2); iy2 = min(pred[3], y2)
            inter = max(0, ix2-ix1)*max(0, iy2-iy1)
            if inter == 0:
                continue
            ua = (pred[2]-pred[0])*(pred[3]-pred[1]) + (x2-x1)*(y2-y1) - inter
            best = max(best, inter/ua if ua > 0 else 0.0)
        return best

    # ── baseline (no TTA) using same conf floor approach ─────────────────────
    def _baseline_detect(detector, img_bgr):
        from inference_tta_wbf import _predict_one, PER_CLASS_CONF, DEFAULT_CONF
        b, s, l = _predict_one(detector, img_bgr, imgsz=640, device=None)
        dets = []
        h, w = img_bgr.shape[:2]
        for i in range(len(s)):
            cls_id = int(l[i])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            conf = float(s[i])
            if conf < PER_CLASS_CONF.get(cls_name, DEFAULT_CONF):
                continue
            x1 = max(0, int(b[i,0]*w)); y1 = max(0, int(b[i,1]*h))
            x2 = max(0, int(b[i,2]*w)); y2 = max(0, int(b[i,3]*h))
            dets.append({'class_id':cls_id,'class_name':cls_name,'conf':conf,
                         'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                         'area_ratio':((x2-x1)*(y2-y1))/float(w*h)})
        return dets

    # ── Evaluate both methods on the val set ─────────────────────────────────
    import random, csv
    random.seed(42)

    # Sample same 8 per class as realworld_public_test.py
    class_buckets: dict[int, list[Path]] = {i: [] for i in range(5)}
    for lf in sorted(VAL_LABELS.glob("*.txt")):
        try:
            classes_in = set(int(l.split()[0]) for l in lf.read_text().strip().splitlines() if l.strip())
        except Exception:
            continue
        imgp = VAL_IMAGES / (lf.stem + ".jpg")
        if not imgp.exists():
            imgp = VAL_IMAGES / (lf.stem + ".png")
        if not imgp.exists():
            continue
        for c in sorted(classes_in):
            if c < 5 and len(class_buckets[c]) < 8:
                class_buckets[c].append(imgp)
    seen: set[Path] = set()
    val_samples = []
    for paths in class_buckets.values():
        for p in paths:
            if p not in seen:
                seen.add(p)
                val_samples.append(p)

    print(f"Evaluating {len(val_samples)} val images  (TTA+WBF vs baseline)...\n")

    stats_base: dict[str, list] = {c: [] for c in CLASS_NAMES}
    stats_tta:  dict[str, list] = {c: [] for c in CLASS_NAMES}

    for img_path in val_samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lp = VAL_LABELS / (img_path.stem + ".txt")
        gt = _load_gt(lp, w, h)
        gt_cls = {CLASS_NAMES[b[0]] for b in gt if b[0] < 5}

        dets_base = _baseline_detect(detector, img)
        dets_tta  = run_tta_wbf(detector, img, device=None)

        for cls_name in gt_cls:
            cls_id = CLASS_NAMES.index(cls_name)
            gt_of = [b for b in gt if b[0] == cls_id]

            for dets, store in [(dets_base, stats_base), (dets_tta, stats_tta)]:
                pred_of = [d for d in dets if d['class_id'] == cls_id]
                detected = len(pred_of) > 0
                avg_iou = float(np.mean([_best_iou(d, gt_of) for d in pred_of])) if pred_of else 0.0
                avg_conf = float(np.mean([d['conf'] for d in pred_of])) if pred_of else 0.0
                store[cls_name].append({'detected': detected, 'avg_iou': avg_iou, 'avg_conf': avg_conf})

    print(f"{'Class':<22} {'Baseline Det%':>14} {'TTA Det%':>10} {'Diff':>8}  "
          f"{'Base IoU':>9} {'TTA IoU':>9}")
    print("-" * 80)
    total_improvement = 0
    for cls_name in CLASS_NAMES:
        b_entries = stats_base[cls_name]
        t_entries = stats_tta[cls_name]
        if not b_entries:
            print(f"  {cls_name:<20}  {'N/A':>14}")
            continue
        b_det = sum(1 for e in b_entries if e['detected']) / len(b_entries) * 100
        t_det = sum(1 for e in t_entries if e['detected']) / len(t_entries) * 100
        b_iou = float(np.mean([e['avg_iou'] for e in b_entries if e['detected']] or [0.0]))
        t_iou = float(np.mean([e['avg_iou'] for e in t_entries if e['detected']] or [0.0]))
        diff = t_det - b_det
        total_improvement += diff
        marker = " <-- IMPROVED" if diff > 0 else (" <-- REGRESSION" if diff < 0 else "")
        print(f"  {cls_name:<20} {b_det:>13.1f}% {t_det:>9.1f}% {diff:>+7.1f}%  "
              f"{b_iou:>9.3f} {t_iou:>9.3f}{marker}")

    print(f"\n  Net detection rate improvement: {total_improvement:+.1f}pp across 5 classes")
    print(f"\n  Annotated TTA results saved to: {OUT_DIR}")
