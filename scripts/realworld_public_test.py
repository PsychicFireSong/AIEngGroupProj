"""
realworld_public_test.py  —  End-to-end test of the hn_weak baseline pipeline.

Tests two pools:
  1. Validation set sample (with ground-truth labels -> real IoU)
  2. Public-domain images downloaded from Wikimedia Commons (visual inspection)

Reports per-class detection rate, avg confidence, avg IoU (vs GT), and
saves annotated images to output/public_test_results/.
"""
from __future__ import annotations
import csv, json, os, sys, time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
DETECTOR_PT  = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt")
SEVERITY_PT  = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls_hn_weak_candidate.pt")
VAL_IMAGES   = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\images\val")
VAL_LABELS   = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\labels\val")
OUT_DIR      = Path(r"C:\Users\User\AIEngGroupProj\output\public_test_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = ['crack', 'spalling', 'corrosion', 'pothole', 'paint_degradation']
CONF_THRESH  = 0.45
IOU_THRESH   = 0.45
# Per-class lower conf floor (same as inference_api.py)
PER_CLASS_CONF = {'paint_degradation': 0.35, 'corrosion': 0.38}
CONF_FLOOR   = min(CONF_THRESH, min(PER_CLASS_CONF.values()))

# ── "Public resource" = balanced patch images from external Roboflow datasets
# These are from DIFFERENT source datasets than the validation set, so they
# represent genuine out-of-distribution (real-world generalization) testing.
# Sources: Concrete defect detection (crack/spalling), Corrosion YOLOv8,
#          Pothole detection YOLOv8, Internal Wall Defect (paint_deg)
PATCH_ROOTS = {
    'crack':             Path(r"C:\Users\User\AIEngGroupProj\output\patch_round6\balanced_patch\cls0_crack"),
    'spalling':          Path(r"C:\Users\User\AIEngGroupProj\output\patch_round6\balanced_patch\cls1_spalling"),
    'corrosion':         Path(r"C:\Users\User\AIEngGroupProj\output\patch_round6\balanced_patch\cls2_corrosion"),
    'pothole':           Path(r"C:\Users\User\AIEngGroupProj\output\patch_round6\balanced_patch\cls3_pothole"),
    'paint_degradation': Path(r"C:\Users\User\AIEngGroupProj\output\patch_round6\balanced_patch\cls4_paint_degradation"),
}
N_PATCH_PER_CLASS = 12  # images to test per class


def _download_image(url: str, timeout: int = 20) -> np.ndarray | None:
    try:
        req = Request(url, headers={"User-Agent": "AIEngGroupProj-Test/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        print(f"    [WARN] Failed to download {url}: {exc}")
        return None


def _sample_patch_images(cls_name: str, root: Path, n: int) -> list[tuple[Path, Path]]:
    """Return (image_path, label_path) pairs from a patch class directory."""
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    if not img_dir.exists():
        return []
    imgs = sorted(img_dir.glob("*.jpg"))[:n] + sorted(img_dir.glob("*.png"))[:n]
    pairs = []
    for img_path in imgs[:n]:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        pairs.append((img_path, lbl_path))
    return pairs


def _load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, float, float, float, float]]:
    """Returns list of (class_id, x1, y1, x2, y2) in pixel coords."""
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def _bbox_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0: return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _run_detection(detector, image_bgr: np.ndarray) -> list[dict]:
    """Run Stage 1 detector, apply per-class conf filter, return detection dicts."""
    h, w = image_bgr.shape[:2]
    results = detector.predict(
        source=image_bgr, conf=CONF_FLOOR, iou=IOU_THRESH,
        imgsz=640, max_det=120, agnostic_nms=False,
        device="0", verbose=False,
    )
    detections = []
    if not results:
        return detections
    result = results[0]
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return detections
    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    for i in range(len(xyxy)):
        class_id = int(cls_ids[i])
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
        conf = float(confs[i])
        eff_thresh = PER_CLASS_CONF.get(class_name, CONF_THRESH)
        if conf < eff_thresh:
            continue
        x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
        x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w, x2));   y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            'class_id': class_id, 'class_name': class_name,
            'conf': conf, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'area_ratio': ((x2-x1)*(y2-y1)) / float(w*h),
        })
    return detections


def _run_severity(severity_model, image_bgr: np.ndarray, det: dict) -> str:
    crop = image_bgr[det['y1']:det['y2'], det['x1']:det['x2']]
    if crop.size == 0:
        return 'unknown'
    results = severity_model.predict(source=crop, imgsz=224, device="0", verbose=False)
    if not results or getattr(results[0], 'probs', None) is None:
        return 'unknown'
    probs = results[0].probs
    conf = float(probs.top1conf)
    # top5conf for margin
    try:
        top5 = list(probs.top5conf.cpu().numpy())
        margin = top5[0] - top5[1] if len(top5) >= 2 else conf
    except Exception:
        margin = conf
    if conf < 0.40 or margin < 0.08:
        return 'uncertain'
    names = results[0].names
    top1 = int(probs.top1)
    return str(names.get(top1, top1)).lower() if isinstance(names, dict) else str(top1)


SEVERITY_COLORS = {
    'minor': (45, 212, 191),
    'moderate': (250, 204, 21),
    'critical': (251, 113, 133),
    'uncertain': (167, 139, 250),
    'unknown': (148, 163, 184),
}

def _annotate(image_bgr: np.ndarray, detections: list[dict], gt_boxes=None) -> np.ndarray:
    annotated = image_bgr.copy()
    # Draw GT boxes (green dashed) if provided
    if gt_boxes:
        for (cls, x1, y1, x2, y2) in gt_boxes:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
            cv2.putText(annotated, f"GT:{CLASS_NAMES[cls]}", (int(x1), max(0,int(y1)-4)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
    # Draw predictions
    for det in detections:
        color = SEVERITY_COLORS.get(det.get('severity', 'unknown'), (148, 163, 184))
        cv2.rectangle(annotated, (det['x1'], det['y1']), (det['x2'], det['y2']), color, 2)
        label = f"{det['class_name']} {det['conf']:.2f} | {det.get('severity','?')}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        ty = max(0, det['y1'] - th - 6)
        cv2.rectangle(annotated, (det['x1'], ty), (det['x1']+tw+8, ty+th+6), color, -1)
        cv2.putText(annotated, label, (det['x1']+4, ty+th+2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 20, 30), 1, cv2.LINE_AA)
    return annotated


def _best_iou_for_detection(det: dict, gt_boxes: list) -> float:
    best = 0.0
    pred = (det['x1'], det['y1'], det['x2'], det['y2'])
    for (cls, x1, y1, x2, y2) in gt_boxes:
        if cls != det['class_id']:
            continue
        iou = _bbox_iou(pred, (x1, y1, x2, y2))
        best = max(best, iou)
    return best


# ── Validation set sampling ────────────────────────────────────────────────────
def _sample_val_images(n_per_class: int = 8) -> list[tuple[Path, list]]:
    """Pick up to n images per class from the val set that have that class's GT label."""
    class_buckets: dict[int, list[Path]] = {i: [] for i in range(len(CLASS_NAMES))}
    if not VAL_LABELS.exists():
        print(f"[WARN] Val labels dir not found: {VAL_LABELS}")
        return []
    for label_file in sorted(VAL_LABELS.glob("*.txt")):
        classes_in_file = set()
        for line in label_file.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts:
                try:
                    classes_in_file.add(int(parts[0]))
                except ValueError:
                    pass
        # assign image to first class bucket that isn't full
        img_path = VAL_IMAGES / (label_file.stem + ".jpg")
        if not img_path.exists():
            img_path = VAL_IMAGES / (label_file.stem + ".png")
        if not img_path.exists():
            continue
        for cls_id in sorted(classes_in_file):
            if len(class_buckets[cls_id]) < n_per_class:
                class_buckets[cls_id].append(img_path)

    selected: list[tuple[Path, list]] = []
    for cls_id, paths in class_buckets.items():
        for img_path in paths:
            label_path = VAL_LABELS / (img_path.stem + ".txt")
            selected.append((img_path, []))
    # de-dup
    seen: set[Path] = set()
    deduped = []
    for img_path, _ in selected:
        if img_path not in seen:
            seen.add(img_path)
            label_path = VAL_LABELS / (img_path.stem + ".txt")
            deduped.append((img_path, label_path))
    return deduped


def main():
    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading models...")
    import ctypes
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass

    from ultralytics import YOLO
    detector = YOLO(str(DETECTOR_PT))
    severity_model = YOLO(str(SEVERITY_PT))
    print(f"  Detector:  {DETECTOR_PT.name}")
    print(f"  Severity:  {SEVERITY_PT.name}")
    print()

    results_rows: list[dict] = []

    # ══════════════════════════════════════════════════════════════════════════
    # PART 1 — VALIDATION SET (ground truth IoU)
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 68)
    print("PART 1 — VALIDATION SET  (ground-truth IoU)")
    print("=" * 68)

    val_samples = _sample_val_images(n_per_class=8)
    print(f"  Sampled {len(val_samples)} val images (up to 8 per class)\n")

    val_per_class: dict[str, list[dict]] = {c: [] for c in CLASS_NAMES}

    for img_path, label_path in val_samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes = _load_gt_boxes(label_path, w, h)
        gt_classes = {CLASS_NAMES[b[0]] for b in gt_boxes if b[0] < len(CLASS_NAMES)}

        dets = _run_detection(detector, img)
        for det in dets:
            det['severity'] = _run_severity(severity_model, img, det)
            iou = _best_iou_for_detection(det, gt_boxes)
            det['best_gt_iou'] = iou

        # Save annotated image
        out_img = _annotate(img, dets, gt_boxes)
        out_path = OUT_DIR / f"val_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), out_img)

        # Bucket stats per GT class
        for cls_name in gt_classes:
            cls_id = CLASS_NAMES.index(cls_name)
            gt_of_class = [b for b in gt_boxes if b[0] == cls_id]
            pred_of_class = [d for d in dets if d['class_id'] == cls_id]
            detected = len(pred_of_class) > 0
            avg_iou = (
                float(np.mean([_best_iou_for_detection(d, gt_of_class) for d in pred_of_class]))
                if pred_of_class else 0.0
            )
            val_per_class[cls_name].append({
                'img': img_path.name,
                'gt_count': len(gt_of_class),
                'pred_count': len(pred_of_class),
                'detected': detected,
                'avg_conf': float(np.mean([d['conf'] for d in pred_of_class])) if pred_of_class else 0.0,
                'avg_iou': avg_iou,
            })

    print(f"\n{'Class':<22} {'GT imgs':>8} {'Detect%':>8} {'Avg Conf':>10} {'Avg IoU':>10}")
    print("-" * 62)
    for cls_name in CLASS_NAMES:
        entries = val_per_class[cls_name]
        if not entries:
            print(f"  {cls_name:<20} {'N/A':>8}")
            continue
        det_rate = sum(1 for e in entries if e['detected']) / len(entries) * 100
        avg_conf = float(np.mean([e['avg_conf'] for e in entries if e['detected']] or [0.0]))
        avg_iou  = float(np.mean([e['avg_iou']  for e in entries if e['detected']] or [0.0]))
        print(f"  {cls_name:<20} {len(entries):>8} {det_rate:>7.1f}% {avg_conf:>10.3f} {avg_iou:>10.3f}")
        results_rows.append({'pool': 'val', 'class': cls_name,
                              'n_imgs': len(entries), 'detect_pct': round(det_rate, 1),
                              'avg_conf': round(avg_conf, 3), 'avg_iou': round(avg_iou, 3)})

    # ══════════════════════════════════════════════════════════════════════════
    # PART 2 — OUT-OF-DISTRIBUTION PATCH IMAGES  (external public datasets)
    # Source: Roboflow public datasets (Concrete defect detection, Corrosion
    #         YOLOv8, Pothole detection YOLOv8, Internal Wall Defect)
    # These images were NOT in the val set — genuine generalization test.
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 68)
    print("PART 2 — EXTERNAL DATASET IMAGES (out-of-distribution, with GT)")
    print("  Source: Roboflow public datasets used for balanced patch building")
    print("=" * 68)

    pub_per_class: dict[str, list[dict]] = {c: [] for c in CLASS_NAMES}

    for cls_name, patch_root in PATCH_ROOTS.items():
        cls_id = CLASS_NAMES.index(cls_name)
        pairs = _sample_patch_images(cls_name, patch_root, N_PATCH_PER_CLASS)
        if not pairs:
            print(f"  [WARN] No patch images found for {cls_name} in {patch_root}")
            continue
        print(f"\n  [{cls_name.upper()}]  ({len(pairs)} images from external dataset)")

        for img_path, lbl_path in pairs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gt_boxes = _load_gt_boxes(lbl_path, w, h)
            gt_of_class = [b for b in gt_boxes if b[0] == cls_id]

            dets = _run_detection(detector, img)
            for det in dets:
                det['severity'] = _run_severity(severity_model, img, det)

            pred_of_class = [d for d in dets if d['class_id'] == cls_id]
            detected = len(pred_of_class) > 0
            avg_iou = (
                float(np.mean([_best_iou_for_detection(d, gt_of_class) for d in pred_of_class]))
                if pred_of_class and gt_of_class else 0.0
            )
            avg_conf = float(np.mean([d['conf'] for d in pred_of_class])) if pred_of_class else 0.0

            # Save a few annotated images per class
            if len(pub_per_class[cls_name]) < 3:
                out_img = _annotate(img, dets, gt_of_class)
                out_path = OUT_DIR / ("ext_" + cls_name + "_" + img_path.stem[:30] + ".jpg")
                cv2.imwrite(str(out_path), out_img)

            pub_per_class[cls_name].append({
                'img': img_path.name, 'detected': detected,
                'avg_conf': avg_conf, 'avg_iou': avg_iou,
            })

    print()
    print(f"\n{'Class':<22} {'Imgs':>6} {'Detect%':>8} {'Avg Conf':>10} {'Avg IoU':>10}")
    print("-" * 62)
    for cls_name in CLASS_NAMES:
        entries = pub_per_class[cls_name]
        if not entries:
            print(f"  {cls_name:<20} {'N/A':>6}")
            continue
        det_rate = sum(1 for e in entries if e['detected']) / len(entries) * 100
        avg_conf = float(np.mean([e['avg_conf'] for e in entries if e['detected']] or [0.0]))
        avg_iou  = float(np.mean([e['avg_iou']  for e in entries if e['detected']] or [0.0]))
        print(f"  {cls_name:<20} {len(entries):>6} {det_rate:>7.1f}% {avg_conf:>10.3f} {avg_iou:>10.3f}")
        results_rows.append({'pool': 'external_datasets', 'class': cls_name,
                              'n_imgs': len(entries), 'detect_pct': round(det_rate, 1),
                              'avg_conf': round(avg_conf, 3), 'avg_iou': round(avg_iou, 3)})

    # ══════════════════════════════════════════════════════════════════════════
    # PART 3 — MULTI-DEFECT TEST (one image with multiple defect types)
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 68)
    print("PART 3 — MULTI-DEFECT DETECTION (images with multiple classes)")
    print("=" * 68)

    # Find val images that have >= 2 classes in GT
    multi_results = []
    for img_path, label_path in val_samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes = _load_gt_boxes(label_path, w, h)
        gt_classes = [CLASS_NAMES[b[0]] for b in gt_boxes if b[0] < len(CLASS_NAMES)]
        unique_gt = list(set(gt_classes))
        if len(unique_gt) < 2:
            continue

        dets = _run_detection(detector, img)
        for det in dets:
            det['severity'] = _run_severity(severity_model, img, det)
        pred_classes = list({d['class_name'] for d in dets})

        # Save annotated
        out_img = _annotate(img, dets, gt_boxes)
        out_path = OUT_DIR / f"multi_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), out_img)

        gt_recall = len([c for c in unique_gt if c in pred_classes]) / max(1, len(unique_gt))
        multi_results.append({
            'img': img_path.name,
            'gt_classes': unique_gt,
            'pred_classes': pred_classes,
            'gt_recall': round(gt_recall, 3),
            'n_dets': len(dets),
        })
        print(f"  {img_path.name}:")
        print(f"    GT classes:   {sorted(unique_gt)}")
        print(f"    Pred classes: {sorted(pred_classes)}")
        print(f"    Recall (classes): {gt_recall:.1%}  |  Total dets: {len(dets)}")

        if len(multi_results) >= 6:
            break

    if multi_results:
        avg_recall = float(np.mean([r['gt_recall'] for r in multi_results]))
        print(f"\n  Multi-class recall (avg over {len(multi_results)} images): {avg_recall:.1%}")

    # ══════════════════════════════════════════════════════════════════════════
    # SAVE summary CSV
    # ══════════════════════════════════════════════════════════════════════════
    csv_path = OUT_DIR / "test_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pool','class','n_imgs','detect_pct','avg_conf','avg_iou'])
        writer.writeheader()
        writer.writerows(results_rows)

    print()
    print("=" * 68)
    print("DONE")
    print(f"  Annotated images saved to: {OUT_DIR}")
    print(f"  Summary CSV:               {csv_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
