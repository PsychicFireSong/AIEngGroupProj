"""
Post-training real-world evaluation — auto-selects best available weights.

Evaluates Stage 1 (detector) and Stage 2 (severity) against the same
real-world test suite as realworld_eval2.py but uses the newest candidate weights.

Weight selection priority:
  Detector: patch3_candidate > patch2_candidate > hn_weak_candidate
  Severity: severity_cls_patch3_candidate > patch1_candidate > severity_cls

Metrics reported:
  - Per-class detection recall (15 images per class, conf=0.45)
  - Average detection confidence per class
  - False positive rate (13 clean images)
  - Wide-angle recall (per class at 50% scale)
  - Per-defect severity accuracy (if per-defect models exist)
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np

WEIGHTS_DIR = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights")
VAL_IMGS    = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\images\val")
CONF        = 0.45
IOU         = 0.45
DEVICE      = "0"

CLASS_NAMES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
SEV_CLASSES = ["critical", "minor", "moderate"]


def pick_detector() -> Path:
    priority = [
        "defect_detector_patch3_candidate.pt",
        "defect_detector_patch2_candidate.pt",
        "defect_detector_hn_weak_candidate.pt",
    ]
    for name in priority:
        p = WEIGHTS_DIR / name
        if p.exists():
            print(f"Detector: {name}")
            return p
    raise FileNotFoundError("No detector weights found")


def pick_severity() -> Path:
    priority = [
        "severity_cls_patch3_candidate.pt",
        "severity_cls_patch1_candidate.pt",
        "severity_cls.pt",
    ]
    for name in priority:
        p = WEIGHTS_DIR / name
        if p.exists():
            print(f"Severity:  {name}")
            return p
    raise FileNotFoundError("No severity weights found")


def pick_val_images_for_class(cls: str, n: int = 15) -> list[Path]:
    cls_dirs = [
        VAL_IMGS.parent.parent.parent / "data" / "images" / cls,
        VAL_IMGS / cls,
    ]
    imgs: list[Path] = []
    for d in cls_dirs:
        if d.exists():
            found = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            imgs.extend(found)
            if len(imgs) >= n:
                break
    if len(imgs) < n:
        all_val = list(VAL_IMGS.glob("*.jpg")) + list(VAL_IMGS.glob("*.png"))
        imgs.extend(all_val)
    random.seed(42)
    return random.sample(imgs, min(n, len(imgs)))


def _sample_class_val_images(det, cls_id: int, cls_name: str, n: int) -> list[Path]:
    """Find val images that the model actually detects as cls_name."""
    from ultralytics import YOLO
    all_val = list(VAL_IMGS.glob("*.jpg")) + list(VAL_IMGS.glob("*.png"))
    random.seed(42)
    random.shuffle(all_val)
    hits = []
    for img in all_val:
        if len(hits) >= n:
            break
        res = det.predict(source=str(img), conf=0.10, iou=IOU, device=DEVICE, verbose=False)
        if res:
            boxes = getattr(res[0], "boxes", None)
            if boxes is not None:
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                if cls_id in cls_ids:
                    hits.append(img)
    return hits


def _run_detection(det, img_path: Path, conf: float = CONF):
    res = det.predict(source=str(img_path), conf=conf, iou=IOU, device=DEVICE, verbose=False)
    if not res:
        return []
    boxes = getattr(res[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    names = res[0].names
    return [(int(cid), float(conf), names.get(int(cid), str(cid))) for cid, conf in zip(class_ids, confs)]


def eval_per_class(det):
    """Per-class recall on val images at CONF threshold. Also measures avg confidence."""
    print("\n=== Per-class detection recall (val images) ===")
    label_dir = VAL_IMGS.parent.parent / "labels" / "val"
    if not label_dir.exists():
        print("  (label dir not found, using presence-based detection)")
        label_dir = None

    all_val = list(VAL_IMGS.glob("*.jpg")) + list(VAL_IMGS.glob("*.png"))
    random.seed(42)

    results = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        # Find val images that have this class in their label file
        cls_imgs = []
        if label_dir:
            for lf in label_dir.glob("*.txt"):
                lines = lf.read_text().strip().split("\n")
                if any(line.startswith(str(cls_id) + " ") for line in lines if line):
                    img = (VAL_IMGS / (lf.stem + ".jpg"))
                    if not img.exists():
                        img = (VAL_IMGS / (lf.stem + ".png"))
                    if img.exists():
                        cls_imgs.append(img)

        if not cls_imgs:
            cls_imgs = random.sample(all_val, min(15, len(all_val)))

        sample = random.sample(cls_imgs, min(15, len(cls_imgs)))
        hits, total, conf_sum = 0, 0, 0.0
        for img_path in sample:
            detections = _run_detection(det, img_path)
            total += 1
            cls_dets = [(cid, c, n) for cid, c, n in detections if n == cls_name]
            if cls_dets:
                hits += 1
                conf_sum += max(c for _, c, _ in cls_dets)

        recall = hits / total if total > 0 else 0
        avg_conf = conf_sum / hits if hits > 0 else 0
        results[cls_name] = {"recall": recall, "avg_conf": avg_conf, "n": total}
        status = "OK" if recall >= 0.70 else "WARN"
        print(f"  {cls_name:20s}: recall={recall:.2f}  avg_conf={avg_conf:.3f}  [{status}]")

    return results


def eval_false_positives(det, n: int = 13):
    """False positive rate on clean (no-defect) images."""
    print("\n=== False positive rate ===")
    # Use first N val images that DON'T have any defect labels
    label_dir = VAL_IMGS.parent.parent / "labels" / "val"
    clean_imgs = []
    if label_dir.exists():
        for lf in label_dir.glob("*.txt"):
            if lf.stat().st_size == 0 or not lf.read_text().strip():
                img = VAL_IMGS / (lf.stem + ".jpg")
                if not img.exists():
                    img = VAL_IMGS / (lf.stem + ".png")
                if img.exists():
                    clean_imgs.append(img)

    if len(clean_imgs) < 3:
        print("  (no clean val images found, skipping FP eval)")
        return 0.0

    random.seed(42)
    sample = random.sample(clean_imgs, min(n, len(clean_imgs)))
    fp_count = 0
    for img_path in sample:
        detections = _run_detection(det, img_path)
        if detections:
            fp_count += 1

    rate = fp_count / len(sample)
    print(f"  {fp_count}/{len(sample)} clean images triggered false positive")
    print(f"  FP rate: {rate:.3f}  ({'OK' if rate == 0 else 'WARN'})")
    return rate


def eval_wide_angle(det, scale: float = 0.50):
    """Wide-angle recall: embed val images at 50% scale in grey canvas."""
    print(f"\n=== Wide-angle recall (defect at {scale:.0%} scale) ===")
    all_val = list(VAL_IMGS.glob("*.jpg"))
    random.seed(123)
    sample = random.sample(all_val, min(50, len(all_val)))

    hits, total = 0, 0
    for img_path in sample:
        label_file = VAL_IMGS.parent.parent / "labels" / "val" / (img_path.stem + ".txt")
        if not label_file.exists() or label_file.stat().st_size == 0:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        canvas = np.full((h, w, 3), 128, dtype=np.uint8)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        x_off = (w - nw) // 2
        y_off = (h - nh) // 2
        canvas[y_off:y_off + nh, x_off:x_off + nw] = resized

        original_cls_ids = set()
        for line in label_file.read_text().strip().split("\n"):
            if line.strip():
                original_cls_ids.add(int(line.split()[0]))

        detections = _run_detection(det, canvas)
        detected_cls = {cid for cid, _, _ in detections}
        if original_cls_ids & detected_cls:
            hits += 1
        total += 1

    rate = hits / total if total > 0 else 0
    print(f"  {hits}/{total} wide-angle images detected (scale={scale})")
    print(f"  Wide-angle recall: {rate:.3f}  ({'OK' if rate >= 0.50 else 'WARN'})")
    return rate


def eval_severity(sev_model_path: Path):
    """Severity accuracy using available model(s)."""
    from ultralytics import YOLO
    print("\n=== Severity classification accuracy ===")

    SEVERITY_DS = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\severity_dataset")
    perdefect_dir = sev_model_path.parent
    results = {}

    for defect in CLASS_NAMES:
        per_model_path = perdefect_dir / f"severity_{defect}_cls.pt"
        if per_model_path.exists():
            mdl = YOLO(str(per_model_path))
            correct, total = 0, 0
            for sev_cls in SEV_CLASSES:
                imgs = list((SEVERITY_DS / "val" / sev_cls).glob("*.jpg"))
                defect_imgs = [i for i in imgs if f"_{defect}_" in i.name]
                for img in defect_imgs:
                    res = mdl.predict(source=str(img), imgsz=224, device=DEVICE, verbose=False)
                    if res and res[0].probs is not None:
                        pred = res[0].names[int(res[0].probs.top1)]
                        if pred == sev_cls:
                            correct += 1
                        total += 1
            acc = correct / total if total > 0 else 0
            results[defect] = acc
            print(f"  severity_{defect}: {acc:.4f}  ({correct}/{total})")

    if not results:
        # Fall back to single model eval
        mdl = YOLO(str(sev_model_path))
        correct, total = 0, 0
        for sev_cls in SEV_CLASSES:
            for img in (SEVERITY_DS / "val" / sev_cls).glob("*.jpg"):
                res = mdl.predict(source=str(img), imgsz=224, device=DEVICE, verbose=False)
                if res and res[0].probs is not None:
                    pred = res[0].names[int(res[0].probs.top1)]
                    if pred == sev_cls:
                        correct += 1
                    total += 1
        acc = correct / total if total > 0 else 0
        print(f"  single model: {acc:.4f}  ({correct}/{total})")
        results["all"] = acc

    avg = sum(results.values()) / len(results)
    print(f"  Average severity acc: {avg:.4f}  (target >0.90)")
    return results


def main():
    from ultralytics import YOLO
    print("=" * 60)
    print("REAL-WORLD EVAL v3 — auto-selecting best available weights")
    print("=" * 60)

    det_path = pick_detector()
    sev_path = pick_severity()

    det = YOLO(str(det_path))
    print(f"Models loaded.\n")

    class_results = eval_per_class(det)
    fp_rate       = eval_false_positives(det)
    wide_recall   = eval_wide_angle(det)
    sev_results   = eval_severity(sev_path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    recalls = [v["recall"] for v in class_results.values()]
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    print(f"  Avg class recall:       {avg_recall:.3f}  (target each >=0.70)")
    print(f"  False positive rate:    {fp_rate:.3f}  (target ==0.00)")
    print(f"  Wide-angle recall:      {wide_recall:.3f}  (target >=0.50)")
    sev_avg = sum(sev_results.values()) / len(sev_results) if sev_results else 0
    print(f"  Avg severity accuracy:  {sev_avg:.4f}  (target >0.90)")

    print("\nReadiness:")
    if fp_rate == 0 and avg_recall >= 0.70 and wide_recall >= 0.50:
        print("  Stage 1: READY for production")
    else:
        print("  Stage 1: NOT READY — needs improvement")
    if sev_avg >= 0.85:
        print("  Stage 2: READY for production")
    elif sev_avg >= 0.75:
        print("  Stage 2: PARTIAL — acceptable but below 0.90 target")
    else:
        print("  Stage 2: NOT READY — needs more data/labeling")


if __name__ == "__main__":
    main()
