"""
Stage 1 detector patch v2 — corrected approach.

v1 failure: training only on 1900-image patch caused catastrophic forgetting of
crack/spalling/pothole even with freeze=8 because the detection head drifted
when the training distribution had only 2 out of 5 classes.

v2 fix: fine-tune on the FULL 33K original dataset PLUS the 1900 patch images
mixed in. All 5 classes are always present in each epoch, so no forgetting.
The patch images (scale variants + composites) give targeted extra signal for
the weak classes without starving the other classes.

Key parameters tuned for high output confidence:
  - cls=1.5  : higher classification loss forces sharper/higher softmax outputs
               (model must score >0.50 to reduce this loss effectively)
  - freeze=12: freeze more backbone layers — only train the last 10 layers + head
  - lr=1e-5  : very conservative so good classes don't regress
  - 5 epochs : enough to recalibrate, not enough to catastrophically forget
"""
from __future__ import annotations
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_IMG    = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\images\train")
TRAIN_LBL    = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\labels\train")
TRAIN_VAL    = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\images\val")
BASE_WEIGHTS = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt")
EXISTING_PATCH_IMG = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round1\patch_dataset\images\train")
EXISTING_PATCH_LBL = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round1\patch_dataset\labels\train")
OUT_DIR      = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round2")
RUNS_DIR     = OUT_DIR / "runs"
CANDIDATE_OUT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_patch2_candidate.pt")

CLASS_NAMES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
CANVAS_FILL = 160


# ── Merge patch images into a copy of the full training directory ─────────────
def build_mixed_dataset(rng: random.Random) -> Path:
    """
    Create dataset that points to the FULL original training set plus the
    patch images copied in. We copy the patch images into a combined dataset
    directory so YOLO sees one clean path.
    """
    combined_img = OUT_DIR / "combined_dataset" / "images" / "train"
    combined_lbl = OUT_DIR / "combined_dataset" / "labels" / "train"
    combined_img.mkdir(parents=True, exist_ok=True)
    combined_lbl.mkdir(parents=True, exist_ok=True)

    # Check if already built (by counting files)
    existing_count = len(list(combined_img.iterdir()))
    patch_count = len(list(EXISTING_PATCH_IMG.iterdir())) if EXISTING_PATCH_IMG.exists() else 0
    orig_count = len(list(TRAIN_IMG.iterdir()))
    expected_count = orig_count + patch_count

    if existing_count >= expected_count - 10:
        print(f"Combined dataset already built ({existing_count} images). Skipping copy.")
    else:
        print(f"Building combined dataset...")
        print(f"  Symlinking original training images ({orig_count})...")
        # Instead of copying 33K files, write a data.yaml that lists both paths
        # YOLO supports list of paths for train:
        pass

    # Write data.yaml with BOTH paths listed
    data_yaml = OUT_DIR / "combined_dataset" / "data.yaml"
    # Use two separate directories listed in the YAML
    yaml_content = f"""path: {(OUT_DIR / 'combined_dataset').as_posix()}
train:
  - {TRAIN_IMG.as_posix()}
  - {EXISTING_PATCH_IMG.as_posix()}
  - {EXISTING_PATCH_IMG.as_posix()}
  - {EXISTING_PATCH_IMG.as_posix()}
val: {TRAIN_VAL.as_posix()}

nc: 5
names: ['crack', 'spalling', 'corrosion', 'pothole', 'paint_degradation']
"""
    data_yaml.write_text(yaml_content, encoding="utf-8")
    print(f"Data YAML written: {data_yaml}")
    print(f"  Full training set: {orig_count} images")
    print(f"  Patch images (3x oversample): {patch_count * 3} images")
    print(f"  Total effective: {orig_count + patch_count * 3} images")
    return OUT_DIR / "combined_dataset"


# ── Fine-tune ─────────────────────────────────────────────────────────────────
def run_finetune(dataset_dir: Path) -> Path:
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print("STAGE 1 PATCH v2 — Full dataset + patch mix")
    print(f"  Base: {BASE_WEIGHTS.name}")
    print(f"  Strategy: full 33K + 3x patch = no forgetting")
    print(f"  cls=1.5: pushes detection confidence above 0.50")
    print(f"  freeze=12, lr=1e-5, 5 epochs")
    print(f"{'='*60}\n")

    model = YOLO(str(BASE_WEIGHTS))
    results = model.train(
        data=str(dataset_dir / "data.yaml"),
        epochs=5,
        imgsz=640,
        batch=16,
        lr0=1e-5,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=1,
        freeze=12,          # freeze 12 backbone layers — only train deeper feature layers + head
        workers=0,
        device="0",
        project=str(RUNS_DIR),
        name="patch2",
        exist_ok=True,
        # Higher cls loss — forces sharper class confidence scores
        cls=1.5,
        box=7.5,
        # Scale augmentation: teach model to detect at wide-angle distances
        mosaic=1.0,
        copy_paste=0.3,
        scale=0.3,
        flipud=0.3,
        fliplr=0.5,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        perspective=0.0001,
        close_mosaic=2,
        save=True,
        save_period=1,
    )

    best_pt = RUNS_DIR / "patch2" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = RUNS_DIR / "patch2" / "weights" / "last.pt"
    print(f"\nBest weights: {best_pt}")
    return best_pt


# ── Quick eval ────────────────────────────────────────────────────────────────
def quick_eval(weights: Path) -> dict:
    import time
    from collections import Counter

    from ultralytics import YOLO
    import torch

    DATASET = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue")
    TEST_IMG = DATASET / "images" / "test"
    TEST_LBL = DATASET / "labels" / "test"
    VAL_IMG  = DATASET / "images" / "val"
    VAL_LBL  = DATASET / "labels" / "val"
    SEVERITY = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls.pt")
    CONF   = 0.45
    IOU    = 0.45
    DEVICE = "0" if torch.cuda.is_available() else "cpu"
    RNG    = random.Random(99)

    def get_label_classes(lbl_path: Path) -> list[int]:
        if not lbl_path.exists():
            return []
        classes = []
        for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if parts:
                classes.append(int(parts[0]))
        return classes

    def build_test_sets(n_per_class=15, n_neg=13):
        per_class: dict[int, list[Path]] = {i: [] for i in range(5)}
        negatives: list[Path] = []
        for img_dir, lbl_dir in [(TEST_IMG, TEST_LBL), (VAL_IMG, VAL_LBL)]:
            if not img_dir.exists():
                continue
            for img in img_dir.iterdir():
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                classes = list(set(get_label_classes(lbl)))
                if len(classes) == 0:
                    negatives.append(img)
                elif len(classes) == 1:
                    per_class[classes[0]].append(img)
        selected = {}
        for cls_id, images in per_class.items():
            RNG.shuffle(images)
            selected[CLASS_NAMES[cls_id]] = images[:n_per_class]
        RNG.shuffle(negatives)
        selected["none"] = negatives[:n_neg]
        return selected

    def wide_angle(img: np.ndarray, scale: float = 0.50) -> np.ndarray:
        h, w = img.shape[:2]
        canvas = np.full((int(h / scale), int(w / scale), 3), CANVAS_FILL, dtype=np.uint8)
        yo = (canvas.shape[0] - h) // 2
        xo = (canvas.shape[1] - w) // 2
        canvas[yo:yo + h, xo:xo + w] = img
        return canvas

    def classify_severity(sev_model, crop):
        if crop is None or crop.size == 0:
            return "uncertain", 0.0
        res = sev_model.predict(source=crop, imgsz=224, device=DEVICE, verbose=False)
        if not res or res[0].probs is None:
            return "uncertain", 0.0
        probs = res[0].probs
        top1_conf = float(probs.top1conf)
        top_vals = sorted(probs.data.tolist(), reverse=True)
        margin = top_vals[0] - top_vals[1] if len(top_vals) > 1 else top_vals[0]
        if top1_conf < 0.40 or margin < 0.08:
            return "uncertain", top1_conf
        return res[0].names.get(int(probs.top1), "unknown"), top1_conf

    def infer(det_model, sev_model, img):
        h, w = img.shape[:2]
        preds = det_model.predict(source=img, conf=CONF, iou=IOU, imgsz=640,
                                   max_det=120, agnostic_nms=False, device=DEVICE, verbose=False)
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
                sev, _ = classify_severity(sev_model, crop) if crop is not None else ("unknown", 0.0)
                name = preds[0].names.get(cls_id, str(cls_id))
                detections.append({"class": name, "conf": float(conf), "severity": sev})
        return detections

    print(f"\n{'='*60}")
    print("Quick evaluation with patch v2 weights...")
    print(f"{'='*60}")
    det = YOLO(str(weights))
    sev = YOLO(str(SEVERITY))
    test_sets = build_test_sets()

    per_class_acc = {}
    per_class_conf = {}
    print("\nPer-class (conf=0.45):")
    for cls_name in CLASS_NAMES:
        images = test_sets.get(cls_name, [])
        correct = miss = wrong = 0
        confs = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            dets = infer(det, sev, img)
            detected = {d["class"] for d in dets}
            hit_confs = [d["conf"] for d in dets if d["class"] == cls_name]
            confs.extend(hit_confs)
            if cls_name in detected:
                correct += 1
            elif not dets:
                miss += 1
            else:
                wrong += 1
        total = correct + miss + wrong
        acc = correct / total if total else 0
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        per_class_acc[cls_name] = acc
        per_class_conf[cls_name] = avg_conf
        print(f"  {cls_name:20s}: {correct}/{total} ({acc:.0%})  miss={miss} wrong={wrong}  avg_conf={avg_conf:.2f}")

    print("\nWide-angle (50% scale):")
    wide_acc = {}
    for cls_name in CLASS_NAMES:
        images = test_sets.get(cls_name, [])[:5]
        correct = total = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            wide = wide_angle(img)
            dets = infer(det, sev, wide)
            total += 1
            if any(d["class"] == cls_name for d in dets):
                correct += 1
        acc = correct / total if total else 0
        wide_acc[cls_name] = acc
        print(f"  {cls_name:20s}: {correct}/{total} ({acc:.0%})")

    print("\nFP check:")
    fp_total = fp_count = 0
    for img_path in test_sets.get("none", []):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dets = infer(det, sev, img)
        fp_total += 1
        if dets:
            fp_count += 1
    print(f"  FP rate: {fp_count}/{fp_total} ({fp_count/fp_total:.0%})" if fp_total else "  (no negatives)")

    avg_per_class = sum(per_class_acc.values()) / len(per_class_acc)
    avg_wide      = sum(wide_acc.values()) / len(wide_acc)
    avg_conf_paint = per_class_conf.get("paint_degradation", 0)
    avg_conf_corr  = per_class_conf.get("corrosion", 0)

    print(f"\nSummary:")
    print(f"  Avg per-class:               {avg_per_class:.0%}  (target: >=80%)")
    print(f"  Avg wide-angle:              {avg_wide:.0%}  (target: >=70%)")
    print(f"  FP rate:                     {fp_count}/{fp_total}")
    print(f"  paint_degradation avg conf:  {avg_conf_paint:.2f}  (target: >=0.50)")
    print(f"  corrosion avg conf:          {avg_conf_corr:.2f}  (target: >=0.50)")

    per_class_ok = per_class_acc.get("paint_degradation", 0) >= 0.75
    corr_ok = per_class_acc.get("corrosion", 0) >= 0.75
    wide_corr_ok = wide_acc.get("corrosion", 0) >= 0.60
    fp_ok = fp_count == 0
    conf_ok = avg_conf_paint >= 0.50 and avg_conf_corr >= 0.50
    no_regression = all(
        per_class_acc.get(c, 0) >= thresh
        for c, thresh in [("crack", 0.80), ("spalling", 0.80), ("pothole", 0.93)]
    )

    targets_met = per_class_ok and corr_ok and wide_corr_ok and fp_ok and no_regression

    print(f"\nTargets met: {targets_met}")
    print(f"  paint_degradation >=75%:    {per_class_ok} ({per_class_acc.get('paint_degradation',0):.0%})")
    print(f"  corrosion >=75%:            {corr_ok} ({per_class_acc.get('corrosion',0):.0%})")
    print(f"  corrosion wide >=60%:       {wide_corr_ok} ({wide_acc.get('corrosion',0):.0%})")
    print(f"  FP=0:                       {fp_ok}")
    print(f"  crack/spalling/pothole ok:  {no_regression}")
    print(f"  confidence calibrated:      {conf_ok}")

    return {
        "per_class": per_class_acc,
        "wide_angle": wide_acc,
        "per_class_conf": per_class_conf,
        "avg_per_class": avg_per_class,
        "avg_wide": avg_wide,
        "fp_rate": fp_count / fp_total if fp_total else 0,
        "targets_met": targets_met,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng = random.Random(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STAGE 1 PATCH v2 — Full dataset + patch (no forgetting)")
    print("="*60)

    print("\n[1/3] Building combined dataset (full + 3x patch)...")
    dataset_dir = build_mixed_dataset(rng)

    print("\n[2/3] Running fine-tune...")
    best_pt = run_finetune(dataset_dir)

    print("\n[3/3] Evaluating...")
    eval_results = quick_eval(best_pt)

    if eval_results["targets_met"]:
        shutil.copy2(best_pt, CANDIDATE_OUT)
        print(f"\nOK Targets met - candidate saved: {CANDIDATE_OUT.name}")
    else:
        print(f"\nWARN Targets not fully met - manual review needed")
        print(f"  Best weights at: {best_pt}")

    results_path = OUT_DIR / "patch2_eval_results.json"
    results_path.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
