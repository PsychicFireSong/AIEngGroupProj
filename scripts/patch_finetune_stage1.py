"""
Targeted patch fine-tuning for Stage 1 detector deficiencies:

Root causes:
  - paint_degradation: 6/15 misses at conf=0.45 (model confidence below threshold)
  - corrosion: 4/15 wrong-class (confused with paint_degradation), 40% wide-angle
  - Both: poor wide-angle due to scale not seen in training

Strategy:
  1. Build 2700-image patch dataset:
     - 300 paint_degradation + 300 corrosion source images
     - Each gets 3 scale-shrink variants (0.3, 0.4, 0.5 into grey canvas)
     - 100 synthetic left-right composites (two classes per image)
  2. Fine-tune 10 epochs from defect_detector_hn_weak_candidate.pt
     - LR=5e-5 (conservative), freeze=8 (only train deeper layers)
     - Heavy scale/mosaic/copy_paste augmentation
     - Workers=0 (Windows)
"""
from __future__ import annotations
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_IMG = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\images\train")
TRAIN_LBL = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\labels\train")
BASE_WEIGHTS = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt")
OUT_DIR = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round1")
PATCH_DS = OUT_DIR / "patch_dataset"
RUNS_DIR = OUT_DIR / "runs"
CANDIDATE_OUT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_patch1_candidate.pt")

CLASS_NAMES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}
CORROSION_ID = 2
PAINT_ID = 4

SCALE_VARIANTS = [0.30, 0.40, 0.50]
N_SOURCE_PER_CLASS = 300
N_COMPOSITES = 100
CANVAS_FILL = 160  # grey canvas matching building wall


# ── Label helpers ─────────────────────────────────────────────────────────────
def read_labels(lbl_path: Path) -> list[list[float]]:
    if not lbl_path.exists():
        return []
    boxes = []
    for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append([float(p) for p in parts[:5]])
    return boxes


def write_labels(lbl_path: Path, boxes: list[list[float]]) -> None:
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
    lbl_path.write_text("\n".join(lines), encoding="utf-8")


def transform_boxes_scale(boxes: list[list[float]], scale: float) -> list[list[float]]:
    """Adjust YOLO boxes after shrinking image to `scale` fraction of canvas.
    Formula: x_c_new = x_c * scale + (1-scale)/2  (centering offset)
             w_new   = w * scale
    Same for y.
    """
    offset = (1.0 - scale) / 2.0
    result = []
    for b in boxes:
        cls_id, xc, yc, w, h = b
        new_xc = xc * scale + offset
        new_yc = yc * scale + offset
        new_w  = w * scale
        new_h  = h * scale
        # clamp to [0,1]
        new_xc = max(0.0, min(1.0, new_xc))
        new_yc = max(0.0, min(1.0, new_yc))
        new_w  = min(new_w, 2 * min(new_xc, 1 - new_xc))
        new_h  = min(new_h, 2 * min(new_yc, 1 - new_yc))
        if new_w > 0.005 and new_h > 0.005:
            result.append([cls_id, new_xc, new_yc, new_w, new_h])
    return result


def shrink_to_canvas(img: np.ndarray, scale: float) -> np.ndarray:
    """Shrink image to `scale` fraction and center it on a larger grey canvas."""
    h, w = img.shape[:2]
    canvas_h, canvas_w = int(h / scale), int(w / scale)
    canvas = np.full((canvas_h, canvas_w, 3), CANVAS_FILL, dtype=np.uint8)
    y_off = (canvas_h - h) // 2
    x_off = (canvas_w - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = img
    return canvas


# ── Per-class image finder ────────────────────────────────────────────────────
def find_class_images(class_id: int, n: int, rng: random.Random) -> list[Path]:
    candidates: list[Path] = []
    for img in TRAIN_IMG.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = TRAIN_LBL / (img.stem + ".txt")
        boxes = read_labels(lbl)
        classes = {int(b[0]) for b in boxes}
        if class_id in classes and len(classes) == 1:  # single-class only (cleaner)
            candidates.append(img)
    rng.shuffle(candidates)
    print(f"  class {CLASS_NAMES[class_id]}: found {len(candidates)} single-class images, sampling {min(n, len(candidates))}")
    return candidates[:n]


# ── Dataset builder ──────────────────────────────────────────────────────────
def build_scale_variants(
    source_imgs: list[Path],
    class_id: int,
    out_img: Path,
    out_lbl: Path,
    tag: str,
) -> int:
    count = 0
    for img_path in source_imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl_path = TRAIN_LBL / (img_path.stem + ".txt")
        boxes = read_labels(lbl_path)
        if not boxes:
            continue
        for scale in SCALE_VARIANTS:
            canvas = shrink_to_canvas(img, scale)
            new_boxes = transform_boxes_scale(boxes, scale)
            if not new_boxes:
                continue
            stem = f"{tag}_{img_path.stem}_s{int(scale*100)}"
            cv2.imwrite(str(out_img / f"{stem}.jpg"), canvas)
            write_labels(out_lbl / f"{stem}.txt", new_boxes)
            count += 1
    return count


def build_composites(
    paint_imgs: list[Path],
    corr_imgs: list[Path],
    crack_imgs: list[Path],
    spall_imgs: list[Path],
    out_img: Path,
    out_lbl: Path,
    n: int,
    rng: random.Random,
) -> int:
    """Create side-by-side composites of two different defect classes."""
    pairs = [
        (paint_imgs, PAINT_ID, corr_imgs, CORROSION_ID),
        (paint_imgs, PAINT_ID, crack_imgs, CLASS_IDS["crack"]),
        (corr_imgs, CORROSION_ID, spall_imgs, CLASS_IDS["spalling"]),
        (crack_imgs, CLASS_IDS["crack"], spall_imgs, CLASS_IDS["spalling"]),
    ]
    count = 0
    per_pair = n // len(pairs)
    for left_imgs, left_cls, right_imgs, right_cls in pairs:
        for i in range(min(per_pair, len(left_imgs), len(right_imgs))):
            li = rng.choice(left_imgs)
            ri = rng.choice(right_imgs)
            limg = cv2.imread(str(li))
            rimg = cv2.imread(str(ri))
            if limg is None or rimg is None:
                continue
            target_h = 640
            limg = cv2.resize(limg, (640, target_h))
            rimg = cv2.resize(rimg, (640, target_h))
            composite = np.hstack([limg, rimg])  # 1280 × 640
            composite = cv2.resize(composite, (640, 640))  # back to 640×640

            # Adjust labels: left half = [0, 0.5] in x, right half = [0.5, 1.0]
            left_boxes = read_labels(TRAIN_LBL / (li.stem + ".txt"))
            right_boxes = read_labels(TRAIN_LBL / (ri.stem + ".txt"))

            # Override class IDs to ensure correct labeling
            new_boxes = []
            for b in left_boxes:
                xc_new = b[1] * 0.5         # left half
                w_new  = b[3] * 0.5
                if w_new > 0.01:
                    new_boxes.append([float(left_cls), xc_new, b[2], w_new, b[4]])
            for b in right_boxes:
                xc_new = b[1] * 0.5 + 0.5  # right half
                w_new  = b[3] * 0.5
                if w_new > 0.01:
                    new_boxes.append([float(right_cls), xc_new, b[2], w_new, b[4]])

            if not new_boxes:
                continue
            stem = f"composite_{left_cls}_{right_cls}_{count:04d}"
            cv2.imwrite(str(out_img / f"{stem}.jpg"), composite)
            write_labels(out_lbl / f"{stem}.txt", new_boxes)
            count += 1
    return count


def build_patch_dataset(rng: random.Random) -> Path:
    out_img = PATCH_DS / "images" / "train"
    out_lbl = PATCH_DS / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    print("Finding source images...")
    paint_imgs = find_class_images(PAINT_ID, N_SOURCE_PER_CLASS, rng)
    corr_imgs  = find_class_images(CORROSION_ID, N_SOURCE_PER_CLASS, rng)
    crack_imgs = find_class_images(CLASS_IDS["crack"], 100, rng)
    spall_imgs = find_class_images(CLASS_IDS["spalling"], 100, rng)

    print("\nBuilding scale-augmented paint_degradation variants...")
    n_paint = build_scale_variants(paint_imgs, PAINT_ID, out_img, out_lbl, "patch_paint")
    print(f"  -> {n_paint} images")

    print("Building scale-augmented corrosion variants...")
    n_corr = build_scale_variants(corr_imgs, CORROSION_ID, out_img, out_lbl, "patch_corr")
    print(f"  -> {n_corr} images")

    print("Building synthetic multi-defect composites...")
    n_comp = build_composites(paint_imgs, corr_imgs, crack_imgs, spall_imgs, out_img, out_lbl, N_COMPOSITES, rng)
    print(f"  -> {n_comp} composite images")

    total = n_paint + n_corr + n_comp
    print(f"\nPatch dataset total: {total} training images")

    # Write data.yaml (single split — YOLO will use train only; no val needed for micro-finetune)
    # We use the original val set for validation
    data_yaml = PATCH_DS / "data.yaml"
    yaml_content = f"""path: {PATCH_DS.as_posix()}
train: images/train
val: {(TRAIN_IMG.parent / 'val').as_posix()}

nc: 5
names: ['crack', 'spalling', 'corrosion', 'pothole', 'paint_degradation']
"""
    data_yaml.write_text(yaml_content, encoding="utf-8")
    print(f"Data YAML: {data_yaml}")
    return PATCH_DS


# ── Fine-tune ─────────────────────────────────────────────────────────────────
def run_finetune(patch_ds: Path) -> Path:
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print("Starting patch fine-tune (Stage 1 detector)")
    print(f"  Base weights: {BASE_WEIGHTS.name}")
    print(f"  Patch dataset: {patch_ds}")
    print(f"  Epochs: 10, LR: 5e-5, Freeze: 8 layers, cls_loss=1.5")
    print(f"{'='*60}\n")

    model = YOLO(str(BASE_WEIGHTS))
    results = model.train(
        data=str(patch_ds / "data.yaml"),
        epochs=10,
        imgsz=640,
        batch=8,
        lr0=5e-5,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=1,
        freeze=8,                  # freeze backbone first 8 layers
        workers=0,                 # Windows: no subprocess workers
        device="0",
        project=str(RUNS_DIR),
        name="patch1",
        exist_ok=True,
        # Higher cls loss weight: forces model to output sharper/higher class confidence
        # Default 0.5 is too gentle — model can score 0.38 and still converge fine.
        # 1.5 pushes softmax outputs to be decisive (>0.50 on genuine detections).
        cls=1.5,
        box=7.5,
        # Augmentation — heavy scale to fix wide-angle gap
        mosaic=1.0,
        copy_paste=0.3,
        scale=0.3,                 # scale 0.3-1.7 range
        flipud=0.3,
        fliplr=0.5,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        perspective=0.0001,
        close_mosaic=3,
        # Save
        save=True,
        save_period=2,
    )

    best_pt = RUNS_DIR / "patch1" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = RUNS_DIR / "patch1" / "weights" / "last.pt"
    print(f"\nBest weights: {best_pt}")
    return best_pt


# ── Quick eval ────────────────────────────────────────────────────────────────
def quick_eval(weights: Path) -> dict:
    """Re-run realworld_eval2 logic with new weights, print summary."""
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
    CONF = 0.45
    IOU  = 0.45
    DEVICE = "0" if torch.cuda.is_available() else "cpu"
    RNG  = random.Random(99)

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
        selected: dict[str, list[Path]] = {}
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
    print("Quick evaluation with patched weights...")
    print(f"{'='*60}")
    det = YOLO(str(weights))
    sev = YOLO(str(SEVERITY))
    test_sets = build_test_sets()

    per_class_acc = {}
    print("\nPer-class (conf=0.45):")
    for cls_name in CLASS_NAMES:
        images = test_sets.get(cls_name, [])
        correct = miss = wrong = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            dets, = [infer(det, sev, img)]
            detected = {d["class"] for d in dets}
            if cls_name in detected:
                correct += 1
            elif not dets:
                miss += 1
            else:
                wrong += 1
        total = correct + miss + wrong
        acc = correct / total if total else 0
        per_class_acc[cls_name] = acc
        print(f"  {cls_name:20s}: {correct}/{total} ({acc:.0%})  miss={miss} wrong={wrong}")

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
    avg_wide = sum(wide_acc.values()) / len(wide_acc)
    print(f"\nSummary:")
    print(f"  Avg per-class: {avg_per_class:.0%}  (target: >=80%)")
    print(f"  Avg wide-angle: {avg_wide:.0%}  (target: >=70%)")
    print(f"  FP: {fp_count}/{fp_total}")
    print()

    return {
        "per_class": per_class_acc,
        "wide_angle": wide_acc,
        "avg_per_class": avg_per_class,
        "avg_wide": avg_wide,
        "fp_rate": fp_count / fp_total if fp_total else 0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng = random.Random(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("="*60)
    print("STAGE 1 PATCH FINE-TUNE")
    print("="*60)
    print("Targets:")
    print("  paint_degradation: 60% -> >=75%")
    print("  corrosion wrong-class: 27% -> <=10%")
    print("  corrosion wide-angle: 40% -> >=65%")
    print("  FP: must stay 0%")
    print()

    print("[1/3] Building patch dataset...")
    patch_ds = build_patch_dataset(rng)

    print("\n[2/3] Running patch fine-tune...")
    best_pt = run_finetune(patch_ds)

    print("\n[3/3] Evaluating patched weights...")
    eval_results = quick_eval(best_pt)

    # Save candidate if targets met
    targets_met = (
        eval_results["per_class"].get("paint_degradation", 0) >= 0.75
        and eval_results["per_class"].get("corrosion", 0) >= 0.75
        and eval_results["wide_angle"].get("corrosion", 0) >= 0.60
        and eval_results["fp_rate"] == 0.0
    )

    if targets_met:
        shutil.copy2(best_pt, CANDIDATE_OUT)
        print(f"\nOK Targets met — candidate saved to {CANDIDATE_OUT.name}")
    else:
        print(f"\nWARN Some targets not met — see scores above")
        print(f"  Candidate NOT promoted automatically")
        print(f"  Manual review recommended. Best weights: {best_pt}")

    # Save results
    results_path = OUT_DIR / "patch1_eval_results.json"
    results_path.write_text(
        json.dumps({"targets_met": targets_met, **eval_results}, indent=2),
        encoding="utf-8"
    )
    print(f"\nFull results: {results_path}")


if __name__ == "__main__":
    main()
