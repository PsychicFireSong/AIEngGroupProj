"""
Severity label quality analysis.

Uses the existing best severity model (65.8%) to find:
1. Consistently mis-predicted samples (potential label noise)
2. High-confidence wrong predictions (hard noise, model is sure but label disagrees)
3. Per-class accuracy breakdown
4. Confusion matrix

This helps diagnose whether the 65.8% ceiling is caused by:
  - Label noise (auto-generated labels have errors)
  - Visual ambiguity (real minor/moderate boundary is fuzzy)
  - Model capacity limits

Run: python analyze_severity_labels.py
Output: CSV of flagged images + confusion matrix
"""
from __future__ import annotations
import csv, shutil
from pathlib import Path

import numpy as np
from PIL import Image

SEV_MODEL    = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls_patch1_candidate.pt")
SEVERITY_DS  = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\severity_dataset")
OUT_DIR      = Path(r"C:\Users\User\AIEngGroupProj\output\analysis\severity_label_quality")

CLASSES      = ["critical", "minor", "moderate"]
IMG_SIZE     = 224
FLIP_CONF_THRESHOLD = 0.80  # flag if model has >80% conf on wrong label


def run_tta_inference(model, img_path: Path, n_aug: int = 5):
    """Run model with TTA (augmentations) and return averaged class probabilities."""
    import torch
    from torchvision import transforms

    base_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    aug_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    tensors = [base_tf(img).unsqueeze(0)]
    if n_aug > 1:
        for _ in range(n_aug - 1):
            tensors.append(aug_tf(img).unsqueeze(0))

    with torch.no_grad():
        all_probs = []
        for t in tensors:
            results = model.predict(source=t.numpy()[0].transpose(1, 2, 0),
                                    imgsz=IMG_SIZE, verbose=False)
            if results:
                probs = results[0].probs.data.cpu().numpy()
                all_probs.append(probs)

    if not all_probs:
        return None
    return np.mean(all_probs, axis=0)


def main():
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading severity model:", SEV_MODEL)
    model = YOLO(str(SEV_MODEL))

    # Get class ordering from model
    sample = list((SEVERITY_DS / "val" / "critical").glob("*.jpg"))[0]
    res = model.predict(source=str(sample), imgsz=IMG_SIZE, verbose=False)
    model_classes = res[0].names  # {0: 'critical', 1: 'minor', 2: 'moderate'}
    print("Model class order:", model_classes)

    flagged = []
    confusion = np.zeros((3, 3), dtype=int)  # [true][pred]

    for split in ["train", "val"]:
        print(f"\n--- {split} set ---")
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = SEVERITY_DS / split / cls_name
            if not cls_dir.exists():
                continue
            imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
            wrong = 0
            for img_path in imgs:
                results = model.predict(source=str(img_path), imgsz=IMG_SIZE, verbose=False)
                if not results:
                    continue
                probs = results[0].probs
                pred_idx = int(probs.top1)
                pred_name = model_classes.get(pred_idx, str(pred_idx))
                pred_conf = float(probs.top1conf)

                # Build confusion matrix
                true_model_idx = next(
                    (k for k, v in model_classes.items() if v == cls_name), None
                )
                if true_model_idx is not None:
                    confusion[true_model_idx][pred_idx] += 1

                if pred_name != cls_name:
                    wrong += 1
                    if pred_conf >= FLIP_CONF_THRESHOLD:
                        flagged.append({
                            "split": split,
                            "true_label": cls_name,
                            "pred_label": pred_name,
                            "pred_conf": round(pred_conf, 4),
                            "path": str(img_path),
                        })

            acc = (len(imgs) - wrong) / len(imgs) if imgs else 0
            print(f"  {cls_name:12s}: {acc:.3f}  ({len(imgs)-wrong}/{len(imgs)} correct)")

    # Print confusion matrix
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(f"{'':15s}", "  ".join(f"{c:12s}" for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        row = confusion[i]
        print(f"  {cls:13s}", "  ".join(f"{v:12d}" for v in row))

    # Save flagged samples
    flagged_csv = OUT_DIR / "flagged_samples.csv"
    if flagged:
        with open(flagged_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split","true_label","pred_label","pred_conf","path"])
            writer.writeheader()
            writer.writerows(sorted(flagged, key=lambda r: -r["pred_conf"]))
        print(f"\nFlagged {len(flagged)} potentially mislabeled images (conf >= {FLIP_CONF_THRESHOLD})")
        print(f"Saved to: {flagged_csv}")

        # Copy top 20 flagged images for visual review
        review_dir = OUT_DIR / "review_images"
        review_dir.mkdir(exist_ok=True)
        for i, row in enumerate(sorted(flagged, key=lambda r: -r["pred_conf"])[:20]):
            src = Path(row["path"])
            dst = review_dir / f"{i:02d}_true_{row['true_label']}_pred_{row['pred_label']}_conf{row['pred_conf']:.2f}{src.suffix}"
            shutil.copy2(src, dst)
        print(f"Top 20 flagged images copied to: {review_dir}")
    else:
        print("\nNo high-confidence wrong predictions found.")

    total = confusion.sum()
    correct = np.trace(confusion)
    print(f"\nOverall accuracy: {correct}/{total} = {correct/total:.4f}")
    print("\nDiagnosis:")
    if len(flagged) > 50:
        print("  HIGH label noise detected (>50 high-confidence flips)")
        print("  -> 90% accuracy is NOT achievable without relabeling")
    elif len(flagged) > 20:
        print("  MODERATE label noise (20-50 high-confidence flips)")
        print("  -> Relabeling flagged images could add +3-5% accuracy")
    else:
        print("  LOW label noise (<20 high-confidence flips)")
        print("  -> Ceiling is visual ambiguity + small dataset, not noise")
        print("  -> Need more data OR per-defect-type severity models to reach 90%")


if __name__ == "__main__":
    main()
