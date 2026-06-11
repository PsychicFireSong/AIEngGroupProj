"""
Per-defect severity classifier training.

Root cause of 65.8% ceiling:
  - One model for 5 defect types: learns ambiguous shared severity concept
  - Minor crack looks nothing like minor spalling — model must learn both
  - Per-defect models learn one severity concept per defect type

Architecture:
  - 5 separate YOLO11n-cls models (one per defect class)
  - Each model trained on 360 images (120/class) + 72 val (24/class)
  - Initialize from best existing severity_cls.pt (65.8% model)
  - Long training with patience to avoid overfitting on small dataset

Inference routing (API update needed after this):
  - Stage 1 predicts "corrosion" -> use severity_corrosion model
  - Stage 1 predicts "crack"     -> use severity_crack model
  - etc.

Expected improvement: 70-80% per defect vs 65.8% combined
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

SEVERITY_DS = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\severity_dataset")
# Prefer v3 output (200ep cosine), else fall back to v1 best (65.8%)
_V3_WEIGHTS = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls_patch3_candidate.pt")
_V1_WEIGHTS = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls_patch1_candidate.pt")
BASE_WEIGHTS = _V3_WEIGHTS if _V3_WEIGHTS.exists() else _V1_WEIGHTS
OUT_DIR = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round5\severity_perdefect")
WEIGHTS_OUT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights")

DEFECT_CLASSES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
SEVERITY_CLASSES = ["critical", "minor", "moderate"]


def build_perdefect_datasets() -> dict[str, Path]:
    """Reorganise mixed severity dataset into per-defect subset folders."""
    ds_root = OUT_DIR / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)

    defect_ds_paths = {}
    for defect in DEFECT_CLASSES:
        defect_ds = ds_root / f"sev_{defect}"
        defect_ds.mkdir(exist_ok=True)

        for split in ["train", "val"]:
            for sev_cls in SEVERITY_CLASSES:
                src_dir = SEVERITY_DS / split / sev_cls
                dst_dir = defect_ds / split / sev_cls
                dst_dir.mkdir(parents=True, exist_ok=True)

                for img in src_dir.glob("*.jpg"):
                    # filename: weak_{defect_type}_{id}_{hash}.jpg
                    parts = img.stem.split("_")
                    img_defect = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
                    if img_defect == defect:
                        # symlink to avoid copying
                        dst = dst_dir / img.name
                        if not dst.exists():
                            try:
                                dst.symlink_to(img)
                            except OSError:
                                shutil.copy2(img, dst)

        defect_ds_paths[defect] = defect_ds

        # Count images in this subset
        for split in ["train", "val"]:
            counts = []
            for sev_cls in SEVERITY_CLASSES:
                n = len(list((defect_ds / split / sev_cls).glob("*.jpg")))
                counts.append(f"{sev_cls}:{n}")
            print(f"  {defect}/{split}: {', '.join(counts)}")

    return defect_ds_paths


def train_perdefect_model(defect: str, ds_path: Path) -> float:
    from ultralytics import YOLO
    import csv as _csv

    run_name = f"sev_{defect}"
    print(f"\n--- Training severity_{defect} ---")
    print(f"  Dataset: {ds_path}")

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(ds_path),
        epochs=150,
        imgsz=224,
        batch=16,          # smaller batch for smaller dataset
        lr0=3e-5,          # very low LR — careful refinement
        lrf=0.01,
        cos_lr=True,
        momentum=0.9,
        weight_decay=5e-3,
        warmup_epochs=3,
        patience=30,       # patient early stopping
        dropout=0.3,       # regularize more on small dataset
        label_smoothing=0.1,
        workers=0,
        device="0",
        project=str(OUT_DIR / "runs"),
        name=run_name,
        exist_ok=True,
        # Medium augmentation
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.3,
        fliplr=0.5,
        degrees=10.0,
        translate=0.05,
        scale=0.2,
        erasing=0.2,
        mixup=0.05,
        copy_paste=0.0,
        close_mosaic=0,
        auto_augment="",
        save=True,
        save_period=20,
    )

    # Find best val accuracy for this defect
    csv_p = OUT_DIR / "runs" / run_name / "results.csv"
    best_acc = 0.0
    if csv_p.exists():
        rows = list(_csv.DictReader(open(csv_p)))
        if rows:
            best = max(rows, key=lambda r: float(r.get("metrics/accuracy_top1", 0)))
            best_acc = float(best.get("metrics/accuracy_top1", 0))
            print(f"  Best val top-1: {best_acc:.4f} (epoch {best['epoch']})")

    best_pt = OUT_DIR / "runs" / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = OUT_DIR / "runs" / run_name / "weights" / "last.pt"

    # Save to named slot for API access
    out_name = f"severity_{defect}_cls.pt"
    out_path = WEIGHTS_OUT / out_name
    if best_pt.exists():
        shutil.copy2(best_pt, out_path)
        print(f"  Saved: {out_name}")

    return best_acc


def main():
    print("=" * 60)
    print("STAGE 2 SEVERITY — Per-Defect Models (5 models)")
    print("=" * 60)
    print(f"  Base weights: {BASE_WEIGHTS.name}")
    print(f"  Output: {WEIGHTS_OUT}")
    print("\nBuilding per-defect datasets...")
    defect_ds_paths = build_perdefect_datasets()

    results = {}
    for defect in DEFECT_CLASSES:
        acc = train_perdefect_model(defect, defect_ds_paths[defect])
        results[defect] = acc

    print("\n" + "=" * 60)
    print("PER-DEFECT SEVERITY RESULTS")
    print("=" * 60)
    for defect, acc in results.items():
        print(f"  {defect:20s}: {acc:.4f}  (target >0.90)")
    overall = sum(results.values()) / len(results) if results else 0
    print(f"\n  Average: {overall:.4f}")
    print(f"  Baseline (mixed model): 0.6580")
    print(f"  Improvement: +{(overall - 0.658):.4f}")

    # Save summary
    summary = OUT_DIR / "perdefect_results.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["defect", "val_top1"])
        for defect, acc in results.items():
            w.writerow([defect, f"{acc:.4f}"])
    print(f"\nSummary saved: {summary}")


if __name__ == "__main__":
    main()
