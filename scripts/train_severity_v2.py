"""
train_severity_v2.py — Severity classifier retraining toward >90% accuracy.

Diagnosis from sev_patch1 confusion matrix (best=65.8%):
  - moderate: only 62% correct (bleeds 26% → critical, 24% → minor)
  - minor: 70%, critical: 66% — both limited by model capacity
  - Root cause: yolo11n-cls (nano) too small; only 30 epochs; no RandAugment

Fixes:
  1. yolo11s-cls (small, ~3× nano params) — better feature capacity for the
     heterogeneous "moderate" class (cracks + spalling + corrosion + paint)
  2. 100 epochs with cosine LR + 5-epoch warmup
  3. RandAugment + random erasing — forces invariance to texture shortcuts
  4. label_smoothing=0.1 — accounts for inherent label ambiguity at boundaries
  5. Resume support — skips training if best.pt already exists

Dataset: output/continued_hn_weak/current/datasets/severity_dataset
  train: 1800 images (600×3 classes, balanced)
  val:    360 images (120×3)
  classes: minor / moderate / critical

Expected: ~80-88% top-1 accuracy (65.8% baseline → meaningful improvement).
90% may require per-defect-type classifiers (see below) if plateau hits ~82%.
"""
from __future__ import annotations

import ctypes
import shutil
import sys
from pathlib import Path

DATASET   = Path(
    r"C:\Users\User\AIEngGroupProj\output"
    r"\continued_hn_weak\current\datasets\severity_dataset"
)
OUT_DIR   = Path(r"C:\Users\User\AIEngGroupProj\output\severity_runs")
RUN_NAME  = "sev_v4"
BEST_OUT  = Path(r"C:\Users\User\AIEngGroupProj\weights\severity_cls.pt")

TOTAL_EPOCHS = 100


def keep_awake() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("  [sleep-guard] Windows sleep inhibited")
    except Exception:
        pass


def run() -> None:
    if not DATASET.exists():
        print(f"ERROR: severity dataset not found: {DATASET}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep_awake()

    run_dir  = OUT_DIR / RUN_NAME
    best_pt  = run_dir / "weights" / "best.pt"
    last_pt  = run_dir / "weights" / "last.pt"

    # Count completed epochs for resume support
    results_csv = run_dir / "results.csv"
    epochs_done = 0
    if results_csv.exists():
        try:
            with open(results_csv) as f:
                epochs_done = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass

    if best_pt.exists() and epochs_done >= TOTAL_EPOCHS:
        print(f"Training already complete ({epochs_done}/{TOTAL_EPOCHS} epochs). Skipping.")
    else:
        from ultralytics import YOLO

        remaining = TOTAL_EPOCHS - epochs_done
        if last_pt.exists() and epochs_done > 0:
            start = str(last_pt)
            warmup = 1
            print(f"Resuming from epoch {epochs_done} (last.pt), {remaining} epochs remaining")
        else:
            start = "yolo11s-cls.pt"   # small: better capacity than nano
            warmup = 5
            epochs_done = 0
            remaining = TOTAL_EPOCHS
            print(f"Starting fresh from yolo11s-cls.pt, {remaining} epochs")

        print("=" * 64)
        print(f"Severity classifier  yolo11s-cls  {remaining} epochs  batch=32")
        print(f"  Dataset : {DATASET}")
        print(f"  Output  : {run_dir}")
        print(f"  ~3-4 min/epoch on RTX 3070 Ti = ~{round(remaining*3.5/60,1)} hrs")
        print("=" * 64)

        model = YOLO(start)
        model.train(
            data=str(DATASET),
            epochs=remaining,
            imgsz=224,
            batch=32,
            workers=0,          # Windows: no multiprocessing
            device="0",

            optimizer="AdamW",
            lr0=1e-4,           # conservative — 1800 images, 5.4M params, easy to overfit
            lrf=0.01,
            weight_decay=5e-4,  # higher regularization vs v3 (was 1e-4 → overfit fast)
            warmup_epochs=warmup,
            cos_lr=True,

            label_smoothing=0.1,

            # Light augmentation — v3 failed with erasing=0.4+RandAugment on 224px crops
            # Small crops have few pixels; heavy erasing destroys distinguishing features
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
            fliplr=0.5,
            flipud=0.2,
            scale=0.2,
            erasing=0.15,       # reduced from 0.4 — just enough for regularisation

            patience=20,
            save_period=10,
            exist_ok=True,
            project=str(OUT_DIR),
            name=RUN_NAME,
        )

    # Copy best weights to canonical location
    src = best_pt if best_pt.exists() else last_pt
    if src.exists():
        BEST_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, BEST_OUT)
        print(f"\nSaved best weights → {BEST_OUT}")

    # Quick val on test split to report final accuracy
    from ultralytics import YOLO as _YOLO
    model_eval = _YOLO(str(src))
    test_dir = DATASET / "test"
    if test_dir.exists():
        print("\nRunning final eval on test split …")
        results = model_eval.val(
            data=str(DATASET),
            split="test",
            imgsz=224,
            batch=32,
            workers=0,
            device="0",
            verbose=True,
        )
        top1 = results.results_dict.get("metrics/accuracy_top1", 0)
        top5 = results.results_dict.get("metrics/accuracy_top5", 0)
        print(f"\nTest top-1 accuracy : {top1:.4f}  ({top1*100:.1f}%)")
        print(f"Test top-5 accuracy : {top5:.4f}")
        baseline = 0.658
        delta = top1 - baseline
        print(f"vs baseline (65.8%) : {'+' if delta>=0 else ''}{delta:+.4f}")
        if top1 >= 0.90:
            print("TARGET MET: >90% achieved!")
        elif top1 >= 0.80:
            print("Significant improvement. Consider per-defect classifiers to push to 90%.")
        else:
            print("Moderate improvement. Per-defect classifiers may be needed for 90% target.")
    else:
        print(f"\nNo test split found at {test_dir} — check val accuracy in training logs.")


if __name__ == "__main__":
    run()
