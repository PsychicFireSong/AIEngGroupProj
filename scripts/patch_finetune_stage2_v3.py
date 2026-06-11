"""
Stage 2 severity classifier v3 — correct approach after v2 failure.

Diagnosis of v2 (YOLO11s-cls) failure:
  - 9.4M params on 1800 training images = severe overfitting
  - Best at epoch 3 (62.2%), then degraded — never converged
  - Bigger model is WORSE on small datasets

v3 strategy:
  - Back to YOLO11n-cls (1.5M params) — proven 65.8% ceiling with 30ep
  - Longer training: 200 epochs with cosine LR decay
  - Gentler augmentation: no heavy mixup (it destroys signal at 1800 images)
  - Lower LR: 5e-5 with careful warmup
  - Target: push nano ceiling from 65.8% to ~72-75%

Realistic expectation:
  - With 1800 training images, 90% top-1 is not achievable (label noise + visual overlap)
  - Practical ceiling: ~72-75% with optimal training
  - To reach 90%: need 10,000+ training images with clean expert labels
"""
from __future__ import annotations
from pathlib import Path
import shutil, csv

SEVERITY_DS   = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\severity_dataset")
BASE_WEIGHTS  = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls_patch1_candidate.pt")
OUT_DIR       = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round3\severity_runs")
CANDIDATE_OUT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls_patch3_candidate.pt")


def run() -> Path:
    from ultralytics import YOLO

    print("="*60)
    print("STAGE 2 SEVERITY v3 — nano, 200ep, cosine LR")
    print("="*60)
    print(f"  Model: YOLO11n-cls (1.5M params) from existing severity_cls.pt")
    print(f"  Epochs: 200, patience=40, cosine LR")
    print(f"  Goal: push nano ceiling from 65.8% toward 72-75%")
    print()

    model = YOLO(str(BASE_WEIGHTS))
    results = model.train(
        data=str(SEVERITY_DS),
        epochs=200,
        imgsz=224,
        batch=32,
        lr0=5e-5,           # low LR — careful refinement of existing weights
        lrf=0.01,
        cos_lr=True,        # cosine decay for smooth convergence
        momentum=0.9,
        weight_decay=1e-3,
        warmup_epochs=5,
        patience=40,        # stop if no improvement for 40 epochs
        dropout=0.2,        # mild regularization
        label_smoothing=0.05,
        workers=0,
        device="0",
        project=str(OUT_DIR),
        name="sev_v3",
        exist_ok=True,
        # Gentle augmentation — heavier augmentation hurts small datasets
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.3,
        flipud=0.3,
        fliplr=0.5,
        degrees=10.0,
        translate=0.05,
        scale=0.2,
        erasing=0.2,
        mixup=0.05,         # very light mixup (was 0.3 in v2 — too much)
        copy_paste=0.0,
        close_mosaic=0,
        auto_augment="",    # no auto augment — keep it simple
        save=True,
        save_period=20,
    )

    best_pt = OUT_DIR / "sev_v3" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = OUT_DIR / "sev_v3" / "weights" / "last.pt"
    print(f"\nBest weights: {best_pt}")

    # Report best val accuracy
    csv_p = OUT_DIR / "sev_v3" / "results.csv"
    if csv_p.exists():
        rows = list(csv.DictReader(open(csv_p)))
        if rows:
            best = max(rows, key=lambda r: float(r.get("metrics/accuracy_top1", 0)))
            print(f"Best epoch: {best['epoch']}  top1={float(best.get('metrics/accuracy_top1',0)):.4f}")

    shutil.copy2(best_pt, CANDIDATE_OUT)
    print(f"Candidate saved: {CANDIDATE_OUT.name}")
    return best_pt


if __name__ == "__main__":
    run()
