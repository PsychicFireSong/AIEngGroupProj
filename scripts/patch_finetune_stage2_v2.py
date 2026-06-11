"""
Stage 2 severity classifier v2 — upgraded backbone.

Problem: YOLO11n-cls (1.5M params) is too small; 65.8% top-1 ceiling.
Fix: YOLO11s-cls (9.4M params, 6x bigger) trained from scratch on severity dataset.
Target: top-1 > 90% on val set.

Changes from v1:
  - Model: yolo11n-cls -> yolo11s-cls (scratch, not from severity_cls.pt)
  - Epochs: 80 (more capacity needs more training)
  - Label smoothing: 0.1 (reduce overfit to noisy labels)
  - Dropout: 0.3 (regularise larger model)
  - Patience: 20 (early stop if stuck)
  - cls=1.0 (default for classification)
"""
from __future__ import annotations
from pathlib import Path
import shutil

SEVERITY_DS  = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\severity_dataset")
OUT_DIR      = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round2\severity_runs")
CANDIDATE_OUT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls_patch2_candidate.pt")


def run() -> Path:
    from ultralytics import YOLO

    print("="*60)
    print("STAGE 2 SEVERITY v2 — YOLO11s-cls (9.4M params)")
    print("="*60)
    print(f"  Model: yolo11s-cls (scratch, not nano)")
    print(f"  Dataset: {SEVERITY_DS}")
    print(f"  Epochs: 80, patience=20 (early stop)")
    print(f"  Target: top-1 > 90%")
    print()

    model = YOLO("yolo11s-cls.pt")  # download pretrained YOLO11s classification
    results = model.train(
        data=str(SEVERITY_DS),
        epochs=80,
        imgsz=224,
        batch=32,
        lr0=5e-4,
        lrf=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        warmup_epochs=5,
        patience=20,              # early stop if val doesn't improve for 20 epochs
        dropout=0.3,              # regularise larger model
        label_smoothing=0.1,      # soften noisy labels
        workers=0,
        device="0",
        project=str(OUT_DIR),
        name="sev_v2",
        exist_ok=True,
        # Strong augmentation
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        flipud=0.4,
        fliplr=0.5,
        degrees=20.0,
        translate=0.1,
        scale=0.4,
        erasing=0.4,
        mixup=0.3,
        auto_augment="randaugment",
        copy_paste=0.0,
        close_mosaic=0,
        save=True,
        save_period=10,
    )

    best_pt = OUT_DIR / "sev_v2" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = OUT_DIR / "sev_v2" / "weights" / "last.pt"
    print(f"\nBest weights: {best_pt}")
    shutil.copy2(best_pt, CANDIDATE_OUT)
    print(f"Candidate saved: {CANDIDATE_OUT.name}")
    return best_pt


if __name__ == "__main__":
    run()
