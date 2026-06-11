"""
Patch fine-tune for Stage 2 severity classifier.

Issue: 47% real-world accuracy vs 60.8% training-time top-1.
Root cause: generalization gap — model overfit to training crop distribution.

Fixes:
  - 30 epochs (vs 20 previously)
  - Stronger augmentation: mixup, erasing, sharper color jitter
  - Lower LR (1e-4) with cosine decay
  - Workers=0 (Windows)
"""
from __future__ import annotations
from pathlib import Path
import shutil

SEVERITY_DS = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\severity_dataset")
BASE_WEIGHTS = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls.pt")
OUT_DIR = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round1\severity_runs")
CANDIDATE_OUT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\severity_cls_patch1_candidate.pt")


def run_severity_retrain() -> Path:
    from ultralytics import YOLO

    print("="*60)
    print("STAGE 2 SEVERITY CLASSIFIER PATCH")
    print("="*60)
    print(f"  Base: {BASE_WEIGHTS.name}")
    print(f"  Dataset: {SEVERITY_DS}")
    print(f"  Epochs: 30 (was 20)")
    print()

    model = YOLO(str(BASE_WEIGHTS))
    results = model.train(
        data=str(SEVERITY_DS),  # classification: pass directory, not yaml
        epochs=30,
        imgsz=224,
        batch=32,
        lr0=1e-4,
        lrf=0.01,
        momentum=0.9,
        weight_decay=1e-3,
        warmup_epochs=3,
        workers=0,
        device="0",
        project=str(OUT_DIR),
        name="sev_patch1",
        exist_ok=True,
        # Strong augmentation to improve generalization
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        flipud=0.4,
        fliplr=0.5,
        degrees=15.0,
        translate=0.1,
        scale=0.3,
        erasing=0.3,
        mixup=0.2,
        copy_paste=0.0,
        close_mosaic=0,
        save=True,
        save_period=5,
    )

    best_pt = OUT_DIR / "sev_patch1" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = OUT_DIR / "sev_patch1" / "weights" / "last.pt"
    print(f"\nBest weights: {best_pt}")

    shutil.copy2(best_pt, CANDIDATE_OUT)
    print(f"Candidate saved: {CANDIDATE_OUT.name}")
    return best_pt


if __name__ == "__main__":
    run_severity_retrain()
