"""
Stage 1 v6 — properly balanced patch fine-tuning.

v5 failure reason: patch_round1 had 0 pothole / 87 spalling / 105 crack images.
Oversampling minority classes with 1x did nothing meaningful.

v6 fix: build_balanced_patch.py extracts ~500 images per class from original
source datasets (with label remapping). Each class is equally represented.

Training is still gentle fine-tuning from the 0.694 hn_weak baseline:
  - freeze=0, lr0=2e-4 (low LR is the forgetting guard, not layer freezing)
  - 20 epochs / patience=15 / warmup=3ep (shorter: data is balanced, no need for long ramp)
"""
from __future__ import annotations
from pathlib import Path
import shutil, csv

BASE_WEIGHTS  = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt")
YAML          = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round6\dataset\data.yaml")
OUT_DIR       = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round6\runs")
CANDIDATE_OUT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_patch6_candidate.pt")


def _best_map50(csv_path: Path) -> float:
    if not csv_path.exists():
        return 0.0
    try:
        rows = list(csv.DictReader(open(csv_path)))
        return max(float(r.get("metrics/mAP50(B)", 0)) for r in rows) if rows else 0.0
    except Exception:
        return 0.0


LAST_PT = OUT_DIR / "patch6" / "weights" / "last.pt"


def run() -> None:
    import ctypes, sys
    # Prevent Windows from sleeping during training
    try:
        ES_CONTINUOUS      = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("  [sleep guard] Windows sleep inhibited for training duration")
    except Exception:
        pass

    from ultralytics import YOLO

    if not YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {YAML}\n"
            "Run build_balanced_patch.py first."
        )

    # Resume from last checkpoint if available (handles crashes/sleep interruptions)
    if LAST_PT.exists():
        print(f"  Resuming from last checkpoint: {LAST_PT}")
        model = YOLO(str(LAST_PT))
        resume = True
    else:
        model = YOLO(str(BASE_WEIGHTS))
        resume = False

    print("=" * 60)
    print("STAGE 1 v6 — balanced patch fine-tuning")
    print("=" * 60)
    print(f"  Base: {BASE_WEIGHTS.name}  (mAP50=0.694)")
    print(f"  Data: {YAML}")
    print(f"  Balanced: ~500 images per class from source datasets")
    print(f"  optimizer=SGD  lr0=2e-4  warmup=3ep  scale=0.5  copy_paste=0.5")
    print(f"  Epochs: 20, patience=15, cos_lr=True  resume={resume}")
    print()

    model.train(
        data=str(YAML),
        epochs=20,
        imgsz=640,
        batch=12,
        workers=0,
        device="0",

        # ── Optimiser ──────────────────────────────────────────
        optimizer="SGD",
        lr0=2e-4,
        lrf=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        warmup_epochs=3,       # shorter warmup: balanced data doesn't need long ramp
        warmup_bias_lr=2e-4,
        warmup_momentum=0.8,
        cos_lr=True,

        # ── Loss weights ───────────────────────────────────────
        cls=1.5,

        # ── Augmentation ───────────────────────────────────────
        mosaic=1.0,
        copy_paste=0.5,
        scale=0.5,
        perspective=0.001,
        degrees=20.0,
        flipud=0.3,
        fliplr=0.5,
        mixup=0.1,
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,

        # ── Training config ────────────────────────────────────
        freeze=0,
        patience=15,
        save_period=2,
        exist_ok=True,
        resume=resume,
        project=str(OUT_DIR),
        name="patch6",
    )

    best_pt = OUT_DIR / "patch6" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = OUT_DIR / "patch6" / "weights" / "last.pt"

    shutil.copy2(best_pt, CANDIDATE_OUT)

    best     = _best_map50(OUT_DIR / "patch6" / "results.csv")
    baseline = 0.694
    status   = "BEATS BASELINE" if best > baseline else "below baseline"
    print(f"\nBest mAP50: {best:.4f}  ({status}, baseline={baseline})")
    print(f"Saved: {CANDIDATE_OUT.name}")


if __name__ == "__main__":
    run()
