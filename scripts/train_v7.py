"""
train_v7.py - YOLO11m progressive-unfreeze fine-tune from baseline.

Strategy (two-phase, avoids catastrophic forgetting):
  Phase 1 - freeze=10, 10 epochs, lr=0.002
    Backbone frozen. Only detection head adapts to new data distribution.
    Fast (~12 min/epoch), no risk of forgetting existing class knowledge.

  Phase 2 - freeze=0, 25 epochs, lr=0.0005
    Full model fine-tunes. Backbone adapts to white/cream walls and new
    corrosion variants. Low LR prevents overwriting Phase 1 gains.

Fixes applied vs baseline (0.694 mAP50):
  DATA
    + CDD light-colored wall images (domain gap fix)
    + 2x oversample crack and corrosion (low-recall fix)
    + Hard negatives (paint_degradation FP fix)

  AUGMENTATION
    hsv_h=0.2, hsv_s=0.9, hsv_v=0.8   color invariance across surface types
    erasing=0.1                          was 0.4, was hiding defects
    copy_paste=0.0                       disabled: doubles CPU load (workers=0)

  LOSS
    cls=2.0                              up from 1.5 - pushes class separation
    fl_gamma=2.0                         focal loss focuses on hard examples

Estimated time:
  Phase 1: 10 x ~12 min = ~2 h
  Phase 2: 25 x ~14 min = ~6 h
  Total  : ~8 h  (overnight)
"""
from __future__ import annotations

import csv
import ctypes
import shutil
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_YAML  = Path(r"C:\Users\User\AIEngGroupProj\output\v7_fixed_dataset\data.yaml")
OUT_DIR       = Path(r"C:\Users\User\AIEngGroupProj\output\v7_runs")
BASE_WEIGHTS  = Path(
    r"C:\Users\User\AIEngGroupProj_colab_outputs"
    r"\continued_hn_weak_finetune\runs\current\weights"
    r"\defect_detector_hn_weak_candidate.pt"
)
CANDIDATE_OUT = Path(
    r"C:\Users\User\AIEngGroupProj_colab_outputs"
    r"\continued_hn_weak_finetune\runs\current\weights"
    r"\defect_detector_v7_candidate.pt"
)

PHASE1_OUT = OUT_DIR / "v7_phase1"
PHASE2_OUT = OUT_DIR / "v7_phase2b"


# ── Helpers ───────────────────────────────────────────────────────────────────

def best_map50(csv_path: Path) -> float:
    if not csv_path.exists():
        return 0.0
    try:
        rows = list(csv.DictReader(csv_path.open()))
        return max((float(r.get("metrics/mAP50(B)", 0)) for r in rows), default=0.0)
    except Exception:
        return 0.0


def keep_awake() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("  [sleep-guard] Windows sleep inhibited")
    except Exception:
        pass


COMMON_AUG = dict(
    hsv_h=0.2,        # hue invariance (gray concrete <-> cream/white walls)
    hsv_s=0.9,
    hsv_v=0.8,        # brightness range (dark concrete <-> bright plaster)
    copy_paste=0.0,   # disabled: doubles CPU load with workers=0
    mosaic=1.0,
    scale=0.5,
    perspective=0.001,
    degrees=15.0,
    flipud=0.3,
    fliplr=0.5,
    mixup=0.1,
    erasing=0.1,      # was 0.4 - was hiding defect regions during training
)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    if not DATASET_YAML.exists():
        print(f"ERROR: dataset not found: {DATASET_YAML}")
        print("Run build_v7_fixed.py first.")
        sys.exit(1)

    if not BASE_WEIGHTS.exists():
        print(f"ERROR: baseline weights not found: {BASE_WEIGHTS}")
        sys.exit(1)

    keep_awake()

    from ultralytics import YOLO

    # ── Phase 1: head-only warmup ─────────────────────────────────────────────
    p1_best = PHASE1_OUT / "weights" / "best.pt"
    p1_last = PHASE1_OUT / "weights" / "last.pt"

    if p1_best.exists() or p1_last.exists():
        print("Phase 1 weights found - skipping to Phase 2.")
    else:
        print("=" * 64)
        print("PHASE 1/2  -  head warmup  (freeze=10, 10 epochs, lr=0.002)")
        print("=" * 64)
        model1 = YOLO(str(BASE_WEIGHTS))
        model1.train(
            data=str(DATASET_YAML),
            epochs=10,
            imgsz=640,
            batch=16,
            workers=0,
            device="0",
            freeze=10,          # freeze first 10 layers (backbone)

            optimizer="SGD",
            lr0=0.002,          # higher LR is fine since backbone is frozen
            lrf=0.1,
            momentum=0.937,
            weight_decay=1e-4,
            warmup_epochs=2,
            cos_lr=True,

            cls=2.0,            # stronger class separation signal

            **COMMON_AUG,

            cache="disk",
            patience=10,
            save_period=5,
            exist_ok=True,
            project=str(OUT_DIR),
            name="v7_phase1",
        )

    p1_weights = p1_best if p1_best.exists() else p1_last
    p1_map50 = best_map50(PHASE1_OUT / "results.csv")
    print(f"\nPhase 1 best mAP50: {p1_map50:.4f}")

    # ── Phase 2: full model fine-tune ─────────────────────────────────────────
    p2_best = PHASE2_OUT / "weights" / "best.pt"
    p2_last = PHASE2_OUT / "weights" / "last.pt"

    if not (p2_best.exists() or p2_last.exists()):
        print("\n" + "=" * 64)
        print("PHASE 2/2  -  full fine-tune (freeze=0, 25 epochs, lr=0.0001)")
        print("=" * 64)
        model2 = YOLO(str(p1_weights))
        model2.train(
            data=str(DATASET_YAML),
            epochs=25,
            imgsz=640,
            batch=8,
            workers=0,
            device="0",
            freeze=0,           # unfreeze everything

            optimizer="SGD",
            lr0=1e-4,           # reduced from 5e-4 - prevents backbone disruption
            lrf=0.01,
            momentum=0.937,
            weight_decay=1e-4,
            warmup_epochs=3,
            cos_lr=True,

            cls=2.0,

            **COMMON_AUG,

            cache="disk",
            patience=15,
            save_period=5,
            exist_ok=True,
            project=str(OUT_DIR),
            name="v7_phase2b",
        )

    best_pt = p2_best if p2_best.exists() else p2_last
    CANDIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, CANDIDATE_OUT)

    p2_map50 = best_map50(PHASE2_OUT / "results.csv")
    baseline = 0.694
    status   = "BEATS BASELINE" if p2_map50 > baseline else f"below baseline ({baseline})"
    print(f"\nPhase 2 best mAP50 : {p2_map50:.4f}  ({status})")
    print(f"Saved to           : {CANDIDATE_OUT}")


if __name__ == "__main__":
    run()
