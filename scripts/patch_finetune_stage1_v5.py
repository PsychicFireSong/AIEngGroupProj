"""
Stage 1 v5 — balanced patch fine-tuning.

v4 lessons:
  - Patch dataset is skewed: 3034 paint_deg / 1908 corrosion / 105 crack / 87 spalling / 0 pothole
  - 3x oversample turned ~13% skew into catastrophic forgetting on crack/spalling/corrosion
  - Even corrosion regressed despite having many patches (patch distribution != val distribution)

v5 fixes:
  1. 1x patch ratio (no oversampling) — keeps paint_deg gain, removes skew amplification
  2. scale=0.5 (was 0.3) — simulates small/distant defects for ALL classes via augmentation
  3. copy_paste=0.5 (was 0.3) — helps pothole (0 patches) by pasting instances at varied scales
  4. perspective=0.001 + degrees=20 — covers angle variation equally across all classes
  5. lr0=2e-4 — 5x gentler than v4 (1e-3), prevents forgetting strong classes
  6. warmup_epochs=5 — longer ramp to protect baseline weights
  7. 20 epochs / patience=15 — enough room without wasting compute
  8. Starts from hn_weak baseline (0.694), not from v4 best
"""
from __future__ import annotations
from pathlib import Path
import shutil, csv

BASE_WEIGHTS  = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt")
YAML          = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round5\dataset\data.yaml")
OUT_DIR       = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round5\runs")
CANDIDATE_OUT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_patch5_candidate.pt")


def _best_map50(csv_path: Path) -> float:
    if not csv_path.exists():
        return 0.0
    try:
        rows = list(csv.DictReader(open(csv_path)))
        return max(float(r.get("metrics/mAP50(B)", 0)) for r in rows) if rows else 0.0
    except Exception:
        return 0.0


def run() -> None:
    from ultralytics import YOLO

    print("=" * 60)
    print("STAGE 1 v5 — balanced patch fine-tuning")
    print("=" * 60)
    print(f"  Base: {BASE_WEIGHTS.name}  (mAP50=0.694)")
    print(f"  Patch: 1x (no oversample) — eliminates skew amplification")
    print(f"  optimizer=SGD  lr0=2e-4  warmup=5ep  scale=0.5  copy_paste=0.5")
    print(f"  Epochs: 20, patience=15, cos_lr=True")
    print()

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(YAML),
        epochs=20,
        imgsz=640,
        batch=16,
        workers=0,
        device="0",

        # ── Optimiser ──────────────────────────────────────────
        optimizer="SGD",
        lr0=2e-4,             # 5x gentler than v4; preserves strong-class weights
        lrf=0.01,             # cosine decay to 2e-6
        momentum=0.9,
        weight_decay=5e-4,
        warmup_epochs=5,      # longer ramp — protects crack/spalling/corrosion baseline
        warmup_bias_lr=2e-4,
        warmup_momentum=0.8,
        cos_lr=True,

        # ── Loss weights ───────────────────────────────────────
        cls=1.5,              # slightly higher class loss — encourages sharper per-class discrimination

        # ── Augmentation ───────────────────────────────────────
        mosaic=1.0,
        copy_paste=0.5,       # was 0.3; paste defect crops at varied scales — main pothole fix
        scale=0.5,            # was 0.3; simulates small/distant defects for ALL classes equally
        perspective=0.001,    # slight perspective warp — angle variation at zero distribution cost
        degrees=20.0,         # was 10; more rotation — handheld/drone angle variety
        flipud=0.3,
        fliplr=0.5,
        mixup=0.1,
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,            # was 0.4; broader brightness range for dark/overexposed captures

        # ── Training config ────────────────────────────────────
        freeze=0,
        patience=15,
        save_period=2,
        exist_ok=True,
        project=str(OUT_DIR),
        name="patch5",
    )

    best_pt = OUT_DIR / "patch5" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = OUT_DIR / "patch5" / "weights" / "last.pt"

    shutil.copy2(best_pt, CANDIDATE_OUT)

    best = _best_map50(OUT_DIR / "patch5" / "results.csv")
    baseline = 0.694
    status = "BEATS BASELINE" if best > baseline else "below baseline"
    print(f"\nBest mAP50: {best:.4f}  ({status}, baseline={baseline})")
    print(f"Saved: {CANDIDATE_OUT.name}")


if __name__ == "__main__":
    run()
