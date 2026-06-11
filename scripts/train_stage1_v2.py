"""
train_stage1_v2.py — Stage 1 detector improvement: gradual unfreeze with CIoU.

Why NWD (Phase 2) failed (-0.032 vs baseline):
  NWD loss changed the gradient landscape too much during full unfreeze.
  Backbone representations drifted away from the Phase 1 calibrated head,
  causing crack (-0.041) and corrosion (-0.067) to degrade the most.

This approach — standard CIoU, gradual unfreeze, AdamW, tiny LR:
  Phase A: freeze=15, 8 epochs  — warm up top neck + head from Phase 1 weights
  Phase B: freeze=10, 8 epochs  — open mid-neck; let features adapt
  Phase C: freeze=5,  8 epochs  — open most of backbone
  Phase D: freeze=0,  12 epochs — full fine-tune, very low LR

AdamW replaces SGD (baseline used SGD). AdamW adapts per-parameter LRs and
generalises better on small/imbalanced datasets (corrosion=9.3% of instances).

Starting point: staged_phase1/best.pt (mAP50=0.683, head calibrated).
Baseline to beat: defect_detector_hn_weak_candidate.pt (mAP50=0.694).

Class distribution (v7_lean train):
  crack: 21.8% | spalling: 24.3% | corrosion: 9.3% | pothole: 15.7% | paint: 29.0%
  corrosion is 3× under-represented vs paint_degradation — primary weakness.

Note: Adding more corrosion data (Roboflow corrosion-bi3q3, CC BY 4.0) would
have higher impact than training tricks alone. Run integrate_extra_data.py first
if that dataset has been downloaded.
"""
from __future__ import annotations

import csv
import ctypes
import shutil
import sys
from pathlib import Path

DATASET_YAML  = Path(r"C:\Users\User\AIEngGroupProj\output\v7_lean_dataset\data.yaml")
PHASE1_BEST   = Path(
    r"C:\Users\User\AIEngGroupProj\output\staged_runs\staged_phase1\weights\best.pt"
)
OUT_DIR       = Path(r"C:\Users\User\AIEngGroupProj\output\staged_runs")
CANDIDATE_OUT = Path(
    r"C:\Users\User\AIEngGroupProj_colab_outputs"
    r"\continued_hn_weak_finetune\runs\current\weights"
    r"\defect_detector_staged_v2_candidate.pt"
)
BASELINE_MAP  = 0.694

# Gradual unfreeze schedule: (run_name, freeze_layers, lr0, epochs, warmup)
PHASES = [
    ("stage1v2_phA", 15, 3e-4, 8,  2),   # top neck + head only
    ("stage1v2_phB", 10, 2e-4, 8,  1),   # mid-neck unlocked
    ("stage1v2_phC",  5, 1e-4, 8,  1),   # most of backbone
    ("stage1v2_phD",  0, 3e-5, 12, 1),   # full network, tiny LR
]

COMMON_AUG = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    copy_paste=0.0,
    mosaic=1.0,
    scale=0.5, perspective=0.0005, degrees=10.0,
    flipud=0.3, fliplr=0.5,
    mixup=0.1,        # slight increase vs baseline (0.05) to help corrosion generalisation
    erasing=0.1,
    close_mosaic=3,
)


def keep_awake() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("  [sleep-guard] Windows sleep inhibited")
    except Exception:
        pass


def best_map50(run_dir: Path) -> float:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return 0.0
    try:
        rows = list(csv.DictReader(open(csv_path)))
        return max((float(r.get("metrics/mAP50(B)", 0)) for r in rows), default=0.0)
    except Exception:
        return 0.0


def run() -> None:
    for path, name in [(DATASET_YAML, "dataset"), (PHASE1_BEST, "phase1 weights")]:
        if not path.exists():
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    keep_awake()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    start_weights = str(PHASE1_BEST)

    for phase_name, freeze, lr0, total_ep, warmup in PHASES:
        run_dir = OUT_DIR / phase_name
        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        results = run_dir / "results.csv"

        done = 0
        if results.exists():
            try:
                with open(results) as f:
                    done = max(0, sum(1 for _ in f) - 1)
            except Exception:
                pass

        remaining = total_ep - done

        if remaining <= 0:
            m = best_map50(run_dir)
            print(f"{phase_name}: complete (mAP50={m:.4f}). Skipping.")
            start_weights = str(best_pt) if best_pt.exists() else str(last_pt)
            continue

        phase_start = str(last_pt) if (last_pt.exists() and done > 0) else start_weights
        resume_note = f"(resuming from ep{done})" if done > 0 else ""

        print("=" * 64)
        print(f"{phase_name}  freeze={freeze}  lr={lr0:.0e}  {remaining} ep  {resume_note}")
        print(f"  ~23 min/ep × {remaining} = ~{round(remaining*23/60,1)} hrs")
        print("=" * 64)

        model = YOLO(phase_start)
        model.train(
            data=str(DATASET_YAML),
            epochs=remaining,
            imgsz=640,
            batch=12,
            workers=0,
            device="0",
            freeze=freeze,

            optimizer="AdamW",      # vs SGD in baseline — adapts per-param LR
            lr0=lr0,
            lrf=0.01,
            momentum=0.937,
            weight_decay=5e-4,
            warmup_epochs=warmup,
            cos_lr=True,

            cls=2.0,               # slightly higher cls loss — helps imbalanced corrosion
            box=7.5,
            dfl=1.5,
            label_smoothing=0.1,

            **COMMON_AUG,

            cache="disk",
            patience=6,            # short patience per phase — phases are short
            save_period=4,
            exist_ok=True,
            project=str(OUT_DIR),
            name=phase_name,
        )

        m = best_map50(run_dir)
        print(f"\n{phase_name} best mAP50: {m:.4f}")
        start_weights = str(best_pt) if best_pt.exists() else str(last_pt)

    # Overall best across all phases
    best_map = 0.0
    best_wt  = None
    for phase_name, *_ in PHASES:
        m  = best_map50(OUT_DIR / phase_name)
        pt = OUT_DIR / phase_name / "weights" / "best.pt"
        if m > best_map and pt.exists():
            best_map = m
            best_wt  = pt

    if best_wt:
        CANDIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_wt, CANDIDATE_OUT)
        delta  = best_map - BASELINE_MAP
        status = "BEATS BASELINE" if best_map > BASELINE_MAP else f"below baseline ({BASELINE_MAP})"
        print(f"\nBest mAP50 across all phases : {best_map:.4f}  ({status})")
        print(f"Delta vs baseline            : {'+' if delta>=0 else ''}{delta:+.4f}")
        print(f"Saved to                     : {CANDIDATE_OUT}")


if __name__ == "__main__":
    run()
