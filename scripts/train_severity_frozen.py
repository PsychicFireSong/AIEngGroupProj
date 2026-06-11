"""
train_severity_frozen.py — Severity classifier via proper frozen-backbone transfer learning.

Why previous runs (sev_v3, sev_v4) failed at ~65%:
  yolo11s-cls (5.4M params) trained end-to-end on only 1,800 images → overfit.
  Both runs peaked at ep6 then declined as val_loss diverged from train_loss.

Fix — 3-phase gradual unfreeze:
  Phase 1: freeze=10  → train ONLY the 3-class Classify head (layer 10, ~662K params)
           High LR (5e-4) is safe — backbone is frozen, head adapts quickly.
           Target: stable 68%+ before unfreezing anything.
  Phase 2: freeze=7   → unfreeze top 3 backbone blocks (C3k2+C2PSA) + head
           Lower LR (1e-4). Let top-level features adapt to defect semantics.
  Phase 3: freeze=0   → full fine-tune at very low LR (2e-5)
           Entire network, minimal perturbation to avoid forgetting Phase 1-2.

YOLO classify freeze=N freezes the first N layers (0-indexed).
yolo11s-cls has layers 0-9 (backbone) + layer 10 (Classify head).
freeze=10 → all backbone frozen, only head trains. ✓

Dataset: severity_dataset (1,800 train / 360 val / 360 test, balanced 3-class)
Expected: 72-82% top-1 — better than 65.8% baseline without the overfit collapse.
"""
from __future__ import annotations

import ctypes
import csv
import shutil
import sys
from pathlib import Path

DATASET   = Path(
    r"C:\Users\User\AIEngGroupProj\output"
    r"\continued_hn_weak\current\datasets\severity_dataset"
)
OUT_DIR   = Path(r"C:\Users\User\AIEngGroupProj\output\severity_runs")
BEST_OUT  = Path(r"C:\Users\User\AIEngGroupProj\weights\severity_cls.pt")

PHASES = [
    # (name, freeze, lr0, epochs, warmup)
    ("sev_frozen_p1", 10, 5e-4, 30, 3),   # head only
    ("sev_frozen_p2",  7, 1e-4, 25, 1),   # top blocks + head
    ("sev_frozen_p3",  0, 2e-5, 20, 1),   # full fine-tune
]


def keep_awake() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def best_top1(run_dir: Path) -> float:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return 0.0
    try:
        rows = list(csv.DictReader(open(csv_path)))
        return max((float(r.get("metrics/accuracy_top1", 0)) for r in rows), default=0.0)
    except Exception:
        return 0.0


def run() -> None:
    if not DATASET.exists():
        print(f"ERROR: severity dataset not found: {DATASET}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep_awake()

    from ultralytics import YOLO

    start_weights = "yolo11s-cls.pt"

    for phase_name, freeze, lr0, total_ep, warmup in PHASES:
        run_dir  = OUT_DIR / phase_name
        best_pt  = run_dir / "weights" / "best.pt"
        last_pt  = run_dir / "weights" / "last.pt"
        results  = run_dir / "results.csv"

        # Count done epochs for resume support
        done = 0
        if results.exists():
            try:
                with open(results) as f:
                    done = max(0, sum(1 for _ in f) - 1)
            except Exception:
                pass

        remaining = total_ep - done

        if remaining <= 0:
            top1 = best_top1(run_dir)
            print(f"{phase_name}: already complete (best top1={top1:.4f}). Skipping.")
            if best_pt.exists():
                start_weights = str(best_pt)
            elif last_pt.exists():
                start_weights = str(last_pt)
            continue

        # Resume from last checkpoint if interrupted mid-phase
        if last_pt.exists() and done > 0:
            phase_start = str(last_pt)
            print(f"\n{phase_name}: resuming from ep{done}, {remaining} epochs left")
        else:
            phase_start = start_weights
            print(f"\n{phase_name}: starting from {phase_start}")

        frozen_params = "head only" if freeze == 10 else f"freeze={freeze}"
        print(f"  freeze={freeze} ({frozen_params}), lr={lr0:.0e}, {remaining} ep, warmup={warmup}")
        print(f"  ~3-4 min/epoch → ~{round(remaining * 3.5 / 60, 1)} hrs")

        model = YOLO(phase_start)
        model.train(
            data=str(DATASET),
            epochs=remaining,
            imgsz=224,
            batch=32,
            workers=0,
            device="0",
            freeze=freeze,

            optimizer="AdamW",
            lr0=lr0,
            lrf=0.01,
            weight_decay=5e-4,
            warmup_epochs=warmup,
            cos_lr=True,

            label_smoothing=0.1,

            # Moderate augmentation — enough to regularise without destroying small crops
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.3,
            fliplr=0.5,
            flipud=0.2,
            scale=0.15,
            erasing=0.1,

            patience=15,
            save_period=5,
            exist_ok=True,
            project=str(OUT_DIR),
            name=phase_name,
        )

        top1 = best_top1(run_dir)
        print(f"\n{phase_name} best top1: {top1:.4f} ({top1*100:.1f}%)")

        # Next phase starts from this phase's best
        start_weights = str(best_pt) if best_pt.exists() else str(last_pt)

    # Determine overall best across all phases
    best_top1_val = 0.0
    best_weights  = None
    for phase_name, *_ in PHASES:
        run_dir = OUT_DIR / phase_name
        t = best_top1(run_dir)
        pt = run_dir / "weights" / "best.pt"
        if t > best_top1_val and pt.exists():
            best_top1_val = t
            best_weights  = pt

    if best_weights:
        BEST_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, BEST_OUT)
        print(f"\nOverall best top1 : {best_top1_val:.4f} ({best_top1_val*100:.1f}%)")
        print(f"Saved             : {BEST_OUT}")
        baseline = 0.658
        delta = best_top1_val - baseline
        print(f"vs baseline 65.8% : {'+' if delta>=0 else ''}{delta:+.4f}")
        if best_top1_val >= 0.90:
            print("TARGET MET: >90% achieved!")
        elif best_top1_val >= 0.80:
            print("Good improvement — consider per-defect classifiers to push to 90%.")
        else:
            print("Marginal improvement — dataset ceiling likely ~65-70% without relabeling.")


if __name__ == "__main__":
    run()
