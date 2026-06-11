"""
Stage 1 detector v4 — correct fine-tuning hyperparameters.

Root cause of v3 failure:
  optimizer=auto overrode lr0=2e-5 with MuSGD(lr=0.01) — 500x higher than intended.
  With freeze=0 and warmup_bias_lr=0.1, epoch 1 collapsed from 0.694 to 0.413.
  First 10 epochs were spent recovering rather than improving.

v4 fixes:
  - optimizer="SGD"  — explicit, YOLO will not override this
  - lr0=1e-3         — 10x gentler than the 0.01 that broke v3
  - warmup_bias_lr=1e-3 — same as lr0, no warmup surge on biases
  - Starting from hn_weak_candidate.pt (0.694) — clean baseline, above v3 best
  - 15 epochs with cos_lr — enough room to improve without wasting GPU time
"""
from __future__ import annotations
from pathlib import Path
import shutil, csv

BASE_WEIGHTS  = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt")
YAML          = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round3\dataset\data.yaml")
OUT_DIR       = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round4\runs")
CANDIDATE_OUT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_patch4_candidate.pt")


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
    print("STAGE 1 v4 — fixed fine-tuning hyperparameters")
    print("=" * 60)
    print(f"  Base: {BASE_WEIGHTS.name}  (mAP50=0.694)")
    print(f"  optimizer=SGD  lr0=1e-3  warmup_bias_lr=1e-3")
    print(f"  Epochs: 15, patience=10, cos_lr=True")
    print()

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(YAML),
        epochs=15,
        imgsz=640,
        batch=16,
        workers=0,
        device="0",

        # --- Correct optimizer (v3 lesson) ---
        optimizer="SGD",          # explicit — prevents auto override
        lr0=1e-3,                 # gentle fine-tuning, 10x below what broke v3
        lrf=0.01,                 # cos decay to 1e-5
        momentum=0.9,
        weight_decay=5e-4,
        warmup_epochs=3,
        warmup_bias_lr=1e-3,      # same as lr0 — no warmup surge on biases
        warmup_momentum=0.8,
        cos_lr=True,

        # --- Regularisation ---
        cls=1.2,
        dropout=0.0,

        # --- Augmentation (same as v3) ---
        mosaic=1.0,
        copy_paste=0.3,
        scale=0.3,
        flipud=0.3,
        fliplr=0.5,
        mixup=0.1,
        erasing=0.4,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # --- Misc ---
        freeze=0,
        patience=10,
        save_period=2,
        exist_ok=True,
        project=str(OUT_DIR),
        name="patch4",
    )

    best_pt = OUT_DIR / "patch4" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = OUT_DIR / "patch4" / "weights" / "last.pt"

    shutil.copy2(best_pt, CANDIDATE_OUT)

    best = _best_map50(OUT_DIR / "patch4" / "results.csv")
    baseline = 0.694
    status = "BEATS BASELINE" if best > baseline else "below baseline"
    print(f"\nBest mAP50: {best:.4f}  ({status}, baseline={baseline})")
    print(f"Saved: {CANDIDATE_OUT.name}")


if __name__ == "__main__":
    run()
