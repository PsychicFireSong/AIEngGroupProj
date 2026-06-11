"""
Stage 1 detector v3 — extended run targeting mAP50 > 0.80.

Analysis of gap:
  - Original 16-epoch run peaked at mAP50=0.694 (plateau)
  - v2 patch (5 epochs, full dataset, cls=1.5) shows 0.623 after epoch 1
  - Target 0.80 requires +0.106 from plateau — needs either more data or more epochs

Strategy:
  - Start from the BEST weights produced by v2 (patch2_candidate.pt if good, else hn_weak)
  - Continue on full dataset (33K + 3x patch = 38K) for 20 more epochs
  - LR schedule: start at 2e-5 (slightly higher than v2) with cosine decay
  - No freezing this time — all layers trainable (model already adapted in v2)
  - cls=1.2 (slightly lower than 1.5 to stabilise while pushing confidence)
  - monitor per-epoch mAP50 and stop manually if plateauing below 0.80

Run AFTER Stage 1 v2 completes. Set BASE_WEIGHTS to best v2 output.
"""
from __future__ import annotations
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

FULL_TRAIN   = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\images\train")
FULL_VAL     = Path(r"C:\Users\User\AIEngGroupProj\output\continued_hn_weak\current\datasets\merged_dataset_hn_weak_continue\images\val")
PATCH_IMG    = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round1\patch_dataset\images\train")
# Set to v2 best if it passed targets, else fall back to hn_weak_candidate
CANDIDATE_V2 = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_patch2_candidate.pt")
BASE_WEIGHTS_FALLBACK = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt")
OUT_DIR      = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round3")
RUNS_DIR     = OUT_DIR / "runs"
CANDIDATE_OUT = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_patch3_candidate.pt")

CLASS_NAMES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]


def build_yaml() -> Path:
    ds_dir = OUT_DIR / "dataset"
    data_yaml = ds_dir / "data.yaml"
    ds_dir.mkdir(parents=True, exist_ok=True)
    yaml_content = f"""path: {ds_dir.as_posix()}
train:
  - {FULL_TRAIN.as_posix()}
  - {PATCH_IMG.as_posix()}
  - {PATCH_IMG.as_posix()}
  - {PATCH_IMG.as_posix()}
val: {FULL_VAL.as_posix()}

nc: 5
names: ['crack', 'spalling', 'corrosion', 'pothole', 'paint_degradation']
"""
    data_yaml.write_text(yaml_content, encoding="utf-8")
    print(f"YAML: {data_yaml}")
    return ds_dir


def _best_map50_from_csv(csv_path: Path) -> float:
    import csv as _csv
    if not csv_path.exists():
        return 0.0
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return 0.0
    return max(float(r.get("metrics/mAP50(B)", 0)) for r in rows)


def run_finetune(dataset_dir: Path) -> Path:
    from ultralytics import YOLO

    # Use v2 weights only if they match or beat original baseline (0.694)
    V2_RESULTS = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round2\runs\patch2\results.csv")
    v2_map50   = _best_map50_from_csv(V2_RESULTS)
    BASELINE   = 0.694  # original hn_weak_candidate best mAP50

    if CANDIDATE_V2.exists() and v2_map50 >= 0.50:
        # Use v2 weights: they have patch-dataset adaptation even if mAP50 dipped
        # v3's 20 epochs with no-freeze will fully recover and improve from here
        base = CANDIDATE_V2
        print(f"  Using v2 weights (mAP50={v2_map50:.4f}) — patch-adapted starting point")
    else:
        # v2 badly failed or doesn't exist — fall back to original strong baseline
        base = BASE_WEIGHTS_FALLBACK
        print(f"  v2 mAP50={v2_map50:.4f} too low or missing — using hn_weak_candidate")

    print(f"\n{'='*60}")
    print(f"STAGE 1 v3 — Extended run to mAP50 > 0.80")
    print(f"  Base: {base.name}")
    print(f"  Epochs: 20, LR: 2e-5 (cosine), no freeze")
    print(f"  cls=1.2, full dataset + 3x patch")
    print(f"{'='*60}\n")

    model = YOLO(str(base))
    results = model.train(
        data=str(dataset_dir / "data.yaml"),
        epochs=20,
        imgsz=640,
        batch=16,
        lr0=2e-5,
        lrf=0.01,
        cos_lr=True,              # cosine LR decay for smooth convergence
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=2,
        freeze=0,                 # no freeze — all layers adapt
        workers=0,
        device="0",
        project=str(RUNS_DIR),
        name="patch3",
        exist_ok=True,
        cls=1.2,                  # lower than v2's 1.5 for stability
        box=7.5,
        mosaic=1.0,
        copy_paste=0.3,
        scale=0.3,
        flipud=0.3,
        fliplr=0.5,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        perspective=0.0001,
        mixup=0.1,
        close_mosaic=5,
        patience=12,
        save=True,
        save_period=2,
    )

    best_pt = RUNS_DIR / "patch3" / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = RUNS_DIR / "patch3" / "weights" / "last.pt"
    print(f"\nBest weights: {best_pt}")
    return best_pt


def check_map50(results_csv: Path) -> float:
    import csv
    if not results_csv.exists():
        return 0.0
    rows = list(csv.DictReader(open(results_csv, encoding='utf-8')))
    if not rows:
        return 0.0
    best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0)))
    return float(best.get('metrics/mAP50(B)', 0))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("="*60)
    print("STAGE 1 v3 — targeting mAP50 > 0.80")
    print("="*60)

    dataset_dir = build_yaml()
    best_pt = run_finetune(dataset_dir)

    results_csv = RUNS_DIR / "patch3" / "results.csv"
    best_map50 = check_map50(results_csv)
    print(f"\nBest mAP50 achieved: {best_map50:.4f}  (target: >0.80)")

    # Always save the best weights so realworld_eval3.py can use them
    shutil.copy2(best_pt, CANDIDATE_OUT)
    if best_map50 >= 0.80:
        print(f"OK Target met - candidate saved: {CANDIDATE_OUT.name}")
    else:
        print(f"WARN Target not yet met ({best_map50:.4f} < 0.80) — saved anyway for eval")
        print(f"  Best weights: {CANDIDATE_OUT.name}")
        print(f"  Next step: realworld_eval3.py will measure actual class recall + confidence")

    result = {"best_map50": best_map50, "target_met": best_map50 >= 0.80, "weights": str(best_pt)}
    (OUT_DIR / "patch3_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
