"""
train_staged.py  —  Two-phase staged fine-tune toward mAP50 > 0.80.

Why staged (not one-shot full unfreeze):
  v7 failure: clean-start full unfreeze caused catastrophic forgetting.
  Root cause was NOT the LR alone — the detection head recalibrated away
  from the original distribution before the backbone could stabilise.
  Staged approach: Phase 1 adapts the head first (backbone locked),
  Phase 2 fine-tunes everything with NWD loss from a stable head.

Phase 1 — head-only warmup  (freeze=20, batch=16, ~8 min/epoch)
  Freeze first 20 backbone/neck layers, train only the last neck block
  + detection head. Head learns confidence calibration on clean data.
  FAST: 10 epochs x ~8 min = ~1.5 hours.
  Expected: head recalibrates toward 0.70+ AP without forgetting backbone.

Phase 2 — NWD full fine-tune  (freeze=0, batch=8, ~24 min/epoch)
  Full model with NWD box loss. Thin cracks and corrosion get proper
  gradient signals (NWD vs CIoU: gradient doesn't vanish for tiny boxes).
  SAFE: batch=8 keeps VRAM under 6GB (Phase 2b ran fine at batch=8).
  15 epochs x ~24 min = ~6 hours.

Total estimated time: ~7.5 hours.
Expected mAP50: 0.72-0.76 (vs baseline 0.694).

Dataset: v7_lean_dataset — 20,166 train, ORIGINAL (no oversampling, no hard negatives).
"""
from __future__ import annotations

import csv
import ctypes
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_YAML  = Path(r"C:\Users\User\AIEngGroupProj\output\v7_lean_dataset\data.yaml")
BASE_WEIGHTS  = Path(
    r"C:\Users\User\AIEngGroupProj_colab_outputs"
    r"\continued_hn_weak_finetune\runs\current\weights"
    r"\defect_detector_hn_weak_candidate.pt"
)
OUT_DIR       = Path(r"C:\Users\User\AIEngGroupProj\output\staged_runs")
CANDIDATE_OUT = Path(
    r"C:\Users\User\AIEngGroupProj_colab_outputs"
    r"\continued_hn_weak_finetune\runs\current\weights"
    r"\defect_detector_staged_candidate.pt"
)

PHASE1_DIR = OUT_DIR / "staged_phase1"
PHASE2_DIR = OUT_DIR / "staged_phase2"

NWD_C = 12.8

# These imports must be at module level so NWDBboxLoss is a top-level class
# and can be pickled by torch.save() when YOLO saves last.pt / best.pt.
from ultralytics.utils.loss import DFLoss
from ultralytics.utils.tal  import bbox2dist


# ── NWD patch (Phase 2 only) ───────────────────────────────────────────────────
# NWDBboxLoss is defined at MODULE level (not inside a function) so that
# pickle/torch.save can serialise it when writing checkpoints.

def _nwd_loss(pred: torch.Tensor, target: torch.Tensor, stride: torch.Tensor) -> torch.Tensor:
    p = pred   * stride
    t = target * stride
    pcx = (p[:, 0] + p[:, 2]) / 2;  pcy = (p[:, 1] + p[:, 3]) / 2
    tcx = (t[:, 0] + t[:, 2]) / 2;  tcy = (t[:, 1] + t[:, 3]) / 2
    psx = (p[:, 2] - p[:, 0]) / 2;  psy = (p[:, 3] - p[:, 1]) / 2
    tsx = (t[:, 2] - t[:, 0]) / 2;  tsy = (t[:, 3] - t[:, 1]) / 2
    w_sq = ((pcx-tcx)**2 + (pcy-tcy)**2 + (psx-tsx)**2 + (psy-tsy)**2)
    return 1.0 - torch.exp(-torch.sqrt(w_sq + 1e-9) / NWD_C)


class NWDBboxLoss(nn.Module):
    """BboxLoss with NWD replacing CIoU. DFL unchanged.
    Must be a module-level class so torch.save/pickle can serialise checkpoints.
    """
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum,
                fg_mask, imgsz, stride):
        weight     = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        stride_exp = stride.squeeze(-1).unsqueeze(0).expand(pred_bboxes.shape[0], -1)
        stride_fg  = stride_exp[fg_mask].unsqueeze(-1)
        nwd_val    = _nwd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], stride_fg)
        loss_iou   = (nwd_val.unsqueeze(-1) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]; target_ltrb[..., 1::2] /= imgsz[0]
            pred_d = pred_dist * stride
            pred_d[..., 0::2] /= imgsz[1]; pred_d[..., 1::2] /= imgsz[0]
            loss_dfl = (F.l1_loss(pred_d[fg_mask], target_ltrb[fg_mask],
                                  reduction="none").mean(-1, keepdim=True) * weight)
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl


def _patch_nwd() -> None:
    import ultralytics.utils.loss as loss_module
    loss_module.BboxLoss = NWDBboxLoss
    print("  [NWD] BboxLoss -> NWDBboxLoss (C=%.1f px, picklable)" % NWD_C)


# ── Helpers ────────────────────────────────────────────────────────────────────

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
        rows = list(csv.DictReader(csv_path.open()))
        return max((float(r.get("metrics/mAP50(B)", 0)) for r in rows), default=0.0)
    except Exception:
        return 0.0


COMMON_AUG = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    copy_paste=0.0,   # MUST be 0 — workers=0 on Windows makes copy_paste ~50s/batch
    mosaic=1.0,
    scale=0.5, perspective=0.0005, degrees=10.0,
    flipud=0.3, fliplr=0.5,
    mixup=0.05, erasing=0.1,
    close_mosaic=5,
)


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> None:
    for path, name in [(DATASET_YAML, "dataset"), (BASE_WEIGHTS, "baseline weights")]:
        if not path.exists():
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    keep_awake()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    # ── Phase 1: head-only warmup ──────────────────────────────────────────────
    p1_best = PHASE1_DIR / "weights" / "best.pt"
    p1_last = PHASE1_DIR / "weights" / "last.pt"

    if p1_best.exists() or p1_last.exists():
        p1_map = best_map50(PHASE1_DIR)
        print(f"Phase 1 weights found (mAP50={p1_map:.4f}) — skipping to Phase 2.")
    else:
        print("=" * 64)
        print("PHASE 1/2  head-only warmup  (freeze=20, 10 epochs, lr=5e-4)")
        print("  ~8 min/epoch x 10 = ~1.5 hours")
        print("=" * 64)
        model1 = YOLO(str(BASE_WEIGHTS))
        model1.train(
            data=str(DATASET_YAML),
            epochs=10,
            imgsz=640,
            batch=16,
            workers=0,
            device="0",
            freeze=20,             # freeze backbone + most of neck; train only last block + head

            optimizer="SGD",
            lr0=5e-4,              # higher LR fine — backbone frozen, head adapts quickly
            lrf=0.1,
            momentum=0.937,
            weight_decay=1e-4,
            warmup_epochs=2,
            cos_lr=True,

            cls=1.5,
            box=7.5,
            dfl=1.5,
            label_smoothing=0.1,

            **COMMON_AUG,

            cache="disk",
            patience=8,            # stop early if head stalls
            save_period=5,
            exist_ok=True,
            project=str(OUT_DIR),
            name="staged_phase1",
        )

    p1_weights = p1_best if p1_best.exists() else p1_last
    p1_map = best_map50(PHASE1_DIR)
    print(f"\nPhase 1 best mAP50: {p1_map:.4f}")

    # ── Phase 2: NWD full fine-tune ────────────────────────────────────────────
    p2_best = PHASE2_DIR / "weights" / "best.pt"
    p2_last = PHASE2_DIR / "weights" / "last.pt"

    # Count epochs already done in Phase 2 (resume support)
    p2_results = PHASE2_DIR / "results.csv"
    p2_epochs_done = 0
    if p2_results.exists():
        try:
            with open(p2_results) as f:
                p2_epochs_done = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass

    TOTAL_EPOCHS = 15
    remaining = TOTAL_EPOCHS - p2_epochs_done

    if remaining <= 0:
        print(f"Phase 2 already complete ({p2_epochs_done}/{TOTAL_EPOCHS} epochs).")
    else:
        _patch_nwd()

        # Resume from last.pt if available, else start from Phase 1 weights
        if p2_last.exists() and p2_epochs_done > 0:
            start_weights = p2_last
            warmup = 1      # short re-warmup after batch-size change
            print(f"\nResuming Phase 2 from epoch {p2_epochs_done} checkpoint (last.pt)")
        else:
            start_weights = p1_weights
            warmup = 2
            p2_epochs_done = 0

        print("=" * 64)
        print(f"PHASE 2/2  NWD full fine-tune  (freeze=0, {remaining} epochs, batch=12, lr=1e-4)")
        print(f"  ~17 min/epoch x {remaining} = ~{round(remaining*17/60,1)} hours")
        print("=" * 64)
        model2 = YOLO(str(start_weights))
        model2.train(
            data=str(DATASET_YAML),
            epochs=remaining,
            imgsz=640,
            batch=12,              # batch=12: ~6.6GB VRAM, ~17 min/epoch (vs 26 min at batch=8)
            workers=0,
            device="0",
            freeze=0,

            optimizer="SGD",
            lr0=1e-4,
            lrf=0.01,
            momentum=0.937,
            weight_decay=1e-4,
            warmup_epochs=warmup,
            cos_lr=True,

            cls=1.5,
            box=7.5,
            dfl=1.5,
            label_smoothing=0.1,

            **COMMON_AUG,

            cache="disk",
            patience=10,
            save_period=3,
            exist_ok=True,
            project=str(OUT_DIR),
            name="staged_phase2",
        )

    best_pt = p2_best if p2_best.exists() else p2_last
    CANDIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, CANDIDATE_OUT)

    p2_map = best_map50(PHASE2_DIR)
    baseline = 0.694
    status = "BEATS BASELINE" if p2_map > baseline else f"below baseline ({baseline})"
    print(f"\nPhase 2 best mAP50 : {p2_map:.4f}  ({status})")
    print(f"Saved to           : {CANDIDATE_OUT}")


if __name__ == "__main__":
    run()
