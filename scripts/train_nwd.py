"""
train_nwd.py  —  YOLO11m retrain with NWD box loss + improved augmentation.

Why NWD over CIoU:
  CIoU gradient vanishes for thin/small boxes (crack: ~10×3px) when the center
  offset is larger than half the box width — IoU → 0, gradient → 0.
  NWD treats boxes as 2D Gaussians and measures Wasserstein distance, so
  a crack that's "almost right but shifted" still produces a strong gradient.
  Literature: +2-4 mAP on crack/corrosion detection (PMC11175173, MDPI 2024).

Additional changes vs baseline (all proven safe, no distribution shift):
  label_smoothing=0.1  — reduces overconfident suppression during CLS training
  copy_paste=0.3        — pastes instances from other images; safer than oversampling
  close_mosaic=10       — stops mosaic for last 10 epochs → head converges cleanly

Dataset: v7_lean_dataset (ORIGINAL, 20,166 train; no oversampling, no hard negatives)
  Reason: v7_fixed oversampling + progressive unfreeze caused catastrophic forgetting.
  Clean data + small LR is the safe path.

Training: single phase, full model (freeze=0), lr0=1e-4 → prevents forgetting without
  progressive unfreeze overhead.
"""
from __future__ import annotations

import ctypes
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_YAML = Path(r"C:\Users\User\AIEngGroupProj\output\v7_lean_dataset\data.yaml")
BASE_WEIGHTS = Path(
    r"C:\Users\User\AIEngGroupProj_colab_outputs"
    r"\continued_hn_weak_finetune\runs\current\weights"
    r"\defect_detector_hn_weak_candidate.pt"
)
OUT_DIR      = Path(r"C:\Users\User\AIEngGroupProj\output\nwd_runs")
CANDIDATE_OUT = Path(
    r"C:\Users\User\AIEngGroupProj_colab_outputs"
    r"\continued_hn_weak_finetune\runs\current\weights"
    r"\defect_detector_nwd_candidate.pt"
)

NWD_C = 12.8   # Wasserstein normalization constant (calibrated for pixel coords)
               # Original paper value for small-object detection


# ── NWD loss implementation ────────────────────────────────────────────────────

def _nwd_loss(pred: torch.Tensor, target: torch.Tensor, stride: torch.Tensor) -> torch.Tensor:
    """
    Normalized Wasserstein Distance loss for bounding boxes.

    Boxes are in stride-normalized grid coords. We convert to pixel coords
    (multiply by stride) so NWD_C is scale-consistent across all stride levels.

    Reference: Wang et al., arxiv 2110.13389 (NWD original).
    Args:
        pred:   (N, 4) xyxy grid-coord boxes
        target: (N, 4) xyxy grid-coord boxes
        stride: (N, 1) stride value for each anchor (8/16/32)
    Returns:
        (N,) per-box NWD loss, range [0, 1]
    """
    # Convert to pixel coords so NWD_C is in pixel units (scale-agnostic per stride)
    p = pred   * stride   # (N, 4) pixel coords
    t = target * stride

    # Centers
    pcx = (p[:, 0] + p[:, 2]) / 2;  pcy = (p[:, 1] + p[:, 3]) / 2
    tcx = (t[:, 0] + t[:, 2]) / 2;  tcy = (t[:, 1] + t[:, 3]) / 2

    # Half-dimensions (Gaussian "sigma")
    psx = (p[:, 2] - p[:, 0]) / 2;  psy = (p[:, 3] - p[:, 1]) / 2
    tsx = (t[:, 2] - t[:, 0]) / 2;  tsy = (t[:, 3] - t[:, 1]) / 2

    # 2-Wasserstein distance for diagonal Gaussians:
    # W² = ||μ_i - μ_j||² + ||σ_i - σ_j||²_F
    w_sq = (pcx - tcx).pow(2) + (pcy - tcy).pow(2) + \
           (psx - tsx).pow(2) + (psy - tsy).pow(2)
    wasser = torch.sqrt(w_sq + 1e-9)

    return 1.0 - torch.exp(-wasser / NWD_C)


# ── Monkey-patch BboxLoss ─────────────────────────────────────────────────────

def _patch_bbox_loss() -> None:
    """
    Replace BboxLoss.forward to use NWD instead of CIoU for the box regression
    loss. DFL loss is kept unchanged.

    We patch at module level so all subsequent YOLO training calls see NWD.
    """
    import ultralytics.utils.loss as loss_module
    from ultralytics.utils.loss import DFLoss, BboxLoss
    from ultralytics.utils.tal import bbox2dist

    OrigBboxLoss = BboxLoss  # keep reference

    class NWDBboxLoss(nn.Module):
        """BboxLoss with NWD replacing CIoU. DFL unchanged."""

        def __init__(self, reg_max: int = 16):
            super().__init__()
            self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

        def forward(
            self,
            pred_dist:          torch.Tensor,
            pred_bboxes:        torch.Tensor,
            anchor_points:      torch.Tensor,
            target_bboxes:      torch.Tensor,
            target_scores:      torch.Tensor,
            target_scores_sum:  torch.Tensor,
            fg_mask:            torch.Tensor,
            imgsz:              torch.Tensor,
            stride:             torch.Tensor,
        ):
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

            # NWD replaces CIoU here
            # stride: (N_anchors, 1), fg_mask: (B, N_anchors) → broadcast to (B, N_anchors)
            stride_exp = stride.squeeze(-1).unsqueeze(0).expand(pred_bboxes.shape[0], -1)
            stride_fg  = stride_exp[fg_mask].unsqueeze(-1)   # (N_fg, 1)
            nwd_val    = _nwd_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], stride_fg)
            loss_iou  = (nwd_val.unsqueeze(-1) * weight).sum() / target_scores_sum

            # DFL unchanged
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
                target_ltrb[..., 0::2] /= imgsz[1]
                target_ltrb[..., 1::2] /= imgsz[0]
                pred_d = pred_dist * stride
                pred_d[..., 0::2] /= imgsz[1]
                pred_d[..., 1::2] /= imgsz[0]
                loss_dfl = (
                    F.l1_loss(pred_d[fg_mask], target_ltrb[fg_mask], reduction="none")
                    .mean(-1, keepdim=True) * weight
                )
                loss_dfl = loss_dfl.sum() / target_scores_sum

            return loss_iou, loss_dfl

    # Patch the module-level class so v8DetectionLoss.__init__ picks it up
    loss_module.BboxLoss = NWDBboxLoss
    print("  [NWD patch] BboxLoss.forward → NWD loss  (C=%.1f px)" % NWD_C)


# ── Helpers ───────────────────────────────────────────────────────────────────

def keep_awake() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("  [sleep-guard] Windows sleep inhibited")
    except Exception:
        pass


TRAIN_AUG = dict(
    hsv_h=0.015,       # mild hue shift (consistent across surface types)
    hsv_s=0.7,
    hsv_v=0.4,
    copy_paste=0.0,    # MUST be 0 with workers=0 on Windows — otherwise
                       # each batch triggers synchronous multi-image disk reads
                       # (~50s/batch instead of ~0.6s/batch)
    mosaic=1.0,
    scale=0.5,
    perspective=0.0005,
    degrees=10.0,
    flipud=0.3,
    fliplr=0.5,
    mixup=0.05,
    erasing=0.1,
    close_mosaic=10,   # stop mosaic for last 10 epochs → cleaner head convergence
)


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    for path, name in [(DATASET_YAML, "dataset"), (BASE_WEIGHTS, "baseline weights")]:
        if not path.exists():
            print(f"ERROR: {name} not found: {path}")
            sys.exit(1)

    keep_awake()

    # Apply NWD patch BEFORE importing YOLO (so the patched class is used)
    _patch_bbox_loss()

    from ultralytics import YOLO

    run_dir = OUT_DIR / "nwd_v1"

    if (run_dir / "weights" / "best.pt").exists():
        print(f"Checkpoint exists at {run_dir / 'weights' / 'best.pt'} — skipping.")
    else:
        print("=" * 64)
        print("NWD retrain  (freeze=0, 30 epochs, lr=1e-4, clean dataset)")
        print("=" * 64)
        model = YOLO(str(BASE_WEIGHTS))
        model.train(
            data=str(DATASET_YAML),
            epochs=30,
            imgsz=640,
            batch=16,              # ~8-9 min/epoch; reduce to 8 if OOM
            workers=0,
            device="0",
            freeze=0,              # single-phase full model — safe with lr=1e-4

            optimizer="SGD",
            lr0=1e-4,              # small: prevents catastrophic forgetting
            lrf=0.01,
            momentum=0.937,
            weight_decay=1e-4,
            warmup_epochs=3,
            cos_lr=True,

            cls=1.5,               # standard cls weight
            box=7.5,               # standard box weight
            dfl=1.5,

            label_smoothing=0.1,   # reduces overconfident cls suppression
            **TRAIN_AUG,

            cache="disk",
            patience=15,
            save_period=5,
            exist_ok=True,
            project=str(OUT_DIR),
            name="nwd_v1",
        )

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"

    CANDIDATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, CANDIDATE_OUT)

    # Report vs baseline
    import csv
    results_csv = run_dir / "results.csv"
    map50 = 0.0
    if results_csv.exists():
        rows = list(csv.DictReader(results_csv.open()))
        map50 = max(float(r.get("metrics/mAP50(B)", 0)) for r in rows) if rows else 0.0

    baseline = 0.694
    status = "BEATS BASELINE" if map50 > baseline else f"below baseline ({baseline})"
    print(f"\nBest mAP50 : {map50:.4f}  ({status})")
    print(f"Saved to   : {CANDIDATE_OUT}")


if __name__ == "__main__":
    run()
