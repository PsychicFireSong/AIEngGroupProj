"""
eval_staged_comparison.py — Compare baseline vs Phase 1 vs Phase 2 (NWD).

Runs YOLO val on v7_lean_dataset for each checkpoint that exists,
prints per-class AP50 and a verdict on whether NWD helped or hurt.

Usage:
    python scripts/eval_staged_comparison.py
    python scripts/eval_staged_comparison.py --skip-baseline   # faster
    python scripts/eval_staged_comparison.py --phase2-only     # use cached baseline/p1, only run phase2
    python scripts/eval_staged_comparison.py --conf 0.25       # adjust conf threshold
"""
from __future__ import annotations

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── NWD classes — must be at module level so torch.load can unpickle Phase 2 ──
# Phase 2 checkpoints were saved while train_staged.py ran as __main__, so
# pickle stored NWDBboxLoss as __main__.NWDBboxLoss.  Defining it here (also
# in __main__) lets the unpickler find it when loading those checkpoints.

NWD_C = 12.8

try:
    from ultralytics.utils.loss import DFLoss
    from ultralytics.utils.tal  import bbox2dist as _bbox2dist

    def _nwd_loss(pred: torch.Tensor, target: torch.Tensor,
                  stride: torch.Tensor) -> torch.Tensor:
        p = pred * stride;  t = target * stride
        pcx = (p[:, 0] + p[:, 2]) / 2;  pcy = (p[:, 1] + p[:, 3]) / 2
        tcx = (t[:, 0] + t[:, 2]) / 2;  tcy = (t[:, 1] + t[:, 3]) / 2
        psx = (p[:, 2] - p[:, 0]) / 2;  psy = (p[:, 3] - p[:, 1]) / 2
        tsx = (t[:, 2] - t[:, 0]) / 2;  tsy = (t[:, 3] - t[:, 1]) / 2
        w_sq = (pcx-tcx)**2 + (pcy-tcy)**2 + (psx-tsx)**2 + (psy-tsy)**2
        return 1.0 - torch.exp(-torch.sqrt(w_sq + 1e-9) / NWD_C)

    class NWDBboxLoss(nn.Module):
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
                target_ltrb = _bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
                loss_dfl = self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                ) * weight
                loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = torch.tensor(0.0)
            return loss_iou, loss_dfl

except ImportError:
    pass  # ultralytics not yet imported; YOLO import inside eval_model will handle it

# ── Cached results from the completed baseline + phase1 eval run ───────────────
# These were verified on v7_lean_dataset val (3,855 images, conf=0.001).
# Used by --phase2-only to skip re-running those two evals.
CACHED_RESULTS = {
    "baseline": {
        "label": "baseline", "mAP50": 0.690, "mAP50_95": 0.413,
        "precision": 0.767, "recall": 0.623,
        "per_class": {
            "crack": 0.672, "spalling": 0.673, "corrosion": 0.575,
            "pothole": 0.787, "paint_degradation": 0.744,
        },
    },
    "phase1_best": {
        "label": "phase1_best", "mAP50": 0.683, "mAP50_95": 0.397,
        "precision": 0.742, "recall": 0.616,
        "per_class": {
            "crack": 0.665, "spalling": 0.677, "corrosion": 0.571,
            "pothole": 0.789, "paint_degradation": 0.714,
        },
    },
}

DATASET_YAML = Path(r"C:\Users\User\AIEngGroupProj\output\v7_lean_dataset\data.yaml")

CHECKPOINTS = {
    "baseline": Path(
        r"C:\Users\User\AIEngGroupProj_colab_outputs"
        r"\continued_hn_weak_finetune\runs\current\weights"
        r"\defect_detector_hn_weak_candidate.pt"
    ),
    "phase1_best": Path(
        r"C:\Users\User\AIEngGroupProj\output\staged_runs\staged_phase1\weights\best.pt"
    ),
    "phase2_best": Path(
        r"C:\Users\User\AIEngGroupProj\output\staged_runs\staged_phase2\weights\best.pt"
    ),
    "phase2_last": Path(
        r"C:\Users\User\AIEngGroupProj\output\staged_runs\staged_phase2\weights\last.pt"
    ),
}

CLASSES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]


def eval_model(label: str, weights: Path, conf: float) -> dict | None:
    from ultralytics import YOLO

    if not weights.exists():
        print(f"  [{label}] not found, skipping: {weights}")
        return None

    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"  Weights   : {weights}")
    print(f"{'='*60}")

    model = YOLO(str(weights))
    results = model.val(
        data=str(DATASET_YAML),
        imgsz=640,
        batch=8,
        workers=0,
        device="0",
        conf=conf,
        verbose=True,
        save=False,
        plots=False,
        project=None,
    )

    # per-class AP@0.5 (one value per class index)
    ap50_per_class: list[float] = list(results.box.ap50)

    # Pad/clip to 5 classes in case the model class list differs
    while len(ap50_per_class) < len(CLASSES):
        ap50_per_class.append(0.0)
    ap50_per_class = ap50_per_class[: len(CLASSES)]

    return {
        "label":      label,
        "mAP50":      float(results.box.map50),
        "mAP50_95":   float(results.box.map),
        "precision":  float(results.box.mp),
        "recall":     float(results.box.mr),
        "per_class":  dict(zip(CLASSES, ap50_per_class)),
    }


def fmt_delta(v: float, ref: float) -> str:
    d = v - ref
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:+.4f}"


def verdict(v: float, ref: float) -> str:
    d = v - ref
    if d > 0.010:
        return "BETTER"
    if d < -0.010:
        return "WORSE"
    return "~same"


def print_comparison(results: list[dict]) -> None:
    if not results:
        return

    baseline = next((r for r in results if r["label"] == "baseline"), results[0])

    # Overall table
    print("\n" + "=" * 76)
    print(f"  {'MODEL':<20}  {'mAP50':>8}  {'mAP50-95':>9}  {'Prec':>7}  {'Recall':>7}  vs baseline")
    print("-" * 76)
    for r in results:
        ref_map = baseline["mAP50"]
        delta   = fmt_delta(r["mAP50"], ref_map) if r["label"] != "baseline" else "    —  "
        print(
            f"  {r['label']:<20}  {r['mAP50']:>8.4f}  {r['mAP50_95']:>9.4f}"
            f"  {r['precision']:>7.4f}  {r['recall']:>7.4f}  {delta}"
        )

    # Per-class table
    print("\n" + "=" * 76)
    print(f"  Per-class AP@50 comparison")
    print("-" * 76)
    header = f"  {'CLASS':<20}" + "".join(f"  {r['label'][:12]:>13}" for r in results)
    if len(results) > 1:
        header += f"  {'vs_baseline':>12}"
    print(header)
    print("-" * 76)

    for cls in CLASSES:
        row = f"  {cls:<20}"
        vals = [r["per_class"].get(cls, 0.0) for r in results]
        for v in vals:
            row += f"  {v:>13.4f}"
        if len(results) > 1:
            last  = vals[-1]
            ref   = baseline["per_class"].get(cls, 0.0)
            d     = last - ref
            sign  = "+" if d >= 0 else ""
            verd  = verdict(last, ref)
            row  += f"  {sign}{d:+.4f} {verd}"
        print(row)

    # Verdict summary
    if len(results) > 1:
        last_r  = results[-1]
        ref_r   = baseline
        print("\n" + "=" * 76)
        print(f"  VERDICT — {last_r['label']} vs baseline")
        print("-" * 76)
        wins   = [cls for cls in CLASSES if last_r["per_class"].get(cls, 0) - ref_r["per_class"].get(cls, 0) > 0.010]
        losses = [cls for cls in CLASSES if last_r["per_class"].get(cls, 0) - ref_r["per_class"].get(cls, 0) < -0.010]
        same   = [cls for cls in CLASSES if cls not in wins and cls not in losses]
        print(f"  Classes improved  : {', '.join(wins)  if wins   else 'none'}")
        print(f"  Classes degraded  : {', '.join(losses) if losses else 'none'}")
        print(f"  Classes unchanged : {', '.join(same)  if same   else 'none'}")
        overall_d = last_r["mAP50"] - ref_r["mAP50"]
        print(f"\n  Overall mAP50 delta vs baseline: {'+' if overall_d>=0 else ''}{overall_d:+.4f}")
        if overall_d > 0.010:
            print("  -> NWD fine-tune HELPED overall.")
        elif overall_d < -0.010:
            print("  -> NWD fine-tune HURT overall — revert to baseline.")
        else:
            print("  -> NWD fine-tune had negligible overall effect.")

    print("=" * 76)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip baseline eval (save ~10 min)")
    ap.add_argument("--phase2-only", action="store_true",
                    help="Use cached baseline/phase1 results; only run phase2 eval (~10 min)")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="Confidence threshold for val (default 0.001 = standard YOLO eval)")
    args = ap.parse_args()

    if not DATASET_YAML.exists():
        print(f"ERROR: dataset yaml not found: {DATASET_YAML}")
        sys.exit(1)

    # Decide which phase2 checkpoint to use
    p2 = CHECKPOINTS["phase2_best"] if CHECKPOINTS["phase2_best"].exists() else CHECKPOINTS["phase2_last"]
    p2_label = "phase2_best" if CHECKPOINTS["phase2_best"].exists() else "phase2_last"

    if args.phase2_only:
        print(f"\nUsing cached baseline + phase1 results. Running phase2 eval only.")
        print(f"Confidence threshold: {args.conf}  (0.001 = standard mAP eval)\n")
        collected = [CACHED_RESULTS["baseline"], CACHED_RESULTS["phase1_best"]]
        r = eval_model(p2_label, p2, args.conf)
        if r:
            collected.append(r)
        print_comparison(collected)
        return

    eval_order = []
    if not args.skip_baseline:
        eval_order.append(("baseline",    CHECKPOINTS["baseline"]))
    eval_order.append(("phase1_best", CHECKPOINTS["phase1_best"]))
    eval_order.append((p2_label,      p2))

    print(f"\nEvaluating {len(eval_order)} checkpoint(s) on v7_lean_dataset val (3,855 images)")
    print(f"Confidence threshold: {args.conf}  (0.001 = standard mAP eval)")
    print(f"Each eval takes ~8-12 min on RTX 3070 Ti\n")

    collected = []
    for label, weights in eval_order:
        r = eval_model(label, weights, args.conf)
        if r:
            collected.append(r)

    print_comparison(collected)

    if not CHECKPOINTS["phase2_best"].exists():
        print("\nNOTE: Phase 2 is still training. Re-run this script after it completes")
        print("to compare against phase2/best.pt (current comparison uses last.pt).")


if __name__ == "__main__":
    main()
