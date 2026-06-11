"""
eval_v4_perclass.py
Run after v4 training completes. Validates best.pt on the val split and
prints per-class AP50 / precision / recall vs the hn_weak baseline.
"""
from __future__ import annotations
from pathlib import Path

BEST_PT   = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round6\runs\patch6\weights\best.pt")
BASELINE  = Path(r"C:\Users\User\AIEngGroupProj_colab_outputs\continued_hn_weak_finetune\runs\current\weights\defect_detector_hn_weak_candidate.pt")
YAML      = Path(r"C:\Users\User\AIEngGroupProj\output\patch_round6\dataset\data.yaml")
CLASSES   = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]

# hn_weak baseline per-class AP50 (from colab run — used for comparison)
BASELINE_AP50 = {
    "crack":             0.752,
    "spalling":          0.779,
    "corrosion":         0.625,
    "pothole":           0.741,
    "paint_degradation": 0.572,
}


def run_val(pt: Path, label: str) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(pt))
    results = model.val(
        data=str(YAML),
        imgsz=640,
        batch=16,
        workers=0,
        device="0",
        verbose=False,
        plots=False,
    )
    ap50   = results.box.ap50          # per-class AP50 list
    prec   = results.box.p             # per-class precision
    rec    = results.box.r             # per-class recall
    map50  = float(results.box.map50)
    names  = results.names             # {0: 'crack', ...}
    return {
        "label":  label,
        "map50":  map50,
        "ap50":   {names[i]: float(ap50[i]) for i in range(len(ap50))},
        "prec":   {names[i]: float(prec[i]) for i in range(len(prec))},
        "rec":    {names[i]: float(rec[i])  for i in range(len(rec))},
    }


def print_comparison(v4: dict) -> None:
    print(f"\n{'='*68}")
    print(f"  v4 fine-tune  mAP50 = {v4['map50']:.4f}   "
          f"(baseline hn_weak = 0.6940)")
    print(f"{'='*68}")
    header = f"{'Class':<22} {'Baseline AP50':>14} {'v4 AP50':>9} {'Delta':>8}  {'v4 Prec':>8} {'v4 Rec':>8}"
    print(header)
    print("-" * 68)

    beats, total = 0, 0
    for cls in CLASSES:
        base_ap = BASELINE_AP50.get(cls, 0.0)
        v4_ap   = v4["ap50"].get(cls, 0.0)
        delta   = v4_ap - base_ap
        prec    = v4["prec"].get(cls, 0.0)
        rec     = v4["rec"].get(cls, 0.0)
        flag    = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "~")
        print(f"{cls:<22} {base_ap:>14.3f} {v4_ap:>9.3f} {delta:>+8.3f}{flag} "
              f"{prec:>8.3f} {rec:>8.3f}")
        if delta > 0:
            beats += 1
        total += 1

    print("-" * 68)
    print(f"  {beats}/{total} classes improved over baseline")
    verdict = "BEATS BASELINE" if v4["map50"] > 0.694 else "below baseline"
    print(f"  Overall verdict: {verdict}")
    print(f"{'='*68}\n")

    # Recommend next action
    if v4["map50"] > 0.694:
        print("ACTION: Promote v4 best.pt — deploy to API and run Stage 2.")
    else:
        weak = [c for c in CLASSES if v4["ap50"].get(c, 0) < BASELINE_AP50.get(c, 0)]
        if weak:
            print(f"ACTION: v4 is below baseline. Classes that regressed: {weak}")
            print("  Recommendation: run v5 with patch_ratio=1x and lr0=3e-4")
        else:
            print("ACTION: mAP50 below baseline but all classes improved individually.")
            print("  Check if distribution shift hurt overall metric vs per-class gains.")


if __name__ == "__main__":
    print("Running validation on v4 best.pt ...")
    v4 = run_val(BEST_PT, "v4")
    print_comparison(v4)
