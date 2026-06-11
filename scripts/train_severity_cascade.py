"""
train_severity_cascade.py — Binary cascade approach for severity classification.

Why cascade instead of single 3-class model:
  All single-model attempts plateau at ~64-66% regardless of architecture or
  augmentation. Root cause: "moderate" class is too heterogeneous (cracks +
  spalling + corrosion + peeling paint + water damage), causing confusion with
  both "minor" (surface textures overlap) and "critical" (structural severity).

  Splitting into two easier binary problems:
    Model 1 — "is_critical":  critical vs (minor + moderate)
      critical = exposed rebar — visually very distinct (metal bars, structural failure)
      Expected: 88-95% accuracy
    Model 2 — "minor_or_moderate": minor vs moderate (only on non-critical crops)
      Focused binary, no critical confusion noise
      Expected: 75-82% accuracy

  Combined theoretical accuracy on balanced 3-class:
    P(correct) = (P1_critical_acc + P1_noncrit_acc × P2_acc × 2) / 3
    e.g. (92 + 92 × 78 × 2) / 3 ≈ 78-83%  → meaningful improvement over 65.8%

Inference cascade (handled in inference_api.py or via predict_cascade() below):
  1. Run is_critical model → if "critical", return immediately
  2. Else run minor_or_moderate model → return "minor" or "moderate"

Saved weights:
  weights/severity_critical_cls.pt      (binary: critical vs not_critical)
  weights/severity_minor_moderate_cls.pt (binary: minor vs moderate)
"""
from __future__ import annotations

import ctypes
import csv
import shutil
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
SEV_DS   = Path(
    r"C:\Users\User\AIEngGroupProj\output"
    r"\continued_hn_weak\current\datasets\severity_dataset"
)
OUT_DS   = Path(r"C:\Users\User\AIEngGroupProj\output\severity_cascade_datasets")
OUT_RUNS = Path(r"C:\Users\User\AIEngGroupProj\output\severity_runs")
WEIGHTS  = Path(r"C:\Users\User\AIEngGroupProj\weights")

CRITICAL_DS  = OUT_DS / "is_critical"        # binary: critical vs not_critical
MINMOD_DS    = OUT_DS / "minor_or_moderate"  # binary: minor vs moderate

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def keep_awake() -> None:
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def best_top1(run_dir: Path) -> float:
    p = run_dir / "results.csv"
    if not p.exists():
        return 0.0
    try:
        rows = list(csv.DictReader(open(p)))
        return max((float(r.get("metrics/accuracy_top1", 0)) for r in rows), default=0.0)
    except Exception:
        return 0.0


# ── Dataset preparation ────────────────────────────────────────────────────────

def prepare_datasets() -> None:
    """Build two binary datasets from severity_dataset folder structure."""
    print("Preparing binary cascade datasets …")

    for split in ["train", "val", "test"]:
        src_split = SEV_DS / split
        if not src_split.exists():
            continue

        # Model 1: is_critical → classes: critical / not_critical
        for cls_dir in src_split.iterdir():
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.lower()
            binary_cls = "critical" if cls_name == "critical" else "not_critical"
            dst = CRITICAL_DS / split / binary_cls
            dst.mkdir(parents=True, exist_ok=True)
            for img in cls_dir.glob("*"):
                if img.suffix.lower() in IMAGE_EXTS:
                    target = dst / img.name
                    if not target.exists():
                        shutil.copy2(img, target)

        # Model 2: minor_or_moderate → classes: minor / moderate (skip critical)
        for cls_dir in src_split.iterdir():
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.lower()
            if cls_name == "critical":
                continue   # exclude critical images from this binary model
            dst = MINMOD_DS / split / cls_name
            dst.mkdir(parents=True, exist_ok=True)
            for img in cls_dir.glob("*"):
                if img.suffix.lower() in IMAGE_EXTS:
                    target = dst / img.name
                    if not target.exists():
                        shutil.copy2(img, target)

    # Report sizes
    for ds_name, ds_path in [("is_critical", CRITICAL_DS), ("minor_or_moderate", MINMOD_DS)]:
        print(f"\n  {ds_name}/")
        for split in ["train", "val", "test"]:
            sp = ds_path / split
            if not sp.exists():
                continue
            for cls in sorted(sp.iterdir()):
                if cls.is_dir():
                    n = len([f for f in cls.glob("*") if f.suffix.lower() in IMAGE_EXTS])
                    print(f"    {split}/{cls.name}: {n}")

    print("\nDatasets ready.")


# ── Training ───────────────────────────────────────────────────────────────────

TRAIN_KWARGS = dict(
    imgsz=224,
    batch=32,
    workers=0,
    device="0",
    optimizer="AdamW",
    lrf=0.01,
    weight_decay=5e-4,
    cos_lr=True,
    label_smoothing=0.05,   # lower smoothing for binary (less ambiguity)
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.3,
    fliplr=0.5,
    flipud=0.2,
    scale=0.15,
    erasing=0.1,
    save_period=10,
    exist_ok=True,
    project=str(OUT_RUNS),
)


def train_binary(name: str, dataset: Path, lr0: float, epochs: int,
                 warmup: int, patience: int) -> float:
    from ultralytics import YOLO
    run_dir = OUT_RUNS / name
    best_pt = run_dir / "weights" / "best.pt"
    results = run_dir / "results.csv"

    done = 0
    if results.exists():
        try:
            with open(results) as f:
                done = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass

    remaining = epochs - done
    if remaining <= 0:
        t = best_top1(run_dir)
        print(f"  {name}: already complete (best={t:.4f}). Skipping.")
        return t

    start = str((run_dir / "weights" / "last.pt") if (run_dir / "weights" / "last.pt").exists() and done > 0
                else "yolo11n-cls.pt")

    print(f"\n{'='*60}")
    print(f"  {name}  lr={lr0:.0e}  {remaining} ep  warmup={warmup}")
    print(f"{'='*60}")
    model = YOLO(start)
    model.train(
        data=str(dataset),
        epochs=remaining,
        lr0=lr0,
        warmup_epochs=warmup,
        patience=patience,
        name=name,
        **TRAIN_KWARGS,
    )
    t = best_top1(run_dir)
    print(f"\n  {name} best top1: {t:.4f} ({t*100:.1f}%)")
    return t


# ── Cascade evaluation ─────────────────────────────────────────────────────────

def eval_cascade() -> None:
    """Evaluate cascade on test split and compute effective 3-class accuracy."""
    from ultralytics import YOLO
    import pathlib

    crit_pt  = OUT_RUNS / "sev_cascade_critical" / "weights" / "best.pt"
    minmod_pt = OUT_RUNS / "sev_cascade_minmod"  / "weights" / "best.pt"

    if not crit_pt.exists() or not minmod_pt.exists():
        print("Cascade weights not found — skipping eval.")
        return

    m_crit   = YOLO(str(crit_pt))
    m_minmod = YOLO(str(minmod_pt))

    test_dir = SEV_DS / "test"
    correct = total = 0

    for true_cls_dir in sorted(test_dir.iterdir()):
        if not true_cls_dir.is_dir():
            continue
        true_cls = true_cls_dir.name.lower()
        cls_correct = cls_total = 0

        for img in true_cls_dir.glob("*"):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            total += 1
            cls_total += 1

            # Step 1: is it critical?
            # threshold=0.40 (not 0.50): corrects imbalance bias from 1:2 training ratio.
            # At 0.50, borderline critical images (40-49% conf) get routed to Model 2,
            # which has never seen critical images → always wrong. At 0.40, those
            # borderline cases are correctly captured by the cascade.
            CRIT_THRESHOLD = 0.40
            r1 = m_crit.predict(str(img), verbose=False)[0]
            names1 = r1.names
            crit_idx = next((i for i, n in names1.items() if n.lower() == "critical"), 0)
            crit_conf = float(r1.probs.data.cpu().numpy()[crit_idx])
            pred_critical = crit_conf >= CRIT_THRESHOLD

            if pred_critical:
                pred = "critical"
            else:
                # Step 2: minor or moderate?
                r2 = m_minmod.predict(str(img), verbose=False)[0]
                pred = r2.names[int(r2.probs.top1)].lower()

            if pred == true_cls:
                correct += 1
                cls_correct += 1

        if cls_total:
            print(f"  {true_cls:15s}: {cls_correct}/{cls_total} = {cls_correct/cls_total*100:.1f}%")

    overall = correct / total if total else 0
    print(f"\nCascade test accuracy : {overall:.4f} ({overall*100:.1f}%)")
    print(f"Baseline (single 3-cls): 65.8%")
    if overall > 0.658:
        print(f"IMPROVEMENT: +{(overall-0.658)*100:.1f} pts over baseline")
    else:
        print("No improvement over baseline.")
    return overall


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> None:
    if not SEV_DS.exists():
        print(f"ERROR: severity dataset not found: {SEV_DS}")
        sys.exit(1)

    keep_awake()

    # 1. Prepare binary datasets
    prepare_datasets()

    # 2. Train Model 1: is_critical (binary, yolo11n-cls, 60 epochs)
    #    critical images are very distinctive → high LR fine, converges fast
    acc_crit = train_binary(
        name="sev_cascade_critical",
        dataset=CRITICAL_DS,
        lr0=5e-4,
        epochs=60,
        warmup=3,
        patience=15,
    )

    # 3. Train Model 2: minor_or_moderate (binary, yolo11n-cls, 80 epochs)
    #    harder problem → lower LR, more epochs
    acc_minmod = train_binary(
        name="sev_cascade_minmod",
        dataset=MINMOD_DS,
        lr0=3e-4,
        epochs=80,
        warmup=3,
        patience=20,
    )

    # 4. Save to canonical weights location
    for src_name, out_name in [
        ("sev_cascade_critical", "severity_critical_cls.pt"),
        ("sev_cascade_minmod",   "severity_minor_moderate_cls.pt"),
    ]:
        src = OUT_RUNS / src_name / "weights" / "best.pt"
        if src.exists():
            dst = WEIGHTS / out_name
            WEIGHTS.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Saved: {dst}")

    # 5. Evaluate cascade on test set
    print("\n" + "="*60)
    print("Evaluating cascade on test split …")
    eval_cascade()

    # Print summary
    print("\n" + "="*60)
    print(f"  is_critical model acc      : {acc_crit:.4f} ({acc_crit*100:.1f}%)")
    print(f"  minor_or_moderate model acc: {acc_minmod:.4f} ({acc_minmod*100:.1f}%)")
    print(f"  Baseline (3-class single)  : 0.6580 (65.8%)")


if __name__ == "__main__":
    run()
