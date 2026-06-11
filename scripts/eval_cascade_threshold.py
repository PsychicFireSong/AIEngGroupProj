"""
eval_cascade_threshold.py — Test different confidence thresholds for cascade routing.

The current cascade routes to "critical" whenever Model 1 predicts critical (any confidence).
A higher threshold (e.g., 0.8) means: only return "critical" if Model 1 is 80%+ confident.
Otherwise, pass to Model 2 (minor_or_moderate).

This trades critical recall for reduced false positives in critical/moderate routing.
"""
from __future__ import annotations

from pathlib import Path

SEV_DS    = Path(
    r"C:\Users\User\AIEngGroupProj\output"
    r"\continued_hn_weak\current\datasets\severity_dataset"
)
OUT_RUNS  = Path(r"C:\Users\User\AIEngGroupProj\output\severity_runs")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CRIT_PT   = OUT_RUNS / "sev_cascade_critical" / "weights" / "best.pt"
MINMOD_PT = OUT_RUNS / "sev_cascade_minmod"   / "weights" / "best.pt"


def eval_at_threshold(m_crit, m_minmod, threshold: float, test_dir: Path) -> dict:
    correct = 0
    total = 0
    per_class = {}

    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        true_cls = cls_dir.name.lower()
        cc = ct = 0

        for img in cls_dir.glob("*"):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            total += 1
            ct += 1

            r1 = m_crit.predict(str(img), verbose=False)[0]
            probs1 = r1.probs.data.cpu().numpy()
            names1 = r1.names
            crit_idx = next((i for i, n in names1.items() if n.lower() == "critical"), 0)
            crit_conf = float(probs1[crit_idx])

            if crit_conf >= threshold:
                pred = "critical"
            else:
                r2 = m_minmod.predict(str(img), verbose=False)[0]
                pred = r2.names[int(r2.probs.top1)].lower()

            if pred == true_cls:
                correct += 1
                cc += 1

        per_class[true_cls] = (cc, ct)

    return {"correct": correct, "total": total, "per_class": per_class}


def run() -> None:
    from ultralytics import YOLO

    if not CRIT_PT.exists() or not MINMOD_PT.exists():
        print("ERROR: cascade weights not found. Run train_severity_cascade.py first.")
        return

    print("Loading models...")
    m_crit   = YOLO(str(CRIT_PT))
    m_minmod = YOLO(str(MINMOD_PT))
    test_dir = SEV_DS / "test"

    print(f"\nThreshold sweep on {test_dir}")
    print(f"{'Threshold':>10} {'Overall':>10} {'critical':>10} {'minor':>10} {'moderate':>10}")
    print("-" * 55)

    best_acc = 0
    best_thr = 0.5
    for thr in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        r = eval_at_threshold(m_crit, m_minmod, thr, test_dir)
        acc = r["correct"] / r["total"]
        pc = r["per_class"]
        line = f"{thr:>10.2f} {acc:>10.4f}"
        for cls in ["critical", "minor", "moderate"]:
            cc, ct = pc.get(cls, (0, 0))
            line += f" {cc/ct if ct else 0:>10.4f}"
        print(line)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    print(f"\nBest threshold: {best_thr} → {best_acc:.4f} ({best_acc*100:.1f}%)")
    print(f"Baseline 3-class: ~65.8%")


if __name__ == "__main__":
    run()
