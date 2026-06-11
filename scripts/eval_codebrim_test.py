"""Threshold sweep on test_codebrim set."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from ultralytics import YOLO

OUT_RUNS  = Path(r"C:\Users\User\AIEngGroupProj\output\severity_runs")
CRIT_PT   = OUT_RUNS / "sev_cascade_critical" / "weights" / "best.pt"
MINMOD_PT = OUT_RUNS / "sev_cascade_minmod"   / "weights" / "best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TEST_CB = Path(
    r"C:\Users\User\AIEngGroupProj\output"
    r"\continued_hn_weak\current\datasets\severity_dataset\test_codebrim"
)
TEST_ORIG = Path(
    r"C:\Users\User\AIEngGroupProj\output"
    r"\continued_hn_weak\current\datasets\severity_dataset\test"
)


def cascade_sweep(m_crit, m_minmod, test_dir, thresholds):
    print(f"\nSweep: {test_dir.name}")
    print(f"{'Thr':>6}  {'Overall':>8}  {'critical':>9}  {'minor':>8}  {'moderate':>9}")
    print("-" * 50)
    best_acc, best_thr = 0, 0.5
    for thr in thresholds:
        correct = total = 0
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
                crit_idx = next(
                    (i for i, n in r1.names.items() if n.lower() == "critical"), 0
                )
                crit_conf = float(probs1[crit_idx])
                if crit_conf >= thr:
                    pred = "critical"
                else:
                    r2 = m_minmod.predict(str(img), verbose=False)[0]
                    pred = r2.names[int(r2.probs.top1)].lower()
                if pred == true_cls:
                    correct += 1
                    cc += 1
            per_class[true_cls] = (cc, ct)
        acc = correct / total
        row = f"{thr:>6.2f}  {acc:>8.4f}"
        for cls in ["critical", "minor", "moderate"]:
            cc, ct = per_class.get(cls, (0, 0))
            row += f"  {cc/ct if ct else 0:>8.4f}"
        print(row)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    print(f"Best: thr={best_thr} -> {best_acc:.4f} ({best_acc*100:.1f}%)")
    return best_thr, best_acc


def main():
    print("Loading models...")
    m_crit   = YOLO(str(CRIT_PT))
    m_minmod = YOLO(str(MINMOD_PT))

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]

    cascade_sweep(m_crit, m_minmod, TEST_ORIG, thresholds)
    cascade_sweep(m_crit, m_minmod, TEST_CB,   thresholds)


if __name__ == "__main__":
    main()
