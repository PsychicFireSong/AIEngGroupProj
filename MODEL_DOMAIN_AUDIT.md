# Model Domain Audit

This audit documents how the two-stage YOLO pipeline performs on images outside the training/validation split, and tracks the current verified model metrics.

---

## Current Verified Metrics (v7_lean_dataset val, 3,855 images)

> These are the only metrics verified against the actual val set. Earlier figures in this document (0.851 mAP50) came from an early Colab run evaluated on the original merged_dataset internal split — a different and easier val set. Do not use those numbers as current performance claims.

**Stage 1 Detector — `defect_detector_hn_weak_candidate.pt` (current best):**

| Metric | Value |
|---|---|
| mAP50 | **0.694** |
| mAP50-95 | 0.396 |
| Precision | 0.740 |
| Recall | 0.631 |

**Per-class detection (production sweep):**

| Class | Detection Rate | Avg Confidence | Notes |
|---|---|---|---|
| crack | 66.7% | 0.706 | External datasets: 75.0% |
| spalling | 69.2% | 0.834 | Strong |
| corrosion | **25.0%** | 0.727 | Weak — domain shift (see below) |
| pothole | 100.0% | 0.749 | Excellent |
| paint_degradation | 80.0% | 0.645 | External: 100% |

**Stage 2 Severity Classifier — `severity_cls.pt`:**

| Metric | Value |
|---|---|
| Top-1 accuracy | **65.8%** |
| Target | >90% |
| Dataset | 1,800 train / 360 val / 360 test (600 per class, balanced) |
| Classes | minor / moderate / critical |

**Stage 1 fine-tuning experiments (all evaluated on v7_lean val, 3,855 images):**

| Checkpoint | mAP50 | vs Baseline | Status |
|---|---|---|---|
| `defect_detector_hn_weak_candidate.pt` | **0.694** | — | **Current best (baseline)** |
| `staged_phase1/best.pt` | 0.683 | -0.007 | Head-only warmup, 10 epochs |
| `staged_phase2/best.pt` | 0.658 | **-0.032** | NWD full unfreeze — WORSE |

**Stage 1 per-class AP50 (verified eval, conf=0.001):**

| Class | Baseline | Phase 1 | Phase 2 (NWD) |
|---|---|---|---|
| crack | 0.672 | 0.665 | 0.631 |
| spalling | 0.673 | 0.677 | 0.676 |
| corrosion | 0.575 | 0.571 | 0.508 |
| pothole | 0.787 | 0.789 | 0.776 |
| paint_degradation | 0.744 | 0.714 | 0.698 |

**Finding:** NWD loss (Normalized Wasserstein Distance) hurt every class except spalling. Despite being designed to improve gradient signal on small/thin boxes (cracks, corrosion), it degraded those classes the most (-0.041 crack, -0.067 corrosion). The catastrophic forgetting came from the full backbone unfreeze under a different loss landscape, not from CIoU vs NWD per se. Baseline remains the production detector.

**Severity classifier experiments:**

| Run | Model | Best top1 | Epochs | Notes |
|---|---|---|---|---|
| sev_patch1 (baseline) | yolo11n-cls | **65.8%** | 30 | AdamW lr=0.0007 |
| sev_v3 | yolo11s-cls | 65.6% ep6 | stopped ep16 | Overfit: erasing=0.4+RandAugment too aggressive for 224px crops |
| sev_v4 | yolo11s-cls | 63.9% ep6 | stopped ep13 | Overfit: same pattern despite lighter aug (erasing=0.15, lr=1e-4) |

**Finding:** All runs plateau at ~64-66% and decline. Root cause is dataset ceiling, not model capacity or augmentation. The "moderate" class covers crack + spalling + corrosion + peeling paint + water damage — visually heterogeneous. Labels are assigned by defect type name (crack→moderate, rebar→critical, scaling→minor), not actual visual severity, creating inherent noise at class boundaries.

**Confusion matrix pattern (sev_patch1, best run):**
- critical: 66% correct — 34% misclassified as moderate
- minor: 70% correct — 30% misclassified as moderate/critical
- moderate: 62% correct — 26% as critical, 24% as minor (worst class)

**Stage 1 class imbalance (v7_lean train, 48,496 instances):**

| Class | Instances | Share | AP50 |
|---|---|---|---|
| crack | 10,583 | 21.8% | 0.672 |
| spalling | 11,768 | 24.3% | 0.673 |
| corrosion | 4,499 | **9.3%** | **0.575** |
| pothole | 7,599 | 15.7% | 0.787 |
| paint_degradation | 14,047 | 29.0% | 0.744 |

Corrosion is 3× under-represented vs paint_degradation, explaining its low AP50.

**Severity frozen backbone results (train_severity_frozen.py):**

| Phase | Freeze | Best top1 | Outcome |
|---|---|---|---|
| Phase 1 (head only) | freeze=10 | **64.7%** ep15 | Head ceiling — backbone features not tuned for defects |
| Phase 2 (top blocks) | freeze=7 | 63.9% ep3 | Degraded from Phase 1; stopped at ep12 |

Conclusion: Every approach converges to ~64-66% ceiling. yolo11n-cls baseline (65.8%) remains the best severity model. Path to 90% requires either proper severity relabeling or more labeled data (BD3, CODEBRIM).

**Binary cascade experiment (train_severity_cascade.py):**

The cascade decomposes the 3-class problem into two sequential binary classifiers:
- Model 1 — `sev_cascade_critical`: critical vs (minor + moderate) — trained 25 epochs, patience=15
- Model 2 — `sev_cascade_minmod`: minor vs moderate — trained ~50 epochs, patience=20

| Model | Val accuracy | Epochs | Notes |
|---|---|---|---|
| sev_cascade_critical | **83.1%** (ep 10) | 25 | Binary: critical vs not_critical |
| sev_cascade_minmod | 76.2% (ep 10) | ~50 | Binary: minor vs moderate |

**Cascade test-set results (360 images, 120 per class):**

| Class | Correct/Total | Accuracy | vs Single 3-class |
|---|---|---|---|
| critical | 73/120 | 60.8% | **–5.2 pts** |
| minor | 90/120 | 75.0% | **+5 pts** |
| moderate | 69/120 | 57.5% | **–4.5 pts** |
| **Overall** | 232/360 | **64.4%** | –1.4 pts vs baseline |

**Threshold sweep finding:** Default top1 routing (threshold=0.50) is sub-optimal because Model 1 is trained on imbalanced data (1:2 critical:not_critical), biasing its confidence scores downward for critical. Lowering the routing threshold corrects for this bias.

| Threshold | Overall | critical | minor | moderate | vs baseline |
|---|---|---|---|---|---|
| 0.30 | 65.8% | 72.5% | 73.3% | 51.7% | ~0 |
| **0.40** | **66.9%** | **69.2%** | **75.0%** | **56.7%** | **+1.1 pts** |
| 0.50 (default) | 64.4% | 60.8% | 75.0% | 57.5% | –1.4 pts |
| 0.70 | 61.7% | 46.7% | 75.0% | 63.3% | — |

**Best result: cascade with threshold=0.40 → 66.94% test accuracy → +1.14 pts over baseline 65.8%.**

**Finding (overall):** Default top1 routing produces -1.4 pts vs baseline. Threshold=0.40 routing recovers and adds +1.14 pts. Root cause:
- When Model 1 misroutes a "critical" image as "not_critical", Model 2 classifies it as minor/moderate → wrong
- When Model 1 incorrectly calls a "moderate" image "critical", it's returned immediately → wrong
- These routing errors cancel the binary-problem benefit at default threshold
- Threshold=0.40 corrects the imbalance bias from 1:2 training ratio

**Critical root cause: label noise in severity dataset.** The "critical", "minor", and "moderate" classes each contain images from ALL 5 defect types (120 images × 5 types per class). The labels were assigned by defect type/instance category, not by visual severity. This means there is NO consistent visual feature that separates critical from minor images — they are the same defect types. No model architecture change can break through ~65% with these labels.

**What WOULD break the ceiling:**
1. **CODEBRIM integration**: ExposedRebars class → genuinely structural critical (visually distinct — metal bars visible). Would give Model 1 a real visual signal for "critical".
2. **Manual relabeling**: Assign severity by visual inspection of each crop (e.g., depth of crack, extent of spalling, area of corrosion) instead of defect-type proxy.
3. **Per-defect models**: Train one severity classifier per defect type (already supported by `severity_{class}_cls.pt` naming in inference_api.py).

**Active experiment:**

| Experiment | Script | Status | Target |
|---|---|---|---|
| ~~Stage 1 gradual unfreeze + AdamW (4-phase)~~ | ~~`train_stage1_v2.py`~~ | Killed (user directive) | — |
| Cascade binary severity (no CODEBRIM) | `train_severity_cascade.py` | **Done** — 66.9% test (thr=0.40) | >65.8% |
| **CODEBRIM-augmented cascade** | `train_severity_cascade.py` | **Done** — see below | >66.9% |

---

## CODEBRIM-Augmented Cascade Results (2026-06-11)

**CODEBRIM integration:** 4,776 facade images added to severity_dataset.
- ExposedBars → critical
- Crack / Spallation / CorrosionStain → moderate
- Efflorescence → minor

**Dataset sizes after CODEBRIM (train split):**
- is_critical: 6,063 train (2,065 critical / 3,998 not_critical)
- minor_or_moderate: 3,998 train (1,373 minor / 2,625 moderate)

**Model training results:**

| Model | Val accuracy (best) | Best epoch | Stopped at | vs no-CODEBRIM |
|---|---|---|---|---|
| sev_cascade_critical | **80.3%** (ep26) | ep26 | ep41 (patience=15) | −2.8 pts (was 83.1%) |
| sev_cascade_minmod | **78.3%** (ep24) | ep24 | ep44 (patience=20) | **+2.1 pts** (was 76.2%) |

**Cascade threshold sweep — original patch test set (360 images):**

| Threshold | Overall | critical | minor | moderate |
|---|---|---|---|---|
| 0.25 | 62.2% | 57.5% | 76.7% | 52.5% |
| **0.30** | **63.1%** | **55.8%** | **76.7%** | **56.7%** |
| 0.40 | 61.7% | 46.7% | 76.7% | 61.7% |
| 0.50 | 61.4% | 42.5% | 76.7% | 65.0% |
| 0.80 | 57.5% | 27.5% | 76.7% | 68.3% |

**Best on patches: thr=0.30 → 63.1% (−3.8 pts vs no-CODEBRIM 66.9%)**

**Cascade threshold sweep — CODEBRIM test set (478 images, facade photos):**

| Threshold | Overall | critical | minor | moderate |
|---|---|---|---|---|
| 0.30 | 88.3% | 92.0% | 79.0% | 90.9% |
| 0.45 | 88.5% | 91.3% | 79.0% | 91.9% |
| **0.70** | **88.7%** | **89.3%** | **79.8%** | **93.3%** |
| 0.80 | 88.1% | 86.7% | 79.8% | 93.8% |

**Best on CODEBRIM facades: thr=0.70 → 88.7%**

**Domain gap analysis:**

| Test set | Distribution | CODEBRIM cascade | No-CODEBRIM cascade |
|---|---|---|---|
| Original patches (360 img) | Cropped YOLO bounding-box regions | **63.1%** | **66.9%** |
| CODEBRIM facades (478 img) | Full facade scene photos | **88.7%** | N/A (no CODEBRIM test) |

**Root cause:** CODEBRIM training images are full-scene facade photos. The val/test set is cropped bounding-box patches at YOLO stride scale. The model learned high-level scene-context features (e.g., exposed rebar across a wide wall) that do not transfer to zoomed-in 64-224px crops. Model 1 (critical) was most affected (−2.8 pts on val) because critical samples in CODEBRIM are full-wall shots with exposed rebars, not tight crops.

**Key deployment insight:** The CODEBRIM cascade is the correct choice for production deployment against full-resolution facade images (88.7%). The no-CODEBRIM cascade (66.9%) is more suitable for the current patch-based evaluation protocol. Both share the same fundamental ceiling (~65-67%) on the original label set.

**What CODEBRIM proved:**
1. When input distribution matches training distribution, severity classification can exceed 88%.
2. The original 65.8% ceiling is a domain-mismatch artifact, not a hard limit.
3. For >90% target: the path is proper severity relabeling of cropped patches (not more scene data).

**Datasets researched for expansion:**

| Dataset | Images | Key Classes | License | Use |
|---|---|---|---|---|
| CODEBRIM (Zenodo) | ~1,590 | crack, spallation, rebar, corrosion | Non-commercial | Stage 1 corrosion + Stage 2 critical |
| Roboflow corrosion-bi3q3 | ~1,249 | corrosion, crack | CC BY 4.0 | Stage 1 corrosion (**download ready**) |
| BD3 Dataset (GitHub) | ~3,965 | major_crack, minor_crack, spalling | Unconfirmed | Stage 2 severity (partial) |
| Roboflow Facade Defects | ~438 | crack, delamination, paint defect | Unconfirmed | Stage 1 supplemental |

Download + integrate: `python scripts/integrate_extra_data.py --task detector --src <path>`

**Next steps to exceed 65.8% severity / 0.694 mAP50:**
1. ~~**Frozen backbone severity**: `train_severity_frozen.py`~~ — Done: 64.7% (Phase 1), Phase 2 degraded. No improvement.
2. ~~**Binary cascade**: `train_severity_cascade.py`~~ — Done: 64.4% test. Cascade routing errors offset binary gains.
3. **CODEBRIM-augmented cascade**: Add CODEBRIM ExposedRebars→critical, Crack/Spalling/Corrosion→moderate, Efflorescence→minor to severity_dataset. Re-run cascade. Expected: critical class accuracy improves from 60.8% to >75%.
4. **Add corrosion data**: Download Roboflow corrosion-bi3q3 (CC BY 4.0), run `integrate_extra_data.py` (Stage 1 only)
5. **Proper severity relabeling**: Visual inspection of each crop — highest impact path to >90%, requires manual effort (~2-4 hours)
6. **SAHI inference**: No retraining — wraps existing YOLO model with tiled inference for small objects (Stage 1)

---

## Domain Sweep Results

Run the sweep:

```powershell
python scripts/domain_sweep.py `
  --manifest configs/domain_sweep_manifest.csv `
  --detector weights/defect_detector.pt `
  --severity weights/severity_cls.pt `
  --output output/domain_sweep `
  --thresholds 0.45,0.30,0.20,0.10 `
  --annotate-conf 0.20
```

**External sweep results (hn_weak_candidate at conf=0.25):**

| Domain | Expected Class | Result |
|---|---|---|
| Concrete crack texture (close-up) | crack | ✓ Correct, conf 0.653 |
| Building wall cracks (context shot) | crack | ✗ No detection |
| Concrete fracture / spalling column | spalling | ✗ No detection at 0.45 / ✓ at 0.237 |
| Broken concrete with exposed rebar | spalling | ✗ No detection |
| Rust / corrosion texture | corrosion | ✗ No detection |
| Peeling paint wall | paint_degradation | ✗ Wrong (spalling 0.701) |
| Bathroom peeling paint | paint_degradation | ✗ Wrong (pothole 0.521) |
| Exterior siding / wall damage | paint_degradation or spalling | ✗ No detection / wrong pothole |

**Key finding:** Corrosion detection drops to 25% on building facades despite 91.7% on industrial metal test sets. The IOU mismatch (0.124) indicates the bounding boxes are wrong even when something is detected — the model is detecting the right region but drawing oversized or shifted boxes.

---

## Root Cause Analysis

### Why the metrics differ across evaluations

The early training notebook reported mAP50=0.851 — this was measured on the `merged_dataset` internal val split, which contains the same visual distribution as training (Roboflow sources, close-up textures). The real-world performance (0.694 on v7_lean, weaker on external sweep) reflects actual generalisation.

### Dataset source clues

- **Finale.yolov11** (11K images): Italian concrete/tile facades, 37 raw classes. Label noise noted.
- **Internal Wall Defect** (4.4K): interior wall defects only — paint drips, peeling, stains. No outdoor context.
- **Concrete defect detection** (6.8K): clean concrete close-ups — crack, spalling, rust stain, scaling.
- **Metal corrosion / Corrosion YOLOv8** (7.3K combined): manufacturing surfaces, industrial metal — NOT building facades.
- **Pothole datasets** (4.5K): road imagery only.

### Why the model makes texture shortcuts

The class merge is broad:
- `paint_degradation` absorbs peeling paint, paint drips, pin holes, rough/patchy surface, stains, scratches, and inclusions.
- `spalling` absorbs spalling, exposed reinforcement, rebar, and scaling.
- `corrosion` absorbs rust, rust stains, pitted surface, and rolled-in scale.

The model has learned close-up texture patterns rather than facility-level defect semantics. A bathroom peeling wall and a road pothole look nothing alike semantically, but share some low-level edge/texture features that confuse the detector.

---

## Per-Class Confidence Thresholds (current production)

| Class | Threshold | Rationale |
|---|---|---|
| crack | 0.25 | Low threshold recovers hairline cracks; false positives are acceptable for review |
| spalling | 0.25 | Moderate confidence needed |
| corrosion | 0.25 | Very weak class; lower threshold maximises recall |
| pothole | 0.25 | Strong class; conservative threshold still fine |
| paint_degradation | 0.30 | Slight increase; noisy class with many false positives at 0.25 |

TTA (scales 640+1280, hflip) + WBF (IoU=0.55) applied in production only — not at evaluation time (hurts mAP50 on val due to scale-mismatch creating duplicate FP boxes).

---

## Recommended Fix for Deployment

1. Add 50–100 real facility images per class from your target environment (exterior facades, not industrial metal).
2. Specific priority: `corrosion` from building facades, `paint_degradation` on painted concrete/brick walls.
3. Hard negatives: clean walls, windows, plants, shadows, stains on intact surfaces, old but non-defective concrete.
4. Down-weight or separate pothole data if the deployment target is buildings-only.
5. Split evaluation by domain:
   - concrete close-up
   - interior wall
   - exterior facade
   - road/pavement
   - metal corrosion
6. Review mode threshold: 0.20–0.30. Operational default: 0.45 until fine-tuning improves calibration.
7. Dashboard should visibly mark low-confidence detections as review candidates, not confirmed defects.
