# Scripts

All scripts are standalone and run from the repo root. Grouped by prefix:

| Prefix | Purpose |
|---|---|
| `train_*` | Stage 1 detector training (YOLO11m). Start with `train_staged.py`. |
| `patch_finetune_*` | Stage 2 severity fine-tuning iterations. Latest: `patch_finetune_stage2_v4.py`. |
| `build_*` | Dataset assembly — merge sources, build v7_lean, build severity crop sets. |
| `eval_*` | Model evaluation — per-class thresholds, TTA, cascade comparison. |
| `realworld_*` | Production-style sweeps on external/held-out domain images. |
| `diagnose_*` | Debug individual images or validation splits. |
| `analyze_*` | Label quality and class distribution analysis. |
| `extract_*` | Extract bounding-box crop datasets from Stage 1 detections. |
| `merge_*` | Multi-source dataset consolidation. |
| `prepare_*` | External dataset preparation (e.g., CODEBRIM). |
| `integrate_*` | Add external data into existing datasets. |
| `audit_*` | Coverage and scenario audits. |

## Key entry points

```powershell
# Build training dataset
python scripts/merge_datasets.py --config configs/merge_config.yaml --preserve-splits
python scripts/build_v7_lean.py

# Train Stage 1 (two-phase staged fine-tune — recommended)
python scripts/train_staged.py

# Train Stage 2 severity cascade
python scripts/train_severity_cascade.py

# Extract severity crops from Stage 1 detections
python scripts/extract_severity_crops.py

# Evaluate production performance
python scripts/wider_production_sweep.py
python scripts/eval_cascade_threshold.py
```
