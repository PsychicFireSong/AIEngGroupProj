# Quick Start

Two-stage YOLO defect detection pipeline: detector training → severity classifier → inference API → dashboard.

---

## 0. Environment Setup

```powershell
git clone https://github.com/PsychicFireSong/AIEngGroupProj.git
cd AIEngGroupProj
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dashboard dependencies:
```powershell
cd apps/facility-dashboard && npm install && cd ../..
```

> **Windows note:** Always use `workers=0` and `copy_paste=0.0` in training configs.
> `copy_paste` with `workers=0` causes ~50 s/batch (vs 0.6 s). Both are already set correctly in all scripts.

---

## 1. Build Training Dataset

Merge 7 source datasets into the 5-class `v7_lean_dataset`:

```powershell
python scripts/merge_datasets.py --config configs/merge_config.yaml --preserve-splits
python scripts/build_v7_lean_dataset.py
```

This produces `output/v7_lean_dataset/data.yaml` with 20,166 train / 3,855 val images.
Disk cache (`.npy`) is built on first training run automatically.

---

## 2. Train Stage 1 Detector

### Recommended: Staged fine-tune from existing checkpoint

Two-phase approach (safer than one-shot full unfreeze):

```powershell
python scripts/train_staged.py
```

- **Phase 1** (freeze=20, batch=16, 10 epochs, ~3 h): head-only warmup
- **Phase 2** (freeze=0, batch=8, 15 epochs, ~6 h): NWD full fine-tune

Output: `output/staged_runs/staged_phase2/weights/best.pt`

### From scratch (not recommended — use staged instead)

```powershell
python scripts/train_nwd.py
```

### Resume interrupted training

`train_staged.py` auto-resumes: Phase 1 is skipped if `staged_phase1/weights/best.pt` exists.
Phase 2 is skipped if `staged_phase2/weights/best.pt` or `last.pt` exists.

---

## 3. Train Stage 2 Severity Classifier

Extract bounding-box crops from Stage 1 detections:

```powershell
python scripts/extract_severity_crops.py --config configs/merge_config.yaml
```

Train classifier:

```powershell
yolo classify train model=yolo11n-cls.pt data=severity_dataset epochs=50 imgsz=224 workers=0
```

---

## 4. Place Model Weights

Copy trained checkpoints to the canonical locations the API auto-discovers:

```powershell
New-Item -ItemType Directory -Force weights
Copy-Item output\staged_runs\staged_phase2\weights\best.pt weights\defect_detector.pt
Copy-Item runs\classify\stage2_severity\weights\best.pt weights\severity_cls.pt
```

The API also searches: repo root, `models/`, `runs/`, Colab Drive path, and any paths in `AIENG_MODEL_ROOTS`.

---

## 5. Run Inference API

```powershell
python -m uvicorn apps.inference_api:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `http://localhost:8000/health`

Key endpoints:
- `POST /predict` — image file
- `POST /predict/url` — image URL
- `POST /predict/video` — async video job
- `GET /jobs/{job_id}` — poll job result

---

## 6. Run Dashboard

```powershell
cd apps/facility-dashboard
npm run dev
```

Open `http://localhost:3000`

If API is hosted elsewhere:
```powershell
$env:NEXT_PUBLIC_INFERENCE_API_URL="https://your-api-host"
npm run dev
```

Pages: **Dashboard** (fleet overview) · **Monitoring** (live upload) · **Analytics** (trends) · **History** (evidence archive) · **Settings** (model config)

---

## 7. Evaluate Model

Run production-style sweep across domain datasets:

```powershell
python scripts/wider_production_sweep.py --help
```

Evaluate per-class confidence thresholds:

```powershell
python scripts/eval_v4_perclass.py
```

Benchmark TTA configurations (note: TTA hurts mAP50 on val — use production only):

```powershell
python scripts/eval_tta_enhanced.py
```

---

## 8. Deployment

**Frontend (Vercel):**
```
Root Directory:  apps/facility-dashboard
Build Command:   npm run build
Env var:         NEXT_PUBLIC_INFERENCE_API_URL=https://your-api-host
```

**Backend:** GPU VM or containerised service.
```powershell
python -m uvicorn apps.inference_api:app --host 0.0.0.0 --port 8000
```

Required files: `apps/inference_api.py`, `requirements.txt`, `weights/defect_detector.pt`, `weights/severity_cls.pt`

---

## Useful Environment Variables

```
AIENG_MODEL_ROOTS              extra model search paths (semicolon-separated)
AIENG_VIDEO_SCAN_FPS           default 1.0
AIENG_MAX_VIDEO_SCAN_FPS       default 30.0
AIENG_MAX_VIDEO_SCAN_SECONDS   default 600
AIENG_VIDEO_JOB_WORKERS        default 1
AIENG_MAX_VIDEO_JOBS           default 6
```

---

## Current Best Weights

| File | mAP50 | Notes |
|---|---|---|
| `defect_detector_hn_weak_candidate.pt` | **0.694** | Current best. Hard negative + weak class fine-tune. |
| `staged_phase1/weights/best.pt` | **0.682** | Phase 1 head warmup complete. |
| `staged_phase2/weights/best.pt` | **TBD** | Phase 2 NWD in progress. Target >0.80. |
| `severity_cls.pt` | **65.8%** | Severity classifier. Target >90%. |

> All metrics measured on `v7_lean_dataset` val set (3,855 images).

---

## Generate Presentation

```powershell
python scripts/generate_presentation.py
# Output: AIEngGroupProj_Presentation.pptx
```

## Take App Screenshots (after training, requires GPU)

```powershell
python scripts/capture_screenshots.py
# Starts frontend + backend, captures all pages, saves to output/screenshots/
# Run only after training is complete — loads model into VRAM
```
