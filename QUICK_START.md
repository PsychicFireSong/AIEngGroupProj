# Quick Start

Two-stage YOLO defect detection pipeline: dataset → Stage 1 detector → Stage 2 severity cascade → inference API → dashboard.

---

## 0. Environment Setup

```powershell
git clone https://github.com/PsychicFireSong/AIEngGroupProj.git
cd AIEngGroupProj
git lfs pull                             # download model weights (~160 MB)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd apps/facility-dashboard && npm install && cd ../..
```

> **Windows note:** Always use `workers=0` and `copy_paste=0.0` in training scripts.
> `copy_paste` with `workers=0` causes ~50 s/batch vs 0.6 s. Both are set correctly in all scripts.

---

## 1. Build Training Dataset

Merge 7 source datasets into the 5-class `v7_lean_dataset`:

```powershell
python scripts/merge_datasets.py --config configs/merge_config.yaml --preserve-splits
python scripts/build_v7_lean.py
```

Produces `output/v7_lean_dataset/data.yaml` — 20,166 train / 3,855 val images.

Required API keys (set as environment variables or Colab secrets):
```
ROBOFLOW_API_KEY
KAGGLE_USERNAME
KAGGLE_KEY
```

---

## 2. Train Stage 1 Detector

### Recommended: two-phase staged fine-tune

```powershell
python scripts/train_staged.py
```

- **Phase 1** (`freeze=20`, batch=16, 10 epochs, ~3 h): head-only warmup → `output/staged_runs/staged_phase1/weights/best.pt`
- **Phase 2** (`freeze=0`, batch=8, 15 epochs, ~6 h): full unfreeze → `output/staged_runs/staged_phase2/weights/best.pt`

`train_staged.py` auto-resumes — safe to interrupt and restart.

### Train from scratch

```powershell
python scripts/train_nwd.py
```

---

## 3. Train Stage 2 Severity Cascade

Extract bounding-box crops from Stage 1:

```powershell
python scripts/extract_severity_crops.py
```

Train the two binary classifiers:

```powershell
python scripts/train_severity_cascade.py
```

Outputs:
- `weights/severity_critical_cls.pt` — Model 1: is_critical?
- `weights/severity_minor_moderate_cls.pt` — Model 2: minor or moderate?

---

## 4. Place Model Weights

```powershell
New-Item -ItemType Directory -Force weights
Copy-Item output\staged_runs\staged_phase2\weights\best.pt weights\defect_detector.pt
# Cascade models are saved directly to weights/ by train_severity_cascade.py
```

The API searches: `weights/`, repo root, `models/`, `runs/`, Colab Drive path, and any paths in `AIENG_MODEL_ROOTS`.

---

## 5. Run Inference API

```powershell
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload
```

Health check: `http://localhost:8000/health`

Endpoints:
- `POST /predict` — image file
- `POST /predict/url` — image/YouTube URL
- `POST /predict/video` — async video job
- `GET /jobs/{job_id}` — poll job result

---

## 6. Run Dashboard

```powershell
cd apps/facility-dashboard
npm run dev
# Open: http://localhost:3000
```

If the API is hosted elsewhere:
```powershell
$env:NEXT_PUBLIC_INFERENCE_API_URL="https://your-api-host"
npm run dev
```

Pages: **Dashboard** · **Monitoring** (upload / live camera) · **Analytics** · **History** · **Settings**

---

## 7. Evaluate Model

```powershell
# Production sweep across domain images
python scripts/wider_production_sweep.py

# Per-class threshold tuning
python scripts/eval_v4_perclass.py

# Cascade vs single-model comparison
python scripts/eval_staged_comparison.py

# TTA benchmark
python scripts/eval_tta_enhanced.py
```

---

## 8. Deploy

### Frontend → Vercel (free)

```
Root Directory:  apps/facility-dashboard
Build Command:   npm run build
Env var:         NEXT_PUBLIC_INFERENCE_API_URL=https://your-api-host
```

### Backend → Hugging Face Spaces (free, CPU-only)

HF Spaces hosts the **backend only** — the Next.js frontend goes to Vercel.

1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces) → SDK: **Docker**
2. Connect this GitHub repo. HF Spaces finds the `Dockerfile` at the repo root automatically.
3. Set Space secret → `AIENG_MODEL_ROOTS=/app/weights`
4. Copy the Space URL and set it as the Vercel env var:
   `NEXT_PUBLIC_INFERENCE_API_URL=https://username-spacename.hf.space`

CPU inference is ~1–3 s/image — fine for a demo. Free with no time limit.

---

## Production Weights

| File | Description | Accuracy |
|---|---|---|
| `weights/defect_detector.pt` | Stage 1 YOLO11m detector | mAP50 0.694 |
| `weights/severity_cls.pt` | Stage 2 single-model baseline | 65.8% |
| `weights/severity_critical_cls.pt` | Stage 2 cascade — is_critical | 80.3% val |
| `weights/severity_minor_moderate_cls.pt` | Stage 2 cascade — minor vs moderate | 78.3% val |

All metrics on `v7_lean_dataset` val set (3,855 images).

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
