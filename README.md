# AIEngGroupProj

Two-stage AI pipeline for automated infrastructure defect detection and severity triage.

---

## What the System Does

**Stage 1 — Defect Detector** (YOLO11m, 20M params)  
Detects five defect types in facility images and video:  
`crack` · `spalling` · `corrosion` · `pothole` · `paint_degradation`

**Stage 2 — Severity Cascade** (two binary classifiers, 3 MB each)  
Classifies each detected crop: `minor` · `moderate` · `critical`  
Binary cascade: Model 1 (is_critical) → Model 2 (minor_or_moderate)

**Dashboard** (Next.js + FastAPI)  
Image upload, live camera, video analysis, analytics, inspection history.

---

## Current Model Performance

| Metric | Value |
|---|---|
| Stage 1 mAP50 (production) | **0.694** |
| Stage 2 cascade accuracy | **66.9%** |

**Per-class detection (production sweep):**

| Class | Detection Rate | Avg Confidence |
|---|---|---|
| crack | 66.7% | 0.706 |
| spalling | 69.2% | 0.834 |
| corrosion | 25.0% | 0.727 |
| pothole | 100.0% | 0.749 |
| paint_degradation | 80.0% | 0.645 |

---

## Architecture

```
Image / Video Input
        │
        ▼
┌───────────────────┐       ┌──────────────────────────┐
│  Stage 1          │       │  Stage 2 Severity Cascade │
│  YOLO11m Detector │──────▶│  Model 1: is_critical?    │
│  5 classes, 640px │       │  Model 2: minor/moderate? │
└───────────────────┘       └──────────────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │  FastAPI Backend  │
              │  inference/api.py │
              │  Port 8000        │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │  Next.js Dashboard│
              │  Vercel hosted    │
              │  Port 3000 (local)│
              └──────────────────┘
```

---

## Repository Layout

```
apps/
  facility-dashboard/          Next.js 16 dashboard (Vercel-deployable)
    public/samples/            6 pre-loaded demo images
    src/components/            UI components + charts
    src/lib/sampleImages.ts    Sample image config

inference/
  api.py                       FastAPI inference backend (main entry point)
  api_tta_wbf.py               TTA + WBF ensemble variant
  app.py                       Legacy Streamlit demo
  Dockerfile                   Container for Hugging Face Spaces deployment

notebooks/
  training_pipeline_colab.ipynb   Baseline Colab training notebook
  revised_training_colab.ipynb    Revised training with cascaded severity
  v7_training.ipynb               v7_lean staged fine-tuning

scripts/                       Training, eval, and dataset scripts (see scripts/README.md)
  train_staged.py              Stage 1 recommended training entry point
  train_severity_cascade.py    Stage 2 binary cascade training
  build_v7_lean.py             Dataset assembly
  eval_cascade_threshold.py    Threshold sweep
  ...

weights/                       Production model weights (Git LFS)
  defect_detector.pt           Stage 1 YOLO11m (153 MB)
  severity_cls.pt              Stage 2 single-model baseline (3 MB)
  severity_critical_cls.pt     Stage 2 cascade — is_critical (3 MB)
  severity_minor_moderate_cls.pt  Stage 2 cascade — minor_or_moderate (3 MB)

configs/
  merge_config.yaml            90+ label synonyms → 5 classes
requirements.txt               Full deps (training + inference + UI)
requirements-inference.txt     Inference-only deps (for Docker deployment)
```

---

## Dataset

**v7_lean_dataset** — production training set, 7 sources merged via `configs/merge_config.yaml`.

| Split | Images |
|---|---|
| Train | 20,166 |
| Val | 3,855 |

Datasets are not stored in this repo (multi-GB). Download from Roboflow/Zenodo or run `scripts/merge_datasets.py` after setting up API keys.

---

## Local Setup

```powershell
git clone https://github.com/PsychicFireSong/AIEngGroupProj.git
cd AIEngGroupProj
git lfs pull                             # download model weights
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd apps/facility-dashboard && npm install && cd ../..
```

---

## Run Locally

**Inference backend:**
```powershell
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload
```

**Dashboard:**
```powershell
cd apps/facility-dashboard
npm run dev
# Open: http://localhost:3000
```

If backend is remote:
```powershell
$env:NEXT_PUBLIC_INFERENCE_API_URL="https://your-inference-api-host"
npm run dev
```

---

## Deployment

### Frontend — Vercel (free)

| Setting | Value |
|---|---|
| Root Directory | `apps/facility-dashboard` |
| Framework | Next.js |
| Build Command | `npm run build` |
| Environment Variable | `NEXT_PUBLIC_INFERENCE_API_URL=https://your-api-host` |

### Backend deployment

**Live camera requires GPU inference (~50 ms/frame). There is no free always-on GPU cloud service — GPU compute costs money everywhere.** For a lab demo, run the backend on the presenting machine; the GPU you already have is the best possible setup.

---

#### For a lab demo — run locally ✅ Best option

Your RTX 3070 Ti gives ~50 ms/image — live camera, image upload, and video all work at full speed. No cloud cost, no cold starts, no credit limits.

```powershell
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:3000`. The Vercel env var `NEXT_PUBLIC_INFERENCE_API_URL` defaults to `http://localhost:8000` when not set.

---

#### For a public URL — choose based on which features you need

| Need | Service | Cost | Notes |
|---|---|---|---|
| Image + video upload only | **HF Spaces** (Docker) | Free | CPU, ~1–2 s/image, no live camera |
| Image + video upload only | **Oracle Cloud Always Free** | Free | 4 ARM cores, 24 GB RAM, never sleeps |
| Live camera + GPU, credits-based | **Modal.com** | Free $10/month then billed | A10G, ~50 ms/image, scales to zero |
| Live camera + GPU, pay-per-hour | **Vast.ai / RunPod** | ~$0.10–0.20/hr | Rent GPU only when demo is running |

**HF Spaces setup:**
1. Create Space → Docker SDK → connect this repo → set secret `AIENG_MODEL_ROOTS=/app/weights`
2. Copy the Space URL → set as `NEXT_PUBLIC_INFERENCE_API_URL` in Vercel
3. Pre-warm before demos: `curl https://username-spacename.hf.space/warmup`

**Modal setup (GPU, live camera):**
```bash
pip install modal && modal setup
modal volume create aieng-weights
modal volume put aieng-weights weights/ /
modal deploy inference/modal_deploy.py
# Copy the printed URL → set as NEXT_PUBLIC_INFERENCE_API_URL in Vercel
```

---

#### Not suitable

| Service | Reason |
|---|---|
| Vercel full-stack (Python) | 250 MB bundle limit (YOLO deps alone exceed this); 60 s timeout kills video jobs |
| Render / Koyeb / Fly.io free | 256–512 MB RAM — cannot load YOLO11m (needs ~2 GB) |

---

## Inference API Endpoints

```
POST /predict          image file upload
POST /predict/url      public image or YouTube URL
POST /predict/video    async video job
GET  /jobs/{job_id}    poll video job status
GET  /health           backend health check
```

**Key options:**
- Per-class confidence thresholds: 0.25 (crack/spalling/corrosion/pothole), 0.30 (paint_degradation)
- TTA + WBF via `use_tta=True` — +37 pp corrosion recall, +11 pp crack recall

---

## Training

See [QUICK_START.md](QUICK_START.md) for the full training workflow and [scripts/README.md](scripts/README.md) for a script index.

Colab notebooks in `notebooks/` run the full pipeline on Google Drive with GPU.

**Windows training constraints (RTX 3070 Ti 8 GB):**
- `workers=0` — required to avoid DataLoader deadlock
- `copy_paste=0.0` — required with workers=0 (otherwise ~50 s/batch)
- Phase 2 `batch=8` — full unfreeze at batch=16 risks OOM

---

## More

- Dashboard details: [apps/facility-dashboard/README.md](apps/facility-dashboard/README.md)
- Quick training workflow: [QUICK_START.md](QUICK_START.md)
- Script index: [scripts/README.md](scripts/README.md)
- Domain analysis: [MODEL_DOMAIN_AUDIT.md](MODEL_DOMAIN_AUDIT.md)
