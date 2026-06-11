# AIEngGroupProj

Two-stage AI pipeline for automated infrastructure defect detection and severity triage.

---

## What the System Does

**Stage 1 — Defect Detector** (YOLO11m, 20M params)
Detects five defect types in facility images and video:
`crack` · `spalling` · `corrosion` · `pothole` · `paint_degradation`

**Stage 2 — Severity Classifier** (lightweight CNN, 3.2 MB)
Classifies each detected crop: `minor` · `moderate` · `critical`

**Dashboard** (Next.js + FastAPI)
Image upload, live camera, video analysis, analytics, inspection history.

---

## Current Model Performance

| Metric | Value | Notes |
|---|---|---|
| Stage 1 mAP50 (baseline) | **0.694** | `defect_detector_hn_weak_candidate.pt` |
| Stage 1 mAP50 (Phase 1 staged) | **0.682** | Head-only warmup complete |
| Stage 1 mAP50 (Phase 2 NWD) | **in progress** | Full unfreeze + NWD loss, ~6h remaining |
| Stage 1 mAP50 target | **> 0.80** | NWD staged training |
| Stage 2 accuracy (current) | **65.8%** | Severity classifier |
| Stage 2 accuracy target | **> 90%** | Per-defect specialist models |

**Per-class detection (production sweep, baseline):**

| Class | Detection Rate | Avg Confidence | Notes |
|---|---|---|---|
| crack | 66.7% | 0.706 | External dataset: 75.0% |
| spalling | 69.2% | 0.834 | Strong |
| corrosion | 25.0% | 0.727 | Domain shift — industrial metal vs building facade |
| pothole | 100.0% | 0.749 | Excellent |
| paint_degradation | 80.0% | 0.645 | External: 100% |

---

## Architecture

```
Image / Video Input
        │
        ▼
┌───────────────────┐       ┌───────────────────────┐
│  Stage 1          │       │  Stage 2               │
│  YOLO11m Detector │──────▶│  Severity Classifier   │
│  5 classes, 640px │       │  minor/moderate/critical│
└───────────────────┘       └───────────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        ▼
              ┌──────────────────┐
              │  FastAPI Backend  │
              │  inference_api.py │
              │  Port 8000, GPU   │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │  Next.js Dashboard│
              │  Vercel hosted    │
              │  Port 3000 (local)│
              └──────────────────┘
```

---

## Dataset

**v7_lean_dataset** — final production training set.

| Split | Images |
|---|---|
| Train | 20,166 |
| Val | 3,855 |

**7 source datasets → 5 classes** (90+ label synonyms mapped via `configs/merge_config.yaml`):

| Source | Images | Mapped Classes | Domain |
|---|---|---|---|
| Finale.yolov11 (Roboflow) | 11,046 | crack, spalling, paint_degradation | Concrete facades |
| Concrete Defect Detection | 6,806 | crack, spalling, corrosion, paint_degradation | Clean concrete |
| Internal Wall Defect | 4,417 | paint_degradation | Interior walls |
| Metal Corrosion | 5,697 | crack, corrosion, paint_degradation | Industrial metal |
| Corrosion YOLOv8 | 1,644 | corrosion | Metal rust |
| Pothole Detection YOLOv8 | 608 | pothole | Road surfaces |
| Kaggle Pothole Archive | 3,940 | pothole | Road surfaces |
| **Total** | **34,158** | | |

---

## Training History

| Run | mAP50 | Result |
|---|---|---|
| Baseline (merged 34K) | 0.851 | Initial. Domain shift found on external eval. |
| Hard negative + weak class fine-tune | **0.694** | Current best. `defect_detector_hn_weak_candidate.pt` |
| v7_fixed oversampled ❌ | 0.612 | Catastrophic forgetting. Killed. |
| Staged Phase 1 (freeze=20, head-only) | 0.682 | Complete. Head calibrated on clean data. |
| Staged Phase 2 (NWD, full unfreeze) | **TBD** | Running. Expected +2-4 mAP pts from NWD loss. |

**Key training constraints (Windows / RTX 3070 Ti 8 GB):**
- `workers=0` — required, avoids DataLoader deadlock on Windows
- `copy_paste=0.0` — required with workers=0; copy_paste=0.3 causes ~50 s/batch
- Phase 2 `batch=8` — full unfreeze at batch=16 risks OOM (7.77 GB / 8.19 GB)

---

## Repository Layout

```
apps/
  facility-dashboard/        Next.js 16 dashboard frontend
  inference_api.py           FastAPI GPU inference backend
  inference_app.py           Legacy Streamlit demo

configs/
  merge_config.yaml          Dataset merge class mapping (90+ synonyms → 5 classes)
  production_eval_manifest.csv

scripts/
  train_staged.py            Two-phase staged fine-tuning (current)
  train_nwd.py               NWD loss single-phase training
  merge_datasets.py          Multi-source dataset consolidation
  extract_severity_crops.py  Crop ROIs for severity classifier
  build_v7_lean_dataset.py   Build v7_lean training split
  eval_tta_enhanced.py       TTA evaluation benchmark
  wider_production_sweep.py  Production-style eval across domains
  generate_presentation.py   Generate PowerPoint presentation
  ... (58 scripts total)

two_stage_yolo_defect_pipeline_colab.ipynb        Baseline training notebook
two_stage_yolo_defect_pipeline_revised_training_colab.ipynb
```

---

## Local Setup

```powershell
git clone https://github.com/PsychicFireSong/AIEngGroupProj.git
cd AIEngGroupProj
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd apps/facility-dashboard && npm install && cd ../..
```

Place model weights:
```
weights/defect_detector.pt      # Stage 1 detector
weights/severity_cls.pt         # Stage 2 classifier
```

---

## Run Locally

**Inference backend:**
```powershell
python -m uvicorn apps.inference_api:app --host 0.0.0.0 --port 8000 --reload
# Health check: http://localhost:8000/health
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

## Inference API

`POST /predict` — image file upload  
`POST /predict/url` — public image URL  
`POST /predict/video` — async video job  
`GET /jobs/{job_id}` — poll video job result  
`GET /health` — backend status

**Key inference options:**
- Per-class confidence thresholds: 0.25 (crack/spalling/corrosion/pothole), 0.30 (paint_degradation)
- TTA + WBF via `use_tta=True` — +37 pp corrosion recall, +11 pp crack recall in production
- Video: configurable FPS, max duration, async job queue

**Environment variables:**
```
AIENG_MODEL_ROOTS              extra model search paths (semicolon-separated)
AIENG_VIDEO_SCAN_FPS           default 1.0
AIENG_MAX_VIDEO_SCAN_FPS       default 30.0
AIENG_MAX_VIDEO_SCAN_SECONDS   default 600
AIENG_VIDEO_JOB_WORKERS        default 1
AIENG_MAX_VIDEO_JOBS           default 6
```

---

## Google Colab Training

```
two_stage_yolo_defect_pipeline_colab.ipynb          — baseline setup
two_stage_yolo_defect_pipeline_revised_training_colab.ipynb  — revised training
```

Required Colab secrets: `ROBOFLOW_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`

Outputs stored at: `/content/drive/MyDrive/AIEngGroupProj_colab_outputs/`

---

## Deployment

**Frontend (Vercel):**
```
Root Directory:   apps/facility-dashboard
Framework:        Next.js
Build Command:    npm run build
Env var:          NEXT_PUBLIC_INFERENCE_API_URL=https://your-api-host
```

**Backend:** GPU VM or containerised service (needs CUDA, Ultralytics, OpenCV).
```powershell
python -m uvicorn apps.inference_api:app --host 0.0.0.0 --port 8000
```

---

## Validation

```powershell
# Frontend lint + build
cd apps/facility-dashboard && npm run lint && npm run build

# Python syntax check
py -3 -m py_compile apps/inference_api.py scripts/merge_datasets.py

# Production sweep (needs weights)
python scripts/wider_production_sweep.py --help

# Generate presentation
python scripts/generate_presentation.py
```

---

## Security Rules

- Do not commit `.pt` files outside `weights/` (tracked by Git LFS).
- Do not commit API keys (Roboflow, Kaggle).
- Do not commit `output/`, `runs/`, downloaded datasets, inspection history.

---

## More

- Dashboard details: [apps/facility-dashboard/README.md](apps/facility-dashboard/README.md)
- Quick workflow: [QUICK_START.md](QUICK_START.md)
- Domain analysis: [MODEL_DOMAIN_AUDIT.md](MODEL_DOMAIN_AUDIT.md)
- Presentation: `AIEngGroupProj_Presentation.pptx` (run `python scripts/generate_presentation.py`)
