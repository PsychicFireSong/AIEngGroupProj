# AIEngGroupProj

Two-stage YOLO pipeline for facilities defect detection and severity triage.

## Deployment Status

This repository has been pushed and prepared for deployment, but it has not been deployed yet.

The intended production shape is:

- Next.js dashboard frontend deployed from `apps/facility-dashboard`.
- Python FastAPI inference backend hosted separately with access to trained `.pt` weights.
- Frontend points to the backend through `NEXT_PUBLIC_INFERENCE_API_URL`.

The current local-testing detector and severity checkpoints are included under `weights/` through Git LFS so teammates can run the app immediately after cloning. Larger experiments and older checkpoints should still stay in Google Drive, model storage, or the inference server filesystem.

## What The System Does

- Stage 1 detects defect type: `crack`, `spalling`, `corrosion`, `pothole`, `paint_degradation`.
- Stage 2 classifies cropped defect regions by severity: `minor`, `moderate`, `critical`.
- The dashboard supports image upload, camera/live monitoring, video/link analysis, history review, and aggregate analytics.
- Analytics includes condition/risk trend, defect distribution, severity mix, action workload, condition flags, and recommendation quality.
- History is evidence-first: saved inspections, visual thumbnails, expandable video defect frames, and focused result restore.

## Repository Layout

```text
apps/
  facility-dashboard/        Next.js dashboard frontend
  inference_api.py           FastAPI inference backend
  inference_app.py           Legacy Streamlit demo
configs/
  merge_config.yaml          Baseline dataset merge mapping
  production_eval_manifest.csv
scripts/
  merge_datasets.py
  extract_severity_crops.py
  auto_collect_domain_sources.py
  build_*_dataset.py
  wider_production_sweep.py
two_stage_yolo_defect_pipeline_colab.ipynb
two_stage_yolo_defect_pipeline_revised_training_colab.ipynb
```

## Fresh Local Setup

Clone the repo:

```powershell
git clone https://github.com/PsychicFireSong/AIEngGroupProj.git
cd AIEngGroupProj
```

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install dashboard dependencies:

```powershell
cd apps/facility-dashboard
npm install
cd ../..
```

## Model Weights Setup

The API discovers `.pt` files from these locations:

- repo root
- `weights/`
- `models/`
- `runs/`
- Colab Drive path: `/content/drive/MyDrive/AIEngGroupProj_colab_outputs/weights`
- optional extra paths through `AIENG_MODEL_ROOTS`

For local testing, create a local `weights/` folder and place:

```text
weights/defect_detector.pt
weights/severity_cls.pt
```

These two canonical local-test files are tracked with Git LFS. Other `.pt` files are ignored by Git.

## Run Locally

Start the inference backend from the repo root:

```powershell
python -m uvicorn apps.inference_api:app --host 0.0.0.0 --port 8000 --reload
```

Check:

```text
http://localhost:8000/health
```

Start the dashboard:

```powershell
cd apps/facility-dashboard
npm run dev
```

Open:

```text
http://localhost:3000
```

If the backend is hosted somewhere else, set:

```powershell
$env:NEXT_PUBLIC_INFERENCE_API_URL="https://your-inference-api-host"
npm run dev
```

## Google Colab Training

Use the notebooks for recoverable training:

- `two_stage_yolo_defect_pipeline_colab.ipynb` for baseline setup/training.
- `two_stage_yolo_defect_pipeline_revised_training_colab.ipynb` for revised domain/balanced training.

Before running in Colab, configure secrets or environment variables instead of hard-coding keys:

```text
ROBOFLOW_API_KEY
KAGGLE_USERNAME
KAGGLE_KEY
```

Training outputs should be stored in Google Drive, usually:

```text
/content/drive/MyDrive/AIEngGroupProj_colab_outputs/
```

After training, copy or export the selected detector and severity classifier as:

```text
defect_detector.pt
severity_cls.pt
```

## Vercel Frontend Deployment

Use Vercel for the Next.js frontend only.

Recommended Vercel settings:

```text
Root Directory: apps/facility-dashboard
Framework Preset: Next.js
Build Command: npm run build
Install Command: npm install
Output Directory: .next
```

Set this Vercel environment variable:

```text
NEXT_PUBLIC_INFERENCE_API_URL=https://your-inference-api-host
```

The dashboard will deploy as a static/Next.js frontend, but detection will only work when the API URL points to a live inference backend.

## Inference Backend Deployment

The Python backend is heavier than a normal frontend function because it loads Ultralytics YOLO, OpenCV, model checkpoints, and background video jobs.

Recommended backend options:

- GPU VM or cloud instance.
- Dockerized Python service on a provider that supports long-running processes.
- Render/Railway/Fly.io/AWS/GCP/Azure container service.
- University/local server for demo deployment.

Backend start command:

```powershell
python -m uvicorn apps.inference_api:app --host 0.0.0.0 --port 8000
```

Required backend files:

```text
apps/inference_api.py
requirements.txt
weights/defect_detector.pt
weights/severity_cls.pt
```

Useful backend environment variables:

```text
AIENG_MODEL_ROOTS=extra/path/one;extra/path/two
AIENG_VIDEO_SCAN_FPS=1.0
AIENG_MAX_VIDEO_SCAN_FPS=30.0
AIENG_MAX_VIDEO_SCAN_SECONDS=600
AIENG_MAX_VIDEO_SCAN_SAMPLES=9000
AIENG_MAX_RETURNED_VIDEO_FRAMES=24
AIENG_VIDEO_JOB_WORKERS=1
AIENG_MAX_VIDEO_JOBS=6
```

## Validation Commands

Frontend:

```powershell
cd apps/facility-dashboard
npm run lint
npm run build
```

Python syntax check:

```powershell
py -3 -m py_compile apps/inference_api.py scripts/merge_datasets.py scripts/extract_severity_crops.py scripts/domain_sweep.py
```

Production-style sweep after weights are available:

```powershell
python scripts/wider_production_sweep.py --help
```

## Security And Cleanup Rules

- Do not commit extra `.pt` model weights outside the canonical Git LFS files in `weights/`.
- Do not commit Roboflow/Kaggle API keys.
- Do not commit `output/`, `runs/`, downloaded datasets, or generated inspection history.
- Store training outputs, experimental checkpoints, and old model artifacts in Google Drive or backend storage.

## More Details

- Dashboard-specific instructions: [apps/facility-dashboard/README.md](apps/facility-dashboard/README.md)
- Short command workflow: [QUICK_START.md](QUICK_START.md)
