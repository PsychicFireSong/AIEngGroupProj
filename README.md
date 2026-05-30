# AIEngGroupProj

Two-stage YOLO pipeline for facilities defect detection:

- Stage 1 detects defect type: crack, spalling, corrosion, pothole, paint degradation.
- Stage 2 classifies cropped defects by severity: minor, moderate, critical.
- The deployable dashboard now uses a Next.js frontend with a FastAPI YOLO inference backend.

## Dashboard App

Start the Python inference API from the repo root:

```powershell
python -m uvicorn apps.inference_api:app --host 0.0.0.0 --port 8000 --reload
```

Start the web dashboard:

```powershell
cd apps/facility-dashboard
npm install
npm run dev
```

Open `http://localhost:3000`.

The frontend discovers model checkpoints from the repo root, `weights/`, `models/`, `runs/`, and the Colab Drive output path. For deployment, set:

```text
NEXT_PUBLIC_INFERENCE_API_URL=https://your-inference-api.example.com
```

The old Streamlit app remains in `apps/inference_app.py` as a legacy local demo, but the intended deployable UI is `apps/facility-dashboard`.

See [QUICK_START.md](QUICK_START.md) for training and local workflow details.
