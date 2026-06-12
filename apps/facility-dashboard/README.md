# Facility AI Dashboard

Next.js operations dashboard for the two-stage YOLO defect detection pipeline.

## Local Run

Start the FastAPI inference backend from the repository root:

```powershell
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload
```

Start the dashboard:

```powershell
cd apps/facility-dashboard
npm install
npm run dev
```

Open `http://localhost:3000`.

## Vercel Frontend Deployment

Use `apps/facility-dashboard` as the Vercel project root directory.

Set this environment variable in Vercel:

```text
NEXT_PUBLIC_INFERENCE_API_URL=https://your-inference-api-host
```

The dashboard can deploy as a normal Next.js app. The repo currently includes canonical local-test `.pt` files through Git LFS, but the YOLO inference backend should still be hosted separately for deployment because Ultralytics, OpenCV, model loading, and long video jobs are better served by a dedicated Python inference service.

## Pages

- `/` dashboard overview with fleet-level metrics and trends.
- `/monitoring` live/image/video evidence review and two-stage inference details.
- `/analytics` aggregate charts for condition, risk, defect mix, action workload, and condition flags.
- `/history` visual evidence archive with focused result restore.
- `/settings` model and inference configuration.
