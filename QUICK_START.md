# Quick Start

This repo trains a baseline YOLO model on a merged 5-class defect dataset and includes a Streamlit app for inference.

## 1. Set up Python

From the repo root, activate the project virtual environment if it exists:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

If you are setting up a fresh environment, make sure you are using a Python version that works with Ultralytics and PyTorch on Windows.

## 2. Merge the datasets

The datasets in `archive/` and the other dataset folders are merged into one unified dataset before training.

```powershell
python scripts/merge_datasets.py --config configs/merge_config.yaml
```

If your class mapping changes, update `configs/merge_config.yaml` first.

## 3. Train the baseline model

Train the default baseline with YOLO11 nano:

```powershell
python scripts/train_baseline.py --data merged_dataset/data.yaml --weights yolo11n.pt --epochs 50 --imgsz 640 --batch 16 --device 0
```

Useful options:

- Use `--device cpu` if you do not have a GPU available.
- Change `--weights` to `yolo11s.pt` or another checkpoint if you want a stronger starting point.
- The training output is written under `runs/train/baseline` by default.

## 4. Run inference

Start the Streamlit app from the repo root:

```powershell
python -m streamlit run apps/inference_app.py
```

The app will try to load `runs/train/baseline/weights/best.pt` automatically if it exists. If your trained weights live somewhere else, paste that path into the sidebar.

## 5. What to check next

- Confirm that `merged_dataset/data.yaml` was created.
- Confirm that training produced `runs/train/baseline/weights/best.pt`.
- Open the Streamlit app and test a few images before using the model on new data.
