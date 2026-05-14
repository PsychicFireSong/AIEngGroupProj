from __future__ import annotations

import io
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


st.set_page_config(page_title="YOLO Inference", layout="wide")


def _default_model_path() -> str:
    candidates = [
        Path("runs/train/baseline/weights/best.pt"),
        Path("runs/detect/train/baseline/weights/best.pt"),
        Path("yolo11n.pt"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


@st.cache_resource
def _load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def _device_options() -> list[str]:
    options = ["auto", "cpu"]
    if torch is not None and torch.cuda.is_available():
        options.insert(1, "cuda:0")
    return options


def _device_arg(device_choice: str) -> str | None:
    if device_choice == "auto":
        return None
    return device_choice


def _annotate_image(model: YOLO, image: np.ndarray, conf: float, iou: float, device: str | None) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model.predict(source=image_bgr, conf=conf, iou=iou, device=device, verbose=False)
    return results[0].plot()


def _render_video(
    model: YOLO,
    input_path: Path,
    output_path: Path,
    conf: float,
    iou: float,
    device: str | None,
    progress: st.progress,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Unable to open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated = _annotate_image(model, frame_rgb, conf, iou, device)
        writer.write(annotated)

        processed += 1
        if total > 0:
            progress.progress(min(processed / total, 1.0))

    cap.release()
    writer.release()
    progress.progress(1.0)


st.title("YOLO Inference")

with st.sidebar:
    st.header("Model")
    default_model = _default_model_path()
    model_path = st.text_input("Model path", value=default_model, placeholder="path/to/best.pt")

    device_choice = st.selectbox("Device", _device_options())
    conf = st.slider("Confidence", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    iou = st.slider("IoU", min_value=0.1, max_value=0.9, value=0.6, step=0.05)

if not model_path:
    st.warning("Provide a valid model path in the sidebar.")
    st.stop()

model_file = Path(model_path)
if not model_file.exists():
    st.error(f"Model not found: {model_file}")
    st.stop()

model = _load_model(str(model_file))


def _extract_model_names(model_obj) -> list[str]:
    names = None
    try:
        names = getattr(model_obj, "names", None)
    except Exception:
        names = None
    if names is None and hasattr(model_obj, "model"):
        try:
            names = getattr(model_obj.model, "names", None)
        except Exception:
            names = None

    if isinstance(names, dict):
        items = list(names.values())
    elif isinstance(names, (list, tuple)):
        items = list(names)
    else:
        items = []
    return items


model_names = _extract_model_names(model)
st.sidebar.markdown("**Loaded model**")
st.sidebar.write(str(model_file))
st.sidebar.write(f"Classes: {len(model_names)}")
if model_names:
    st.sidebar.write("Example labels: " + ", ".join(model_names[:10]))
    lower_names = {n.lower() for n in model_names}
    if "person" in lower_names and "surfboard" in lower_names:
        st.sidebar.warning(
            "Model appears to be COCO-pretrained (contains 'person' and 'surfboard').\n"
            "Load your trained `best.pt` (or point the sidebar to your run's weights) to get project classes."
        )

st.subheader("Image Inference")
image_files = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if image_files:
    for uploaded_file in image_files:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        annotated = _annotate_image(model, image_array, conf, iou, _device_arg(device_choice))
        st.image(annotated, caption=uploaded_file.name, channels="BGR", use_container_width=True)

st.subheader("Video Inference")
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False)

if video_file:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        input_path = temp_dir_path / video_file.name
        output_path = temp_dir_path / f"annotated_{video_file.name.rsplit('.', 1)[0]}.mp4"
        input_path.write_bytes(video_file.read())

        progress = st.progress(0)
        status = st.empty()
        status.write("Processing video...")
        _render_video(model, input_path, output_path, conf, iou, _device_arg(device_choice), progress)
        status.write("Done.")
        st.video(output_path.read_bytes())
