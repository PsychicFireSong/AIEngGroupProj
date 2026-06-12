FROM python:3.11-slim

# System deps needed by OpenCV and yt-dlp
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements change)
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy inference package and weights
COPY inference/ ./inference/
COPY weights/ ./weights/

# HF Spaces exposes port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "7860"]
