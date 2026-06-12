FROM python:3.11-slim

# System deps: OpenCV, ffmpeg, git + git-lfs (for weight download)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg git git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless requirements change)
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy inference package and bundled sample media
COPY inference/ ./inference/
COPY samples/ ./apps/facility-dashboard/public/samples/

# Download real model weights from GitHub LFS (public repo, no auth needed)
RUN git clone --depth=1 --no-checkout \
        https://github.com/PsychicFireSong/AIEngGroupProj.git /tmp/weights-repo \
    && cd /tmp/weights-repo \
    && git sparse-checkout init --cone \
    && git sparse-checkout set weights \
    && git checkout HEAD \
    && git lfs pull --include="weights/*.pt" \
    && cp -r weights /app/weights \
    && rm -rf /tmp/weights-repo

# HF Spaces exposes port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "7860"]
