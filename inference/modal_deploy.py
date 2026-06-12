"""
Modal deployment for the AI Facilities inference API.

Provides GPU-backed inference (A10G ~50 ms/image) so live camera works.
Free $10/month credits = ~77 GPU-hours — plenty for weekly lab demos.

Setup (one time):
    pip install modal
    modal setup                          # authenticate
    modal volume create aieng-weights
    modal volume put aieng-weights weights/ /   # upload model weights

Deploy:
    modal deploy inference/modal_deploy.py

Pre-warm before a demo (avoids the ~60 s cold-start penalty):
    curl https://<your-modal-url>/warmup

Your public API URL appears in the modal deploy output.
Set it as NEXT_PUBLIC_INFERENCE_API_URL in Vercel.
"""

import os
import modal

# ── Image ─────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install_from_requirements("requirements-inference.txt")
)

# ── Persistent volume for model weights ───────────────────────────────────────
# Upload once with: modal volume put aieng-weights weights/ /
weights_vol = modal.Volume.from_name("aieng-weights", create_if_missing=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = modal.App("aieng-inference", image=image)


@app.function(
    gpu="A10G",                          # ~50 ms/image — live camera works
    volumes={"/app/weights": weights_vol},
    min_containers=0,                    # scale to zero between sessions (saves credits)
    timeout=600,                         # allow long video jobs
    allow_concurrent_inputs=4,           # handle a few simultaneous users
)
@modal.asgi_app()
def api():
    os.environ.setdefault("AIENG_MODEL_ROOTS", "/app/weights")
    from inference.api import app as fastapi_app
    return fastapi_app
