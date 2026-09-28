# Sentinel AI: one image with the dashboard, the API, the models, and the sample videos.
#   docker compose up --build        -> http://localhost:8000 (demo on a sample video, no camera)

# ---- 1. Dashboard -----------------------------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Runtime -------------------------------------------------------------------------
FROM python:3.11-slim

# libGL + glib are what the opencv-python wheel needs at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# CPU-only torch by default (~200 MB instead of ~2.5 GB of CUDA libraries).
# For an NVIDIA GPU build: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch torchvision --index-url ${TORCH_INDEX}
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/ backend/
COPY config/ config/
COPY scripts/fetch_samples.py scripts/fetch_samples.py

# Bake in the YOLO weights and the checksummed sample videos, so the container
# starts instantly and works offline.
RUN python scripts/fetch_samples.py --models

COPY --from=frontend /frontend/dist frontend/dist

RUN useradd --create-home --uid 1000 sentinel \
    && mkdir -p data/clips \
    && chown -R sentinel:sentinel /app/data
USER sentinel

# Defaults: run the corridor demo on loop. Override with VIDEO_SOURCE=0 / rtsp://... / file.
ENV VIDEO_SOURCE=demo \
    LOOP_VIDEO=true \
    FORCE_CPU=true \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["python", "backend/main.py"]
