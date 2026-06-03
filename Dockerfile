# Dockerfile

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# System packages needed by OpenCV, Gradio, MoviePy, video/image handling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY . /app

RUN mkdir -p \
    data/inputs \
    data/outputs \
    data/masks \
    data/cache \
    data/temp/frames \
    data/temp/processed_frames \
    data/logs \
    models

EXPOSE 7860

CMD ["python", "main.py", "--program", "visual", "--host", "0.0.0.0", "--port", "7860"]