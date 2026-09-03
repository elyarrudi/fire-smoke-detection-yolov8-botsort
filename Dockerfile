FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN rm -f /etc/apt/sources.list.d/cuda*.list \
    /etc/apt/sources.list.d/nvidia*.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        ffmpeg \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0

COPY requirements.txt .
RUN pip3 install --no-cache-dir --default-timeout=3000 --break-system-packages -r requirements.txt

COPY . .

CMD ["python3", "main.py"]