FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    unzip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git rvc

WORKDIR /app/rvc

RUN pip3 install --no-cache-dir -r requirements-py311.txt

RUN pip3 install --force-reinstall 'torch==2.1.0' 'torchaudio==2.1.0' --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --force-reinstall 'numpy>=1.26,<2.0'

RUN pip3 install --no-cache-dir flask requests

COPY models/hubert_base.pt /app/rvc/assets/hubert/hubert_base.pt
COPY models/*.pth /app/rvc/assets/weights/
COPY models/*.index /app/rvc/assets/weights/
COPY api_server.py /app/api_server.py

WORKDIR /app

EXPOSE 5000

CMD ["python3", "api_server.py"]
