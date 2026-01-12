FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-VL-8B-Instruct', cache_dir='/home/user/app/models');"

COPY baked_model /home/user/app/models/baked_model


COPY handler.py .

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1 
ENV HF_DATASETS_OFFLINE=1

CMD ["python", "-u", "handler.py"]