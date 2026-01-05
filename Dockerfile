FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir huggingface_hub

RUN mkdir -p /home/user/app/models
ENV HF_HOME=/home/user/app/models

RUN python3 -c "from huggingface_hub import snapshot_download; \
    print('Downloading Qwen-Image-Edit-2511...'); \
    snapshot_download('Qwen/Qwen-Image-Edit-2511', cache_dir='/home/user/app/models'); \
    print('Downloading Lightning LoRA...'); \
    snapshot_download('lightx2v/Qwen-Image-Edit-2511-Lightning', cache_dir='/home/user/app/models'); \
    print('Downloading Qwen3-VL-8B-Instruct...'); \
    snapshot_download('Qwen/Qwen3-VL-8B-Instruct', cache_dir='/home/user/app/models');"

COPY requirements.txt /home/user/app/requirements.txt
RUN pip uninstall -y diffusers && pip install --no-cache-dir -r /home/user/app/requirements.txt

COPY handler.py /home/user/app/handler.py

RUN python3 -c "from transformers import Qwen3VLForConditionalGeneration; from diffusers import QwenImageEditPipeline; print('Verified')"

ENV PYTHONUNBUFFERED=1
WORKDIR /home/user/app

CMD ["python", "-u", "handler.py"]