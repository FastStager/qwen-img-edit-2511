FROM registry.hf.space/linoyts-qwen-image-edit-2511-fast:latest

RUN pip install --no-cache-dir runpod

RUN mkdir -p /home/user/app/models
ENV HF_HOME=/home/user/app/models

RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen-Image-Edit-2511', cache_dir='/home/user/app/models'); \
    snapshot_download('lightx2v/Qwen-Image-Edit-2511-Lightning', allow_patterns=['*.safetensors'], cache_dir='/home/user/app/models'); \
    snapshot_download('Qwen/Qwen3-VL-8B-Instruct', cache_dir='/home/user/app/models');"

COPY requirements.txt /home/user/app/requirements.txt
RUN pip install --no-cache-dir -r /home/user/app/requirements.txt

COPY handler.py /home/user/app/handler.py

ENV PYTHONUNBUFFERED=1
WORKDIR /home/user/app

CMD ["python", "-u", "handler.py"]