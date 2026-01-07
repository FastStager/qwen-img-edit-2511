FROM nightfury172/qwen-edit-2511-runpod:latest

RUN pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /home/user/app/requirements.txt
RUN pip uninstall -y diffusers && pip install --no-cache-dir -r /home/user/app/requirements.txt

RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-VL-8B-Instruct', cache_dir='/home/user/app/models'); \
    snapshot_download('lightx2v/Qwen-Image-Edit-2511-Lightning', allow_patterns=['*.safetensors'], cache_dir='/home/user/app/models');"

COPY handler.py /home/user/app/handler.py

RUN python3 -c "from transformers import Qwen3VLForConditionalGeneration; from diffusers import FlowMatchEulerDiscreteScheduler; print('Verified')"

ENV PYTHONUNBUFFERED=1
WORKDIR /home/user/app

CMD ["python", "-u", "handler.py"]