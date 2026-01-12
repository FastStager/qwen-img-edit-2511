FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home/user/app

RUN apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY bake.py .
RUN python3 bake.py && \
    rm -rf /root/.cache/huggingface && \
    rm bake.py

COPY handler.py .

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1 
ENV HF_DATASETS_OFFLINE=1

CMD ["python", "-u", "handler.py"]