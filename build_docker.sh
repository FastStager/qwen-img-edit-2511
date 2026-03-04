#!/bin/bash
# ─── Build Docker image on Vast.ai pod ───
# Run from /workspace/ after bake_trt.py has completed.
#
# Usage:
#   bash build_docker.sh YOUR_DOCKERHUB_USER/qwen-edit-trt:latest
#
set -e

IMAGE_TAG="${1:?Usage: bash build_docker.sh REGISTRY/IMAGE:TAG}"

echo "=== Verifying models ==="
for f in models/baked_model/model_index.json models/trt_engine/transformer.pt2 models/Qwen3-VL-8B-Instruct/config.json; do
    if [ ! -f "/workspace/$f" ]; then
        echo "ERROR: /workspace/$f not found. Run bake_trt.py first."
        exit 1
    fi
done

echo "=== Building Docker image ==="
echo "Image: $IMAGE_TAG"
echo "Context: /workspace/"
echo "This will take a while (copying ~30GB of models into image)..."
echo ""

docker build \
    -f /workspace/Dockerfile.trt \
    -t "$IMAGE_TAG" \
    /workspace/

echo ""
echo "=== Build complete ==="
echo "Image: $IMAGE_TAG"
docker images "$IMAGE_TAG"

echo ""
read -p "Push to registry? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker push "$IMAGE_TAG"
    echo "=== Pushed ==="
fi
