#!/bin/bash
set -e

echo "=== Project_H Setup ==="

# ── 1. Python packages ──────────────────────────────────────────
echo "[1/7] Installing Python packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.54.0 \
            accelerate==1.13.0 \
            open-clip-torch==3.3.0 \
            qwen-vl-utils==0.0.14 \
            segment-anything==1.0 \
            pycocotools==2.0.11 \
            spacy==3.8.11 \
            opencv-python==4.13.0.92 \
            Pillow==12.1.1 \
            numpy==2.4.3 \
            requests==2.32.5

# ── 2. SpaCy language model ─────────────────────────────────────
echo "[2/7] Downloading SpaCy model..."
python -m spacy download en_core_web_sm

# ── 3. SAM weights ──────────────────────────────────────────────
echo "[3/7] Downloading SAM ViT-H weights (~2.4GB)..."
mkdir -p weights
wget -q --show-progress -O weights/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# ── 4. GroundingDINO ────────────────────────────────────────────
echo "[4/7] Installing GroundingDINO..."
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e GroundingDINO/

# ── 5. CountGD ──────────────────────────────────────────────────
echo "[5/7] Installing CountGD..."
git clone https://github.com/niki-amini-naieni/CountGD.git
pip install -e CountGD/

# ── 6. Ollama + Llama 3.1 8B ────────────────────────────────────
echo "[6/7] Installing Ollama and pulling Llama 3.1 8B (~5GB)..."
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
sleep 5
ollama pull llama3.1:8b

# ── 7. Qwen2-VL-7B and OpenCLIP ────────────────────────────────
echo "[7/7] Pre-downloading Qwen2-VL-7B (~15GB) and OpenCLIP..."
python - << 'PYEOF'
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
print("Downloading Qwen2-VL-7B...")
AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto"
)
print("Downloading OpenCLIP ViT-L/14...")
import open_clip
open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
print("Done.")
PYEOF

echo ""
echo "=== Setup Complete ==="
echo ""
echo "NOTE: ComfyUI and diffusion model weights must be set up manually."
echo "See README for ComfyUI installation instructions."
