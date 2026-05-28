# ── PRIDE Server ─────────────────────────────────────────────────────────────
# Base: vllm/vllm-openai ships vLLM + PyTorch + CUDA 12 already compiled.
# We add ffmpeg, the remaining Python deps, and our application code on top.
FROM vllm/vllm-openai:latest

WORKDIR /app

# ffmpeg is required for audio extraction from video files
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (vllm and torch are already present in the base image).
# Uses opencv-python-headless (no X11/GUI dependency) — identical API for
# server-side image processing.
RUN pip install --no-cache-dir \
    "transformers>=4.46.0" \
    "accelerate>=0.27.0" \
    "qwen-vl-utils>=0.0.8" \
    "faster-whisper>=1.0.0" \
    "Pillow>=10.0.0" \
    "opencv-python-headless>=4.8.0" \
    "numpy>=1.24.0" \
    "chromadb>=0.5.0" \
    "sentence-transformers>=3.0.0" \
    "discord.py>=2.4.0" \
    "aiohttp>=3.9.0" \
    "aiofiles>=23.0.0" \
    "fastapi>=0.111.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-multipart>=0.0.9" \
    "PyYAML>=6.0" \
    "python-dotenv>=1.0.0" \
    "jurigged>=0.6.0"

# Application source
COPY src/       ./src/
COPY frontend/  ./frontend/
COPY config.yaml   .
COPY run_web.py    .
COPY run_all.py    .

# Pre-create runtime and ML-cache directories so volume mounts initialise cleanly
RUN mkdir -p \
    /workspace/.cache/huggingface \
    /workspace/.cache/torch \
    /workspace/.cache/triton \
    /workspace/.cache/vllm \
    /workspace/tmp \
    /app/data/db \
    /app/tmp \
    /app/output

EXPOSE 8080

# Default: web-only server.
# Override with "python run_all.py" (or use the 'all' Compose profile)
# to also run the Discord bot — requires DISCORD_TOKEN in the environment.
CMD ["python", "run_web.py", "--host", "0.0.0.0", "--port", "8080"]
