FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system deps — do NOT override apt sources (Round 0 lesson)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js 22 for validator
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    fastapi uvicorn loguru pydantic \
    vllm openai requests httpx aiohttp \
    torch

# Copy validator and install its deps
COPY validator/ /app/validator/
RUN cd /app/validator && npm install --production

# Copy AGENTS.md for code generation prompts
COPY AGENTS.md /app/AGENTS.md

# Copy service code
COPY service/ /app/service/

# Download models at build time (cached in image layer)
# Using smaller models that fit in 4×H200 (~564GB total VRAM):
# - VLM: Qwen2.5-VL-32B (~64GB) on GPU 0
# - Code: Qwen2.5-Coder-32B (~64GB) on GPU 1
# Leaves GPU 2-3 free for parallel generation
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-VL-32B-Instruct', local_dir='/models/vlm'); \
    snapshot_download('Qwen/Qwen2.5-Coder-32B-Instruct', local_dir='/models/coder')"

EXPOSE 10006

CMD ["python", "-m", "service.main"]
