"""SN17 404-GEN Miner Service — batch generation API on port 10006.

Implements the 4-endpoint batch API (health, status, generate, results).
Uses local vLLM servers for VLM and Code LLM inference.
"""

import asyncio
import io
import json
import subprocess
import time
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import StrEnum

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from service.generator import generate_for_prompt, init_clients


class PodStatus(StrEnum):
    WARMING_UP = "warming_up"
    READY = "ready"
    GENERATING = "generating"
    COMPLETE = "complete"
    REPLACE = "replace"


STEM_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"


class PromptItem(BaseModel):
    stem: str = Field(..., pattern=STEM_PATTERN)
    image_url: str


class GenerateBatchRequest(BaseModel):
    prompts: list[PromptItem]
    seed: int


class StatusResponse(BaseModel):
    status: PodStatus
    progress: int | None = None
    total: int | None = None


class MinerState:
    def __init__(self):
        self.status = PodStatus.WARMING_UP
        self.prompts: list[PromptItem] = []
        self.batch_stems: frozenset[str] = frozenset()
        self.results: dict[str, bytes] = {}
        self.failed: dict[str, str] = {}
        self.progress = 0
        self.total = 0
        self.cached_zip_bytes: bytes | None = None
        self._generation_task: asyncio.Task | None = None


def _start_vllm_server(model_path: str, port: int, gpu_ids: str, tp: int = 1):
    """Start a vLLM OpenAI-compatible server as a subprocess."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "16384",
        "--trust-remote-code",
    ]
    env = {"CUDA_VISIBLE_DEVICES": gpu_ids}
    import os
    full_env = {**os.environ, **env}
    proc = subprocess.Popen(cmd, env=full_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(f"Started vLLM server: model={model_path} port={port} gpus={gpu_ids} pid={proc.pid}")
    return proc


def _wait_for_server(port: int, timeout: int = 300):
    """Wait for a vLLM server to become ready."""
    import httpx
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"http://localhost:{port}/v1/models", timeout=5)
            if resp.status_code == 200:
                logger.info(f"vLLM server on port {port} is ready")
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM server on port {port} did not start within {timeout}s")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    state = _app.state.miner
    logger.info("Starting up — launching vLLM servers...")

    # Start VLM on GPU 0-1, Coder on GPU 2-3
    vlm_proc = _start_vllm_server("/models/vlm", 8001, "0,1", tp=2)
    coder_proc = _start_vllm_server("/models/coder", 8002, "2,3", tp=2)

    # Wait for both servers
    await asyncio.to_thread(_wait_for_server, 8001, timeout=600)
    await asyncio.to_thread(_wait_for_server, 8002, timeout=600)

    init_clients(vlm_port=8001, coder_port=8002)

    state.status = PodStatus.READY
    logger.info("Both vLLM servers ready — accepting batches")
    yield

    # Cleanup
    vlm_proc.terminate()
    coder_proc.terminate()


def create_app() -> FastAPI:
    application = FastAPI(title="404-GEN Miner", lifespan=lifespan)
    application.state.miner = MinerState()
    return application


app = create_app()


def _get_state() -> MinerState:
    return app.state.miner


@app.get("/health")
async def health(authorization: str | None = Header(default=None)):
    return {"status": "ok"}


@app.get("/status", response_model_exclude_none=True)
async def status(
    replacements_remaining: int = Query(default=0),
    authorization: str | None = Header(default=None),
) -> StatusResponse:
    state = _get_state()
    if state.status == PodStatus.GENERATING:
        return StatusResponse(status=PodStatus.GENERATING, progress=state.progress, total=state.total)
    return StatusResponse(status=state.status)


@app.post("/generate")
async def generate(
    request: GenerateBatchRequest,
    authorization: str | None = Header(default=None),
):
    state = _get_state()
    incoming_stems = frozenset(p.stem for p in request.prompts)

    if state.status in (PodStatus.GENERATING, PodStatus.COMPLETE) and incoming_stems == state.batch_stems:
        return {"accepted": len(request.prompts)}

    if state.status not in (PodStatus.READY, PodStatus.COMPLETE):
        return JSONResponse(status_code=409, content={"detail": "Cannot accept batch", "current_status": state.status.value})

    state.prompts = request.prompts
    state.batch_stems = incoming_stems
    state.results = {}
    state.failed = {}
    state.progress = 0
    state.total = len(request.prompts)
    state.cached_zip_bytes = None
    state.status = PodStatus.GENERATING

    logger.info(f"Received batch of {state.total} prompts (seed={request.seed})")
    state._generation_task = asyncio.create_task(_run_generation(state, request.seed))
    return {"accepted": state.total}


@app.get("/results")
async def results(authorization: str | None = Header(default=None)):
    state = _get_state()
    if state.status != PodStatus.COMPLETE:
        raise HTTPException(status_code=409, detail=f"Not complete: {state.status}")

    if state.cached_zip_bytes is None:
        state.cached_zip_bytes = _build_zip(state.results, state.failed)

    zip_bytes = state.cached_zip_bytes

    async def stream():
        offset = 0
        while offset < len(zip_bytes):
            yield zip_bytes[offset:offset + 65536]
            offset += 65536

    return StreamingResponse(stream(), media_type="application/zip")


def _build_zip(results_map: dict[str, bytes], failed_map: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem, data in results_map.items():
            zf.writestr(f"{stem}.js", data)
        if failed_map:
            zf.writestr("_failed.json", json.dumps(failed_map, indent=2))
    return buf.getvalue()


async def _run_generation(state: MinerState, seed: int):
    logger.info(f"Starting generation for {len(state.prompts)} prompts")
    start = time.monotonic()

    try:
        for i, prompt in enumerate(state.prompts):
            try:
                js_bytes = await asyncio.to_thread(
                    generate_for_prompt, prompt.stem, prompt.image_url, seed
                )
                state.results[prompt.stem] = js_bytes
            except Exception as e:
                state.failed[prompt.stem] = str(e)
                logger.error(f"Failed {prompt.stem}: {e}")
            state.progress = i + 1
    except Exception as e:
        logger.exception(f"Generation crashed: {e}")
        for prompt in state.prompts:
            if prompt.stem not in state.results and prompt.stem not in state.failed:
                state.failed[prompt.stem] = f"crashed: {e}"
    finally:
        elapsed = time.monotonic() - start
        state.cached_zip_bytes = _build_zip(state.results, state.failed)
        state.status = PodStatus.COMPLETE
        logger.info(f"Batch done: {len(state.results)}/{len(state.prompts)} in {elapsed:.1f}s")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10006)
