"""FastAPI gateway: validates uploads, forwards to Triton over gRPC, enforces the response schema."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile

from docguard.schemas import VerifyResponse

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 2**20)))
ALLOWED_TYPES = {"image/jpeg", "image/png"}
_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")


class PipelineClient(Protocol):
    async def ready(self) -> bool: ...
    async def infer(self, image_bytes: bytes, request_id: str) -> dict: ...
    async def close(self) -> None: ...


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "pipeline"):  # tests inject a fake client before startup
        from .triton_client import TritonPipelineClient

        app.state.pipeline = TritonPipelineClient(os.getenv("TRITON_GRPC_URL", "localhost:8001"))
    yield
    await app.state.pipeline.close()


app = FastAPI(title="doc-guard Gateway", version="0.1.0", lifespan=lifespan)


def get_pipeline(request: Request) -> PipelineClient:
    return request.app.state.pipeline


@app.get("/healthz")
async def healthz(pipeline: PipelineClient = Depends(get_pipeline)):
    if not await pipeline.ready():
        raise HTTPException(503, "triton pipeline not ready")
    return {"status": "ok"}


@app.post("/v1/verify", response_model=VerifyResponse)
async def verify(
    image: UploadFile = File(...),
    request_id: str | None = Form(None),
    pipeline: PipelineClient = Depends(get_pipeline),
):
    t0 = time.perf_counter()
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"unsupported content type {image.content_type}")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image too large")
    if not data.startswith(_MAGIC):
        raise HTTPException(400, "file is not a JPEG/PNG")

    rid = request_id or f"req_{uuid.uuid4().hex[:12]}"
    try:
        result = await pipeline.infer(data, rid)
    except Exception as e:
        raise HTTPException(502, f"inference backend error: {e}") from e

    resp = VerifyResponse.model_validate(result)
    # Report latency as seen by the gateway (includes gRPC hop), keep Triton's stage breakdown.
    resp.latency_ms.total = round((time.perf_counter() - t0) * 1000, 2)
    return resp
