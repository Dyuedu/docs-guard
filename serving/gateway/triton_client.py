"""Thin async gRPC client for the Triton `pipeline` model."""

from __future__ import annotations

import json

import numpy as np


class TritonPipelineClient:
    def __init__(self, url: str, model: str = "pipeline", timeout_s: float = 10.0):
        from tritonclient.grpc import aio as grpcclient

        self._g = grpcclient
        self._client = grpcclient.InferenceServerClient(url=url)
        self._model, self._timeout = model, timeout_s

    async def ready(self) -> bool:
        try:
            return await self._client.is_model_ready(self._model)
        except Exception:
            return False

    async def infer(self, image_bytes: bytes, request_id: str) -> dict:
        img = self._g.InferInput("image_bytes", [len(image_bytes)], "UINT8")
        img.set_data_from_numpy(np.frombuffer(image_bytes, dtype=np.uint8))
        rid = self._g.InferInput("request_id", [1], "BYTES")
        rid.set_data_from_numpy(np.array([request_id.encode()], dtype=object))
        result = await self._client.infer(
            self._model, [img, rid], outputs=[self._g.InferRequestedOutput("result_json")],
            request_id=request_id, client_timeout=self._timeout,
        )
        return json.loads(result.as_numpy("result_json")[0])

    async def close(self) -> None:
        await self._client.close()
