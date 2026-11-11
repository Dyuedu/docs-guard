import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from serving.gateway.app import app


class FakePipeline:
    def __init__(self, response, ready=True, fail=False):
        self.response, self._ready, self.fail, self.calls = response, ready, fail, []

    async def ready(self):
        return self._ready

    async def infer(self, image_bytes, request_id):
        self.calls.append(request_id)
        if self.fail:
            raise RuntimeError("triton down")
        return {**self.response, "request_id": request_id}

    async def close(self):
        pass


@pytest.fixture
def jpeg_bytes():
    ok, buf = cv2.imencode(".jpg", np.full((64, 64, 3), 200, np.uint8))
    return buf.tobytes()


def _client(fake):
    app.state.pipeline = fake
    return TestClient(app)


def test_verify_ok(sample_response, jpeg_bytes):
    fake = FakePipeline(sample_response)
    with _client(fake) as c:
        r = c.post("/v1/verify", files={"image": ("a.jpg", jpeg_bytes, "image/jpeg")}, data={"request_id": "req_1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["request_id"] == "req_1"
    assert body["is_authentic"] is False
    assert body["latency_ms"]["tampering"] == 14.2  # stage breakdown from Triton is preserved
    assert fake.calls == ["req_1"]


def test_verify_generates_request_id(sample_response, jpeg_bytes):
    with _client(FakePipeline(sample_response)) as c:
        r = c.post("/v1/verify", files={"image": ("a.jpg", jpeg_bytes, "image/jpeg")})
    assert r.json()["request_id"].startswith("req_")


def test_rejects_wrong_content_type(sample_response):
    with _client(FakePipeline(sample_response)) as c:
        r = c.post("/v1/verify", files={"image": ("a.gif", b"GIF89a", "image/gif")})
    assert r.status_code == 415


def test_rejects_fake_jpeg(sample_response):
    with _client(FakePipeline(sample_response)) as c:
        r = c.post("/v1/verify", files={"image": ("a.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_backend_error_is_502(sample_response, jpeg_bytes):
    with _client(FakePipeline(sample_response, fail=True)) as c:
        r = c.post("/v1/verify", files={"image": ("a.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 502


def test_healthz(sample_response):
    with _client(FakePipeline(sample_response, ready=False)) as c:
        assert c.get("/healthz").status_code == 503
