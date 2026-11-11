"""Load test for the gateway (proposal §6, system benchmark).

    locust -f loadtest/locustfile.py --host http://localhost:8080 \
        --users 32 --spawn-rate 8 --run-time 3m --headless --csv results/triton_u32

Point IMAGE_DIR at held-out test receipts. Run the same script against the
FastAPI + PyTorch baseline to compare QPS / p95 / p99.
"""

import os
import random
from pathlib import Path

from locust import HttpUser, between, task

IMAGE_DIR = Path(os.getenv("IMAGE_DIR", "data/tamper/test/images"))
IMAGES = [(p.name, p.read_bytes()) for p in sorted(IMAGE_DIR.glob("*.jpg"))[:200]]
if not IMAGES:
    raise SystemExit(f"no .jpg files in {IMAGE_DIR}")


class VerifyUser(HttpUser):
    wait_time = between(0, 0.05)

    @task
    def verify(self):
        name, data = random.choice(IMAGES)
        with self.client.post(
            "/v1/verify", files={"image": (name, data, "image/jpeg")}, name="/v1/verify", catch_response=True
        ) as r:
            if r.status_code != 200:
                r.failure(f"{r.status_code}: {r.text[:200]}")
