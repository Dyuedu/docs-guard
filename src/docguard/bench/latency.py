"""Model-level latency benchmark (proposal §6). End-to-end load tests live in loadtest/.

Usage:
    python -m docguard.bench.latency --onnx artifacts/text_rec.int8.onnx --shape 8,3,32,320 \
        --provider CUDAExecutionProvider --variant B3 --mlflow
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np


def summarize(samples_ms: list[float]) -> dict[str, float]:
    a = np.asarray(samples_ms)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
    }


def measure(fn: Callable[[], object], warmup: int, iterations: int, sync: Callable[[], None] | None = None) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()  # torch.cuda.synchronize for PyTorch baselines; ORT run() is already blocking
        times.append((time.perf_counter() - t0) * 1000)
    return summarize(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--shape", required=True, help="comma-separated input shape, e.g. 1,3,512,512")
    ap.add_argument("--provider", default="CUDAExecutionProvider")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--variant", default="")
    ap.add_argument("--mlflow", action="store_true")
    args = ap.parse_args()

    import onnxruntime as ort

    sess = ort.InferenceSession(args.onnx, providers=[args.provider, "CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.random.rand(*[int(s) for s in args.shape.split(",")]).astype(np.float32)
    stats = measure(lambda: sess.run(None, {inp.name: x}), args.warmup, args.iterations)
    stats["batch"] = x.shape[0]
    print(stats)

    if args.mlflow:
        import mlflow

        from docguard.export.onnx_export import file_size_mb

        with mlflow.start_run(run_name=f"{args.variant}-{args.onnx}"):
            mlflow.log_params({"variant": args.variant, "model": args.onnx, "provider": args.provider, "shape": args.shape})
            mlflow.log_metrics({**stats, "file_size_mb": file_size_mb(args.onnx)})


if __name__ == "__main__":
    main()
