"""Export PyTorch models to ONNX and produce FP16 / INT8 variants (proposal §6, B1–B5).

TensorRT engines (B2–B5) are built from these ONNX files with trtexec, see
scripts/build_trt.sh. ONNX Runtime INT8 here is used for CPU sanity checks and
for comparing quantization error before committing to a TensorRT build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import numpy as np


def export_onnx(
    model, dummy_input, path: str | Path, input_name: str, output_name: str,
    dynamic_batch: bool = True, opset: int = 17,
) -> Path:
    import torch

    model.eval()
    dynamic_axes = {input_name: {0: "batch"}, output_name: {0: "batch"}} if dynamic_batch else None
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, str(path), input_names=[input_name], output_names=[output_name],
            dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True,
        )
    import onnx

    onnx.checker.check_model(str(path))
    return Path(path)


def to_fp16(src: str | Path, dst: str | Path) -> Path:
    import onnx
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(model, str(dst))
    return Path(dst)


class _Reader:
    """onnxruntime CalibrationDataReader over an iterator of preprocessed batches."""

    def __init__(self, input_name: str, batches: Iterator[np.ndarray]):
        self.input_name, self._it = input_name, iter(batches)

    def get_next(self):
        b = next(self._it, None)
        return None if b is None else {self.input_name: b}


def quantize_int8_static(
    src: str | Path, dst: str | Path, input_name: str, calib_batches: Callable[[], Iterator[np.ndarray]],
    per_channel: bool = True,
) -> Path:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    quantize_static(
        str(src), str(dst), _Reader(input_name, calib_batches()),
        quant_format=QuantFormat.QDQ,  # QDQ graphs are consumable by TensorRT as explicit quantization
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        per_channel=per_channel, calibrate_method=CalibrationMethod.Entropy,
    )
    return Path(dst)


def max_abs_diff(a: str | Path, b: str | Path, inputs: dict[str, np.ndarray]) -> float:
    """Output parity check between two ONNX files (used in tests and after each export)."""
    import onnxruntime as ort

    outs = [ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, inputs)[0] for p in (a, b)]
    return float(np.abs(outs[0].astype(np.float32) - outs[1].astype(np.float32)).max())


def file_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / 2**20
