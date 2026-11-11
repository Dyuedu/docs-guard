"""Triton Python-backend BLS orchestrator.

decode -> [tamper_seg on patches]  ||  [text_det -> crop -> text_rec -> KIE] -> decision -> JSON

The tampering branch and the OCR branch are issued concurrently with async BLS.
"""

import asyncio
import json
import os
import time

import numpy as np
import triton_python_backend_utils as pb_utils
import yaml

from docguard.kie.rules import OcrLine, extract_fields
from docguard.pipeline import processing as P


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


async def _infer(model: str, input_name: str, output_name: str, batch: np.ndarray, chunk: int) -> np.ndarray:
    """Send `batch` to `model` in chunks of at most `chunk`, concurrently; concatenate outputs."""
    async def one(x):
        req = pb_utils.InferenceRequest(
            model_name=model, requested_output_names=[output_name], inputs=[pb_utils.Tensor(input_name, x)]
        )
        resp = await req.async_exec()
        if resp.has_error():
            raise pb_utils.TritonModelException(f"{model}: {resp.error().message()}")
        return pb_utils.get_output_tensor_by_name(resp, output_name).as_numpy()

    parts = await asyncio.gather(*(one(np.ascontiguousarray(batch[i : i + chunk])) for i in range(0, len(batch), chunk)))
    return np.concatenate(parts, axis=0)


class TritonPythonModel:
    def initialize(self, args):
        params = json.loads(args["model_config"])["parameters"]
        with open(params["config_path"]["string_value"], encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        vocab_path = os.path.join(params["vocab_dir"]["string_value"], self.cfg["text_rec"]["vocab_file"])
        self.vocab = P.load_vocab(vocab_path)

    async def _tampering(self, image):
        t = self.cfg["tampering"]
        t0 = time.perf_counter()
        patches, coords = P.tile(image, t["patch_size"], t["stride"])
        logits = await _infer("tamper_seg", "image", "mask_logits", patches, 16)
        prob = P.stitch(_sigmoid(logits[:, 0]), coords, image.shape[:2], t["patch_size"])
        boxes = P.mask_to_boxes(prob, t["mask_threshold"], t["min_region_area"])
        area_ratio = float((prob >= t["mask_threshold"]).mean())
        score = float(np.percentile(prob, 99.9))  # robust "max": ignores isolated hot pixels
        return prob, boxes, area_ratio, score, (time.perf_counter() - t0) * 1000

    async def _ocr(self, image):
        d, r = self.cfg["text_det"], self.cfg["text_rec"]
        t0 = time.perf_counter()
        det_in, ratio = P.det_preprocess(image, d["input_size"])
        prob = (await _infer("text_det", "image", "prob", det_in, 1))[0, 0]
        boxes = P.det_postprocess(prob, ratio, d["bin_threshold"], d["box_threshold"], d["unclip_ratio"], image.shape[:2])
        lines = []
        if boxes:
            crops = P.crop_for_rec(image, boxes, r["height"], r["width"])
            logits = await _infer("text_rec", "image", "logits", crops, 64)
            for box, (text, conf) in zip(boxes, P.ctc_greedy_decode(logits, self.vocab)):
                if text.strip():
                    lines.append(OcrLine(text, tuple(box), conf))
        fields = extract_fields(lines, image.shape[0])
        return fields, (time.perf_counter() - t0) * 1000

    async def _run(self, data: bytes, request_id: str) -> dict:
        t_start = time.perf_counter()
        image, scale = P.decode_image(data, self.cfg["tampering"]["max_long_side"])
        t_pre = (time.perf_counter() - t_start) * 1000

        (_, t_boxes, area_ratio, score, t_tamper), (fields, t_ocr) = await asyncio.gather(
            self._tampering(image), self._ocr(image)
        )

        overlaps = sorted(
            name for name, f in fields.items() if f.bbox and any(P.box_intersects(b, f.bbox) for b in t_boxes)
        )
        is_authentic, reason = P.decide(score, area_ratio, self.cfg["decision"])
        inv = 1.0 / scale  # report boxes in the client's original pixel coordinates

        return {
            "request_id": request_id,
            "is_authentic": is_authentic,
            "decision_reason": reason,
            "tampering_result": {
                "tampered_score": round(score, 4),
                "tampered_area_ratio": round(area_ratio, 6),
                "tampered_bboxes": [[int(round(v * inv)) for v in b] for b in t_boxes],
                "overlaps_fields": overlaps,
            },
            "extracted_fields": {
                name: {"value": f.value, "confidence": round(f.confidence, 4)} for name, f in fields.items()
            },
            "latency_ms": {
                "preprocess": round(t_pre, 2),
                "tampering": round(t_tamper, 2),
                "ocr": round(t_ocr, 2),
                "total": round((time.perf_counter() - t_start) * 1000, 2),
            },
        }

    async def execute(self, requests):
        responses = []
        for request in requests:
            try:
                data = pb_utils.get_input_tensor_by_name(request, "image_bytes").as_numpy().tobytes()
                rid_t = pb_utils.get_input_tensor_by_name(request, "request_id")
                rid = rid_t.as_numpy().reshape(-1)[0].decode() if rid_t is not None else ""
                result = await self._run(data, rid)
                out = pb_utils.Tensor("result_json", np.array([json.dumps(result, ensure_ascii=False).encode()], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
            except Exception as e:  # return a per-request error instead of failing the whole batch
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(str(e))))
        return responses
