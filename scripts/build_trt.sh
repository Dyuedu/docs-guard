#!/usr/bin/env bash
# Build TensorRT engines for variants B2/B3 inside the Triton container
# (same TensorRT version as the server, otherwise the .plan will not load).
#
#   docker compose run --rm triton bash /workspace/scripts/build_trt.sh
set -euo pipefail

ART=/workspace/artifacts
REPO=/models

build() {  # name, onnx, precision flag, min/opt/max shapes
  local name=$1 onnx=$2 prec=$3 shapes=$4
  trtexec --onnx="$ART/$onnx" --saveEngine="$REPO/$name/1/model.plan" $prec $shapes \
          --memPoolSize=workspace:2048 --skipInference
}

# Tampering: keep FP16 by default (forensic signal is sensitive to INT8, see B5).
build tamper_seg tamper_seg.onnx "--fp16" \
  "--minShapes=image:1x3x512x512 --optShapes=image:8x3x512x512 --maxShapes=image:16x3x512x512"

# INT8 for OCR uses the QDQ graph produced by docguard.export.onnx_export.quantize_int8_static.
build text_det text_det.int8.onnx "--int8 --fp16" \
  "--minShapes=image:1x3x960x960 --optShapes=image:4x3x960x960 --maxShapes=image:8x3x960x960"
build text_rec text_rec.int8.onnx "--int8 --fp16" \
  "--minShapes=image:1x3x32x320 --optShapes=image:32x3x32x320 --maxShapes=image:64x3x32x320"
