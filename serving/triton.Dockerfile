# Triton + the docguard package so the Python BLS model can import shared pre/post-processing.
# Pin the tag: TensorRT engines (.plan) must be built with the same TensorRT version (scripts/build_trt.sh).
FROM nvcr.io/nvidia/tritonserver:24.08-py3

WORKDIR /workspace
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN pip install --no-cache-dir . && pip uninstall -y opencv-python 2>/dev/null || true
