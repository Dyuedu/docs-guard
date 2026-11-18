"""Spec 003 AC-1 (forward shapes) and AC-2 (ONNX export matches serving config, parity)."""

import re
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from docguard.models.dbnet import DBNet

PBTXT = Path(__file__).resolve().parents[1] / "serving/model_repository/text_det/config.pbtxt"


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return DBNet(pretrained=False)


@pytest.mark.parametrize("n,s", [(1, 640), (2, 640), (1, 960)])
def test_forward_shapes_train_eval(model, n, s):
    x = torch.randn(n, 3, s, s)
    model.train()
    out = model(x)
    assert set(out) == {"prob_logits", "thresh", "binary"}
    for v in out.values():
        assert tuple(v.shape) == (n, 1, s, s)
    model.eval()
    with torch.no_grad():
        prob = model(x)
    assert isinstance(prob, torch.Tensor) and tuple(prob.shape) == (n, 1, s, s)
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0


def _pbtxt_io():
    text = PBTXT.read_text(encoding="utf-8")
    pat = r'\{\s*name:\s*"(\w+)",\s*data_type:\s*(\w+),\s*dims:\s*\[([^\]]*)\]\s*\}'
    (inp,), (out,) = (re.findall(pat, re.search(rf"{k}\s*\[(.*?)\n\]", text, re.S).group(1)) for k in ("input", "output"))
    to_dims = lambda s: [int(v) for v in s.split(",")]
    return (inp[0], inp[1], to_dims(inp[2])), (out[0], out[1], to_dims(out[2]))


@pytest.fixture(scope="module")
def onnx_path(model, tmp_path_factory):
    pytest.importorskip("onnx")
    from docguard.export.onnx_export import export_onnx

    (in_name, _, in_dims), (out_name, _, _) = _pbtxt_io()
    path = tmp_path_factory.mktemp("onnx") / "text_det.onnx"
    export_onnx(model.eval(), torch.randn(1, *in_dims), path, input_name=in_name, output_name=out_name, opset=17)
    return path


def test_export_onnx_matches_pbtxt(onnx_path):
    import onnx

    (in_name, in_type, in_dims), (out_name, out_type, out_dims) = _pbtxt_io()
    assert in_type == out_type == "TYPE_FP32"
    g = onnx.load(str(onnx_path)).graph
    (i,), (o,) = g.input, g.output
    assert (i.name, o.name) == (in_name, out_name)
    for vi, dims in ((i, in_dims), (o, out_dims)):
        t = vi.type.tensor_type
        assert t.elem_type == onnx.TensorProto.FLOAT
        d = t.shape.dim
        assert d[0].dim_param  # dynamic batch
        assert [x.dim_value for x in d[1:]] == dims


@pytest.mark.parametrize("n", [1, 2])
def test_onnx_parity(model, onnx_path, n):
    ort = pytest.importorskip("onnxruntime")
    (in_name, _, in_dims), _ = _pbtxt_io()
    x = np.random.default_rng(n).standard_normal((n, *in_dims)).astype(np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {in_name: x})[0]
    with torch.no_grad():
        ref = model.eval()(torch.from_numpy(x)).numpy()
    assert got.shape == ref.shape == (n, 1, *in_dims[1:])
    assert float(np.abs(got - ref).max()) < 1e-3
