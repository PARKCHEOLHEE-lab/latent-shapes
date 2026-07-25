import json
import os
import sys

import numpy as np
import onnx
import torch
import onnxruntime as ort

HERE = os.path.dirname(__file__)
DEMO_DIR = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(DEMO_DIR, "..", ".."))
RUNS_DIR = os.path.join(REPO, "latent_shapes", "runs", "08-02-2025__17-36-23")

# import the export module under test
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)
# import the trained-time src (mirrors app.py's basedir on sys.path)
if RUNS_DIR not in sys.path:
    sys.path.insert(0, RUNS_DIR)

from src.config import Configuration  # noqa: E402  (trained-time config)
from src.model import SDFDecoder  # noqa: E402


def _load_trained_decoder():
    # no Configuration.set_seed(): it calls torch.cuda.initial_seed() which throws
    # on CPU-only torch. load_state_dict overwrites the random init anyway.
    cfg = Configuration()
    states = torch.load(
        os.path.join(RUNS_DIR, cfg.SAVE_NAME), map_location="cpu", weights_only=False
    )
    decoder = SDFDecoder(cfg).to("cpu")
    decoder.load_state_dict(states["state_dict_decoder"])
    decoder.eval()
    return cfg, decoder


def test_onnx_uses_balanced_mixed_precision_and_matches_pytorch(tmp_path):
    from export_onnx import export_decoder_onnx

    cfg, decoder = _load_trained_decoder()
    input_dim = (cfg.NUM_LATENT_SHAPE_VERTICES + 1) * 3  # 297

    onnx_path = str(tmp_path / "decoder.onnx")
    export_decoder_onnx(runs_dir=RUNS_DIR, out_path=onnx_path)

    model = onnx.load(onnx_path)
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    initializer_types = {initializer.data_type for initializer in model.graph.initializer}
    assert onnx.TensorProto.FLOAT16 in initializer_types
    assert onnx.TensorProto.FLOAT in initializer_types
    assert os.path.getsize(onnx_path) < 15_000_000

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with open(os.path.join(REPO, "docs", "data", "latent_shapes.json")) as handle:
        cages = np.asarray(json.load(handle)["latent_shapes"], dtype=np.float32)
    rng = np.random.default_rng(777)
    bounds_min = np.asarray([-0.614935, -0.767495, -0.666214], dtype=np.float32)
    bounds_max = np.asarray([0.644344, 0.792882, 0.581235], dtype=np.float32)
    rows = []
    for cage_index in (0, 12, 24, 36, 46):
        xyz = rng.uniform(bounds_min, bounds_max, size=(512, 3)).astype(np.float32)
        cage = np.broadcast_to(cages[cage_index].reshape(1, -1), (len(xyz), input_dim - 3))
        rows.append(np.concatenate([xyz, cage], axis=1))
    x = np.concatenate(rows, axis=0).astype(np.float32)

    with torch.inference_mode():
        reference = decoder.forward(torch.from_numpy(x)).cpu().numpy()
    got = session.run(None, {input_name: x})[0]

    assert got.shape == reference.shape == (len(x), 1)
    error = np.abs(got - reference)
    near_surface = np.abs(reference) <= 0.02
    sign_mismatch = np.signbit(got) != np.signbit(reference)
    assert int(near_surface.sum()) >= 20
    assert float(error.mean()) < 4e-4
    assert float(error.max()) < 2e-3
    assert float(sign_mismatch[near_surface].mean()) < 0.05
