import os
import sys

import numpy as np
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


def test_onnx_matches_pytorch(tmp_path):
    from export_onnx import export_decoder_onnx

    cfg, decoder = _load_trained_decoder()
    input_dim = (cfg.NUM_LATENT_SHAPE_VERTICES + 1) * 3  # 297

    onnx_path = str(tmp_path / "decoder.onnx")
    export_decoder_onnx(runs_dir=RUNS_DIR, out_path=onnx_path)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    torch.manual_seed(0)
    for n in (1, 64):  # two batch sizes -> also proves the dynamic batch axis
        x = torch.rand(n, input_dim, dtype=torch.float32)
        with torch.inference_mode():
            reference = decoder.forward(x).cpu().numpy()
        got = session.run(None, {input_name: x.numpy()})[0]

        assert got.shape == (n, 1), f"N={n}: got shape {got.shape}"
        max_abs_diff = float(np.abs(got - reference).max())
        assert np.allclose(got, reference, atol=1e-4), (
            f"N={n}: onnx vs pytorch max|diff|={max_abs_diff:.2e} exceeds atol=1e-4"
        )
