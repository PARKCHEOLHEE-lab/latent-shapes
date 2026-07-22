import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(__file__)
DEMO_DIR = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(DEMO_DIR, "..", ".."))
RUNS_DIR = os.path.join(REPO, "latent_shapes", "runs", "08-02-2025__17-36-23")

if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)
if RUNS_DIR not in sys.path:
    sys.path.insert(0, RUNS_DIR)

from src.config import Configuration  # noqa: E402
from src.model import LatentShapes  # noqa: E402


def _load_reference():
    cfg = Configuration()
    states = torch.load(
        os.path.join(RUNS_DIR, cfg.SAVE_NAME), map_location="cpu", weights_only=False
    )
    latent = LatentShapes(
        latent_shapes=torch.rand(cfg.SLICER, cfg.NUM_LATENT_SHAPE_VERTICES, 3)
    )
    latent.load_state_dict(states["state_dict_latent_shapes"])
    embedding = latent.embedding.detach().cpu().numpy()  # (50, 98, 3), raw xyz
    faces = np.asarray(cfg.BOX.faces)  # (192, 3)
    return cfg, embedding, faces


def test_latent_shapes_json_matches_checkpoint(tmp_path):
    from export_latent_shapes import export_latent_shapes

    cfg, embedding, faces = _load_reference()

    out = str(tmp_path / "latent_shapes.json")
    export_latent_shapes(runs_dir=RUNS_DIR, out_path=out)

    with open(out) as f:
        data = json.load(f)

    cages = np.asarray(data["latent_shapes"], dtype=np.float32)
    got_faces = np.asarray(data["faces"], dtype=np.int64)

    assert cages.shape == (cfg.SLICER, cfg.NUM_LATENT_SHAPE_VERTICES, 3), (
        f"cages shape {cages.shape} != (50, 98, 3)"
    )
    max_abs_diff = float(np.abs(cages - embedding).max())
    assert np.allclose(cages, embedding, atol=1e-6), (
        f"cage values differ from checkpoint embedding: max|diff|={max_abs_diff:.2e}"
    )
    assert got_faces.shape == faces.shape, f"faces shape {got_faces.shape} != {faces.shape}"
    assert np.array_equal(got_faces, faces), "faces differ from config.BOX.faces"
