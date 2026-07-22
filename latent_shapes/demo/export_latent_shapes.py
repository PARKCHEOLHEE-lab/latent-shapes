import json
import os
import sys

import torch


def export_latent_shapes(runs_dir, out_path):
    runs_dir = os.path.abspath(runs_dir)
    if runs_dir not in sys.path:
        sys.path.insert(0, runs_dir)

    from src.config import Configuration
    from src.model import LatentShapes

    cfg = Configuration()
    states = torch.load(
        os.path.join(runs_dir, cfg.SAVE_NAME), map_location="cpu", weights_only=False
    )
    latent = LatentShapes(
        latent_shapes=torch.rand(cfg.SLICER, cfg.NUM_LATENT_SHAPE_VERTICES, 3)
    )
    latent.load_state_dict(states["state_dict_latent_shapes"])

    # raw xyz embedding, (50, 98, 3); the browser mirrors app.py's y/z swap itself
    cages = latent.embedding.detach().cpu().numpy().tolist()
    faces = cfg.BOX.faces.tolist()

    with open(out_path, "w") as f:
        json.dump({"latent_shapes": cages, "faces": faces}, f)
    return out_path


if __name__ == "__main__":
    _demo_dir = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.abspath(os.path.join(_demo_dir, "..", ".."))
    _runs = os.path.join(_repo, "latent_shapes", "runs", "08-02-2025__17-36-23")
    _out = os.path.join(_repo, "docs", "data", "latent_shapes.json")
    os.makedirs(os.path.dirname(_out), exist_ok=True)
    export_latent_shapes(runs_dir=_runs, out_path=_out)
    print(f"wrote {_out}")
