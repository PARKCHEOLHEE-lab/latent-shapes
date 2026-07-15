import inspect
import os
import sys

import torch


def export_decoder_onnx(runs_dir, out_path, opset=17):
    runs_dir = os.path.abspath(runs_dir)
    if runs_dir not in sys.path:
        sys.path.insert(0, runs_dir)

    from src.config import Configuration
    from src.model import SDFDecoder

    cfg = Configuration()
    states = torch.load(
        os.path.join(runs_dir, cfg.SAVE_NAME), map_location="cpu", weights_only=False
    )
    decoder = SDFDecoder(cfg).to("cpu")
    decoder.load_state_dict(states["state_dict_decoder"])
    decoder.eval()

    input_dim = (cfg.NUM_LATENT_SHAPE_VERTICES + 1) * 3
    dummy = torch.rand(1, input_dim, dtype=torch.float32)

    export_kwargs = dict(
        input_names=["cxyz"],
        output_names=["sdf"],
        dynamic_axes={"cxyz": {0: "n"}, "sdf": {0: "n"}},
        opset_version=opset,
    )
    # torch >= 2.6 defaults to the dynamo exporter (needs onnxscript and treats
    # dynamic_axes differently). Force the legacy torchscript exporter so the
    # artifact is identical whether produced here (torch 2.13) or in the
    # devcontainer (torch 2.1, where the dynamo kwarg does not exist yet).
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(decoder, dummy, out_path, **export_kwargs)
    return out_path


if __name__ == "__main__":
    _demo_dir = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.abspath(os.path.join(_demo_dir, "..", ".."))
    _runs = os.path.join(_repo, "latent_shapes", "runs", "08-02-2025__17-36-23")
    _out = os.path.join(_repo, "docs", "models", "decoder.onnx")
    os.makedirs(os.path.dirname(_out), exist_ok=True)
    export_decoder_onnx(runs_dir=_runs, out_path=_out)
    print(f"wrote {_out}")
