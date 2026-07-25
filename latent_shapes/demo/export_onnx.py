import inspect
import os
import sys

import onnx
import torch
from onnxconverter_common.float16 import convert_float_to_float16


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

    model = onnx.load(out_path)
    decoder_nodes = [
        node for node in model.graph.node
        if node.name.startswith("/blocks.1/")
    ]
    fp32_tail = [node.name for node in decoder_nodes[-10:]]
    model = convert_float_to_float16(
        model,
        keep_io_types=True,
        node_block_list=fp32_tail,
    )
    # onnxconverter-common 1.16 changes the public output to FP16 when the
    # final node is blocked. Keep the worker contract and final SDF in FP32.
    if model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
        model.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        output_name = model.graph.output[0].name
        for node in model.graph.node:
            if node.op_type == "Cast" and output_name in node.output:
                for attribute in node.attribute:
                    if attribute.name == "to":
                        attribute.i = onnx.TensorProto.FLOAT
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    return out_path


if __name__ == "__main__":
    _demo_dir = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.abspath(os.path.join(_demo_dir, "..", ".."))
    _runs = os.path.join(_repo, "latent_shapes", "runs", "08-02-2025__17-36-23")
    _out = os.path.join(_repo, "docs", "models", "decoder.onnx")
    os.makedirs(os.path.dirname(_out), exist_ok=True)
    export_decoder_onnx(runs_dir=_runs, out_path=_out)
    print(f"wrote {_out}")
