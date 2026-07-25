import json
import os
import sys
import torch
import uvicorn

from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/08-02-2025__17-36-23/"))
if basedir not in sys.path:
    sys.path.append(basedir)

from src.config import Configuration
from src.model import SDFDecoder, LatentShapes

from reconstruct_stream import reconstruct_adaptive


app = FastAPI(title="latent-shapes")


# serve the shared UI assets (PT Sans fonts + favicon) that interpolator.html references,
# so the local demo looks exactly like the static demo. Both read the same files in docs/,
# keeping one source of truth for the fonts.
_assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../docs"))
app.mount("/fonts", StaticFiles(directory=os.path.join(_assets_dir, "fonts")), name="fonts")
app.mount("/js", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "js")), name="js")


@app.get("/favicon.png")
def favicon():
    return FileResponse(os.path.join(_assets_dir, "favicon.png"))


configuration = Configuration()
configuration.set_seed()

states = torch.load(
    os.path.join(basedir, configuration.SAVE_NAME),
    map_location=configuration.DEVICE,
    weights_only=False,  # trusted local checkpoint; it stores non-tensor objects (e.g. trimesh.Trimesh)
)

latent_shapes = LatentShapes(
    latent_shapes=torch.rand(size=(configuration.SLICER, configuration.NUM_LATENT_SHAPE_VERTICES, 3))
)
latent_shapes.load_state_dict(states["state_dict_latent_shapes"])

sdf_decoder = SDFDecoder(configuration=configuration)
sdf_decoder.load_state_dict(states["state_dict_decoder"])

# Keep this independent from reconstruct_stream.CHUNK_ROWS: 32K is the measured
# MPS inference optimum, while 64K keeps the progressive preview cadence smooth.
_STREAM_INFERENCE_BATCH_ROWS = 32768


class ReconstructRequest(BaseModel):
    latent_shapes: List[List[float]]
    rescale: bool
    map_z_to_y: bool
    ensure_watertight: bool
    resolution: int


@app.get("/")
def index():
    return RedirectResponse(url="/interpolator.html")


@app.get("/interpolator.html")
def interpolator():
    with open(os.path.join(os.path.dirname(__file__), "templates/interpolator.html"), "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/api/latent_shapes")
def get_random_latent_shape():
    random_index_max = latent_shapes.embedding.shape[0]
    random_index_selected = torch.randint(0, random_index_max, (1,))
    latent_shape = latent_shapes(random_index_selected).squeeze(0)
    faces = configuration.BOX.faces

    # map y to z to match the loaded latent shape into the xzy system
    latent_shape[:, [1, 2]] = latent_shape[:, [2, 1]]

    return {"latent_shape": latent_shape.tolist(), "faces": faces.tolist(), "index": f"{random_index_selected.item()}/{random_index_max - 1}"}


@app.post("/api/reconstruct")
def reconstruct(request: ReconstructRequest):
    try:
        configuration.RECONSTRUCTION_GRID_SIZE = request.resolution

        latent_shapes_tensor = torch.tensor(request.latent_shapes).to(configuration.DEVICE)

        # map z to y to match the loaded latent shape into the xyz system
        latent_shapes_tensor[:, [1, 2]] = latent_shapes_tensor[:, [2, 1]]

        reconstruction_results = sdf_decoder.reconstruct(
            latent_shapes=latent_shapes_tensor.unsqueeze(0),
            save_path=os.path.join(os.path.dirname(__file__)),
            check_watertight=request.ensure_watertight,
            map_z_to_y=request.map_z_to_y,
            add_noise=False,
            rescale=request.rescale,
        )

        if reconstruction_results[0] is None:
            raise HTTPException(status_code=400, detail="Reconstruction failed")

        # Extract mesh data from the first result
        mesh = reconstruction_results[0]

        vertices = mesh.vertices.tolist()
        faces = mesh.faces.tolist()
        edges = mesh.edges.tolist()

        return {
            "message": "Reconstruction successful",
            "vertices": vertices,
            "faces": faces,
            "edges": edges,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reconstruction failed: {str(e)}")


def _make_stream_decoder(cage_xyz):
    # cage_xyz: the latent cage in decoder (xyz) space (y/z already swapped from the body).
    # Returns decoder(xyz[N,3]) -> sdf[N], chunked to bound peak memory (cf. model.py:198).
    cage_flat = torch.tensor(cage_xyz, dtype=torch.float32, device=configuration.DEVICE).reshape(1, -1)

    def decoder(xyz_np):
        xyz = torch.as_tensor(xyz_np, dtype=torch.float32, device=configuration.DEVICE)
        out = []
        with torch.inference_mode():
            for chunk in xyz.split(_STREAM_INFERENCE_BATCH_ROWS):
                cxyz = torch.cat([chunk, cage_flat.expand(chunk.shape[0], -1)], dim=1)
                out.append(sdf_decoder.forward(cxyz).squeeze(-1))
        return torch.cat(out).cpu().numpy()

    return decoder


@app.post("/api/reconstruct/stream")
def reconstruct_stream(request: ReconstructRequest):
    # Coarse-to-fine streaming: emit one Server-Sent Event per refinement level (coarse
    # first) so the browser reveals the mesh as it sharpens, matching the static demo.
    # ensure_watertight is intentionally not applied here (the static demo has no watertight
    # step); use /api/reconstruct for a single watertight mesh.
    cage_xyz = [[p[0], p[2], p[1]] for p in request.latent_shapes]  # xzy body -> xyz (cf. app.py:94)
    decoder = _make_stream_decoder(cage_xyz)

    def event_stream():
        for event in reconstruct_adaptive(
            cage_xyz, request.resolution, decoder,
            rescale=request.rescale, map_z_to_y=request.map_z_to_y, adaptive=True,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7777, reload=True)
