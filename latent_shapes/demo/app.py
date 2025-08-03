import os
import sys
import torch
import uvicorn

from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/08-02-2025__17-36-23/"))
if basedir not in sys.path:
    sys.path.append(basedir)

from src.config import Configuration
from src.model import SDFDecoder, LatentShapes


app = FastAPI(title="latent-shapes")


configuration = Configuration()
configuration.set_seed()

states = torch.load(os.path.join(basedir, "states.pth"))

latent_shapes = LatentShapes(latent_shapes=torch.rand(size=(configuration.SLICER, configuration.NUM_LATENT_SHAPE_VERTICES, 3)))
latent_shapes.load_state_dict(states["state_dict_latent_shapes"])

sdf_decoder = SDFDecoder(configuration=configuration)
sdf_decoder.load_state_dict(states["state_dict_decoder"])


class ReconstructRequest(BaseModel):
    latent_shapes: List[List[float]]
    rescale: bool
    normalize: bool
    map_z_to_y: bool
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
    random_index = torch.randint(0, configuration.SLICER, (1,))
    latent_shape = latent_shapes(random_index).squeeze(0)
    faces = configuration.BOX.faces
    
    # map y to z to match the loaded latent shape into the xzy system
    latent_shape[:, [1, 2]] = latent_shape[:, [2, 1]]

    return {"latent_shape": latent_shape.tolist(), "faces": faces.tolist()}


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
            normalize=request.normalize,
            check_watertight=False,
            map_z_to_y=request.map_z_to_y,
            add_noise=False,
            rescale=request.rescale,
            # centraize=False
        )

        if reconstruction_results[0] is None:
            raise HTTPException(status_code=400, detail="Reconstruction failed")

        # Extract mesh data from the first result
        mesh = reconstruction_results[0]

        vertices = mesh.vertices.tolist()
        faces = mesh.faces.tolist()

        return {"message": "Reconstruction successful", "vertices": vertices, "faces": faces}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Reconstruction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7777, reload=True)
