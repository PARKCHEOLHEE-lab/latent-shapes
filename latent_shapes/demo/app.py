import os
import sys

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List

from latent_shapes.src.config import Configuration
from latent_shapes.src.data import SDFDataset
from latent_shapes.src.model import SDFDecoder, LatentShapes


app = FastAPI(title="latent-shapes")


configuration = Configuration()
configuration.set_seed()

sdf_dataset = SDFDataset.create_dataset(
    data_dir=configuration.DATA_PATH_PROCESSED, configuration=configuration, data_slicer=10
)

latent_shapes = LatentShapes(latent_shapes=sdf_dataset.latent_shapes, noise_min=-0.1, noise_max=0.1)

sdf_decoder = SDFDecoder(configuration=configuration)

states = torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/07-13-2025__13-15-20/states.pth")))

sdf_decoder.load_state_dict(states["state_dict_decoder"])
latent_shapes.load_state_dict(states["state_dict_latent_shapes"])


class ReconstructRequest(BaseModel):
    latent_shapes: List[List[float]]
    rescale: bool
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
    random_index = torch.randint(0, sdf_dataset.latent_shapes.shape[0], (1,))
    latent_shape = latent_shapes(random_index).squeeze(0)
    faces = sdf_dataset.faces[random_index].squeeze(0)

    return {"latent_shape": latent_shape.tolist(), "faces": faces.tolist()}


@app.post("/api/reconstruct")
def reconstruct(request: ReconstructRequest):
    try:
        configuration.RECONSTRUCTION_GRID_SIZE = request.resolution

        latent_shapes_tensor = torch.tensor(request.latent_shapes).to(configuration.DEVICE)

        reconstruction_results = sdf_decoder.reconstruct(
            latent_shapes=latent_shapes_tensor.unsqueeze(0),
            save_path=os.path.join(os.path.dirname(__file__)),
            normalize=True,
            check_watertight=False,
            map_z_to_y=False,
            add_noise=False,
            rescale=request.rescale,
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
    uvicorn.run(app, host="0.0.0.0", port=7777)
