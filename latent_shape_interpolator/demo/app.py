import os
import sys

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from latent_shape_interpolator.src.config import Configuration
from latent_shape_interpolator.src.data import SDFDataset
from latent_shape_interpolator.src.model import SDFDecoder


app = Flask(__name__)
CORS(app)

configuration = Configuration()

sdf_dataset = SDFDataset.create_dataset(
    data_dir=configuration.DATA_PATH_PROCESSED, configuration=configuration, data_slicer=10
)

sdf_decoder = SDFDecoder(
    latent_shapes=sdf_dataset.latent_shapes,
    configuration=configuration,
)

sdf_decoder.load_state_dict(
    torch.load(os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/06-23-2025__21-24-31/states.pth")))[
        "state_dict_model"
    ]
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/latent_shapes", methods=["GET"])
def get_random_latent_shape():
    random_index = torch.randint(0, sdf_dataset.latent_shapes.shape[0], (1,))
    latent_shape = sdf_dataset.latent_shapes[random_index].squeeze(0)
    faces = sdf_dataset.faces[random_index].squeeze(0)

    return jsonify({"latent_shape": latent_shape.tolist(), "faces": faces.tolist()})


@app.route("/api/reconstruct", methods=["POST"])
def reconstruct():
    try:
        data = request.get_json()
        latent_shapes = torch.tensor(data["latent_shapes"]).to(configuration.DEVICE).unsqueeze(0)

        reconstruction_results = sdf_decoder.reconstruct(
            latent_shapes=latent_shapes,
            save_path=os.path.join(os.path.dirname(__file__)),
            normalize=True,
            check_watertight=False,
            map_z_to_y=False,
            add_noise=False,
            rescale=True,
        )

        if reconstruction_results[0] is None:
            return jsonify({"message": "Reconstruction failed"}), 400

        # Extract mesh data from the first result
        mesh = reconstruction_results[0]

        vertices = mesh.vertices.tolist()
        faces = mesh.faces.tolist()

        return jsonify({"message": "Reconstruction successful", "vertices": vertices, "faces": faces})

    except Exception as e:
        return jsonify({"message": f"Reconstruction failed: {str(e)}"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
