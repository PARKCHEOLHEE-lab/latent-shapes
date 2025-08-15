import os
import sys
import torch

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/08-02-2025__17-36-23/"))
if basedir not in sys.path:
    sys.path.append(basedir)

from src.config import Configuration
from src.model import SDFDecoder, LatentShapes


def main():
    configuration = Configuration()
    configuration.set_seed()

    states = torch.load(os.path.join(basedir, configuration.SAVE_NAME))

    latent_shapes = LatentShapes(
        latent_shapes=torch.rand(size=(configuration.SLICER, configuration.NUM_LATENT_SHAPE_VERTICES, 3))
    )
    latent_shapes.load_state_dict(states["state_dict_latent_shapes"])

    sdf_decoder = SDFDecoder(configuration=configuration)
    sdf_decoder.load_state_dict(states["state_dict_decoder"])
    
    test_dir = os.path.join(basedir, "test")
    os.makedirs(test_dir, exist_ok=True)
    
    sdf_decoder.reconstruct(
        latent_shapes.embedding.to(configuration.DEVICE),
        save_path=test_dir,
        check_watertight=False,
        map_z_to_y=False,
        add_noise=False,
        rescale=True,
    )
    

if __name__ == "__main__":
    main()