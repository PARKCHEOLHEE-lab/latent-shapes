import os
import sys
import torch
import trimesh

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../runs/08-02-2025__17-36-23/"))
if basedir not in sys.path:
    sys.path.append(basedir)

from src.config import Configuration
from src.model import SDFDecoder, LatentShapes


def main():
    configuration = Configuration()
    configuration.RECONSTRUCTION_GRID_SIZE = 128
    configuration.set_seed(77777)

    states = torch.load(os.path.join(basedir, configuration.SAVE_NAME))

    latent_shapes = LatentShapes(
        latent_shapes=torch.rand(size=(configuration.SLICER, configuration.NUM_LATENT_SHAPE_VERTICES, 3))
    )
    latent_shapes.load_state_dict(states["state_dict_latent_shapes"])

    sdf_decoder = SDFDecoder(configuration=configuration)
    sdf_decoder.load_state_dict(states["state_dict_decoder"])
    
    interpolate_dir = os.path.join(basedir, "interpolate")
    os.makedirs(interpolate_dir, exist_ok=True)
    
    indices_to_interpolate = [
        [2, 5],
        [26, 31],
        [46, 11],
        [22, 33],
    ]
    
    ratios = [
        [0.50, 0.50],
        [0.30, 0.70],
        [0.60, 0.40],
        [0.45, 0.55],
    ]
        
    for ratio, indices in zip(ratios, indices_to_interpolate):
        
        shape_1, shape_2 = latent_shapes(indices)
        ratio_1, ratio_2 = ratio
    
        sdf_decoder.reconstruct(
            shape_1.unsqueeze(0).to(configuration.DEVICE),
            save_path=interpolate_dir,
            check_watertight=False,
            map_z_to_y=False,
            add_noise=False,
            rescale=True,
            additional_title=f"shape_{indices[0]}"
        )
        
        shape_1_mesh = trimesh.Trimesh(vertices=shape_1.detach().numpy(), faces=configuration.BOX.faces)
        shape_1_mesh.export(os.path.join(interpolate_dir, f"latent_shape_{indices[0]}.obj"))
    
        sdf_decoder.reconstruct(
            shape_2.unsqueeze(0).to(configuration.DEVICE),
            save_path=interpolate_dir,
            check_watertight=False,
            map_z_to_y=False,
            add_noise=False,
            rescale=True,
            additional_title=f"shape_{indices[1]}"
        )
        
        shape_2_mesh = trimesh.Trimesh(vertices=shape_2.detach().numpy(), faces=configuration.BOX.faces)
        shape_2_mesh.export(os.path.join(interpolate_dir, f"latent_shape_{indices[1]}.obj"))
    
        sdf_decoder.reconstruct(
            (shape_1 * ratio_1 + shape_2 * ratio_2).unsqueeze(0).to(configuration.DEVICE),
            save_path=interpolate_dir,
            check_watertight=False,
            map_z_to_y=False,
            add_noise=False,
            rescale=True,
            additional_title=f"shape_{indices[0]}_{indices[1]}_{str(ratio_1)}_{str(ratio_2)}".replace(".", "_")
        )
        
        shape_2_mesh = trimesh.Trimesh(vertices=(shape_1 * ratio_1 + shape_2 * ratio_2).detach().numpy(), faces=configuration.BOX.faces)
        shape_2_mesh.export(os.path.join(interpolate_dir, f"shape_{indices[0]}_{indices[1]}_interpolated.obj"))
    

if __name__ == "__main__":
    main()