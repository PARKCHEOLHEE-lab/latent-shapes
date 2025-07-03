import os
import sys
import torch
import skimage
import trimesh
import numpy as np
import torch.nn as nn
import point_cloud_utils as pcu

from typing import Optional

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from latent_shape_interpolator.src.config import Configuration


class LatentShapes(nn.Module):
    def __init__(self, latent_shapes: torch.Tensor, noise_min: Optional[float] = None, noise_max: Optional[float] = None):
        super().__init__()

        self.noise = torch.zeros_like(latent_shapes)
        if None not in (noise_min, noise_max):
            # add noise to latent points
            self.noise = noise_min + torch.rand_like(latent_shapes) * (noise_max - noise_min)

            # check if the noise range is valid
            assert torch.all(self.noise >= noise_min)
            assert torch.all(self.noise <= noise_max)

        # initialize latent points with noise
        self.embedding = nn.Parameter(latent_shapes + self.noise)

    def forward(self, class_number: torch.Tensor) -> torch.Tensor:
        return self.embedding[class_number]


class ResidualLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        
        self.linear_1 = nn.Linear(self.in_dim, self.in_dim)
        self.linear_2 = nn.Linear(self.in_dim, self.out_dim)
        
    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        
        x = self.linear_1(_x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.activation(x)
        
        return x + _x

class SDFDecoder(nn.Module):
    def __init__(self, latent_shapes: torch.Tensor, configuration: Configuration):
        super().__init__()

        self.latent_shapes = latent_shapes
        self.configuration = configuration

        # define latent shapes embedding
        self.latent_shapes_embedding = LatentShapes(latent_shapes=self.latent_shapes)

        self.main_1_in_features = (self.configuration.NUM_LATENT_POINTS + 1) * 3

        self.main_1 = nn.Sequential(
            ResidualLinear(self.main_1_in_features, self.main_1_in_features, nn.ReLU(True)),
            ResidualLinear(self.main_1_in_features, self.main_1_in_features, nn.ReLU(True)),
            ResidualLinear(self.main_1_in_features, self.main_1_in_features, nn.ReLU(True)),
            ResidualLinear(self.main_1_in_features, self.main_1_in_features, nn.ReLU(True)),
            ResidualLinear(self.main_1_in_features, self.main_1_in_features, nn.ReLU(True)),
            nn.Linear(self.main_1_in_features, self.main_1_in_features, nn.ReLU(True)),
            # nn.Linear(self.main_1_in_features, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            # nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            # nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            # nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            # nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            # nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            # nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
        )

        # self.main_2_in_features = self.main_1_in_features + self.configuration.HIDDEN_DIM
        self.main_2_in_features = self.main_1_in_features * 2
        self.main_2 = nn.Sequential(
            ResidualLinear(self.main_2_in_features, self.main_2_in_features, nn.ReLU(True)),
            ResidualLinear(self.main_2_in_features, self.main_2_in_features, nn.ReLU(True)),
            ResidualLinear(self.main_2_in_features, self.main_2_in_features, nn.ReLU(True)),
            ResidualLinear(self.main_2_in_features, self.main_2_in_features, nn.ReLU(True)),
            ResidualLinear(self.main_2_in_features, self.main_2_in_features, nn.ReLU(True)),
            # nn.Linear(self.main_2_in_features, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            # nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            # nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            # nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            # nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            # nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            # getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
            nn.Linear(self.main_2_in_features, 1),
            nn.Tanh(),
        )

        self.to(self.configuration.DEVICE)

    def forward(self, cxyz: torch.Tensor):
        x = self.main_1(cxyz)
        x = torch.cat((x, cxyz), dim=1)
        x = self.main_2(x)

        return x

    @torch.no_grad()
    def reconstruct(
        self,
        latent_shapes: torch.Tensor,
        save_path: str,
        normalize: bool = True,
        check_watertight: bool = False,
        map_z_to_y: bool = True,
        add_noise: bool = False,
        rescale: bool = False,
        epoch: Optional[int] = None,
    ):
        self.eval()

        if add_noise:
            latent_shapes += -self.configuration.LATENT_POINTS_NOISE + torch.rand_like(latent_shapes) * (
                self.configuration.LATENT_POINTS_NOISE - (-self.configuration.LATENT_POINTS_NOISE)
            )

        x = torch.linspace(
            self.configuration.MIN_BOUND,
            self.configuration.MAX_BOUND,
            self.configuration.RECONSTRUCTION_GRID_SIZE,
        )
        y = torch.linspace(
            self.configuration.MIN_BOUND,
            self.configuration.MAX_BOUND,
            self.configuration.RECONSTRUCTION_GRID_SIZE,
        )
        z = torch.linspace(
            self.configuration.MIN_BOUND,
            self.configuration.MAX_BOUND,
            self.configuration.RECONSTRUCTION_GRID_SIZE,
        )
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        xyz = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(self.configuration.DEVICE)

        results = []

        for lsi, latent_shape in enumerate(latent_shapes):
            latent_shape_flattened = latent_shape.reshape(1, -1)

            sdfs = []
            xyz_splitted = xyz.split(self.configuration.RECONSTRUCTION_GRID_SIZE * 20)

            for xyz_batch in xyz_splitted:
                latent_shape_expanded = latent_shape_flattened.expand(xyz_batch.shape[0], -1)
                cxyz = torch.cat([xyz_batch, latent_shape_expanded], dim=-1)

                sdf = self.forward(cxyz)
                sdfs.append(sdf)

            sdfs = torch.cat(sdfs, dim=0).cpu().numpy()

            if not (sdfs.min() <= self.configuration.MARCHING_CUBES_LEVEL <= sdfs.max()):
                print(f"sdf is not in the range of {self.configuration.MARCHING_CUBES_LEVEL}")
                results.append(None)
                continue

            sdf_grid = sdfs.reshape(
                self.configuration.RECONSTRUCTION_GRID_SIZE,
                self.configuration.RECONSTRUCTION_GRID_SIZE,
                self.configuration.RECONSTRUCTION_GRID_SIZE,
            )

            vertices, faces, _, _ = skimage.measure.marching_cubes(
                sdf_grid, level=self.configuration.MARCHING_CUBES_LEVEL
            )

            if normalize:
                vertices = vertices / vertices.max()
                vertices = self.configuration.MIN_BOUND + vertices * (
                    self.configuration.MAX_BOUND - self.configuration.MIN_BOUND
                )

                assert vertices.min() >= self.configuration.MIN_BOUND
                assert vertices.max() <= self.configuration.MAX_BOUND

            mesh = trimesh.Trimesh(vertices, faces)
            mesh.vertices -= mesh.vertices.mean(axis=0)

            if rescale:
                latent_shape_bounds = torch.stack(
                    [latent_shape.amin(dim=0), latent_shape.amax(dim=0)], dim=0
                ).cpu().numpy()

                # compute scale factor
                mesh_size = mesh.bounds[1] - mesh.bounds[0]
                latent_size = latent_shape_bounds[1] - latent_shape_bounds[0]
                scale_factor = latent_size / mesh_size

                # scale
                mesh.vertices = mesh.vertices * scale_factor

                # centralize
                translation = (latent_shape_bounds * 0.5).sum(axis=0) - (mesh.bounds * 0.5).sum(axis=0)
                mesh.vertices = mesh.vertices + translation

            mesh = trimesh.util.concatenate([mesh, trimesh.Trimesh(vertices=latent_shape.cpu().numpy(), faces=[])])

            if check_watertight and not mesh.is_watertight:
                vertices, faces = pcu.make_mesh_watertight(
                    mesh.vertices, mesh.faces, resolution=self.configuration.RECONSTRUCTION_WATERTIGHT_RESOLUTION
                )
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                
            if map_z_to_y:
                mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]

            file_path = os.path.join(save_path, f"mesh_reconstructed_{lsi}.obj")
            if isinstance(epoch, int):
                file_path = os.path.join(save_path, f"mesh_reconstructed_{lsi}_{epoch}.obj")

            mesh.export(file_path)

            results.append(mesh)

        self.train()

        return results
