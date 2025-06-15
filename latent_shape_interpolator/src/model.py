import os
import sys
import torch
import skimage
import trimesh
import torch.nn as nn
import point_cloud_utils as pcu

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from latent_shape_interpolator.src.config import Configuration


class LatentShapes(nn.Module):
    def __init__(self, latent_shapes: int, min_bound: float, max_bound: float):
        super().__init__()

        # add noise to latent points
        self.noise = min_bound + torch.rand_like(latent_shapes) * (max_bound - min_bound)

        # check if the noise range is valid
        assert torch.all(self.noise >= min_bound)
        assert torch.all(self.noise <= max_bound)

        # initialize latent points with noise
        self.embedding = nn.Parameter(latent_shapes + self.noise)

    def forward(self, class_number: torch.Tensor) -> torch.Tensor:
        return self.embedding[class_number]


class SDFDecoder(nn.Module):
    def __init__(self, latent_shapes: torch.Tensor, configuration: Configuration):
        super().__init__()

        self.latent_shapes = latent_shapes
        self.configuration = configuration

        self.latent_shapes_embedding = LatentShapes(
            latent_shapes=self.latent_shapes,
            min_bound=-self.configuration.LATENT_POINTS_NOISE,
            max_bound=self.configuration.LATENT_POINTS_NOISE,
        )

        self.main_1_in_features = (self.configuration.NUM_LATENT_POINTS + 1) * 3
        self.main_1 = nn.Sequential(
            nn.Linear(self.main_1_in_features, self.configuration.HIDDEN_DIM),
            nn.ReLU(True),
            nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            nn.ReLU(True),
            nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            nn.ReLU(True),
            nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            nn.ReLU(True),
            nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
        )

        self.main_2_in_features = self.main_1_in_features + self.configuration.HIDDEN_DIM
        self.main_2 = nn.Sequential(
            nn.Linear(self.main_2_in_features, self.configuration.HIDDEN_DIM),
            nn.ReLU(True),
            nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            nn.ReLU(True),
            nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            nn.ReLU(True),
            nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
            nn.ReLU(True),
            nn.Linear(self.configuration.HIDDEN_DIM, 1),
            nn.Tanh(),
        )

        self.to(self.configuration.DEVICE)

    def forward(self, class_number: torch.Tensor, xyz: torch.Tensor, cxyz_1: torch.Tensor = None):
        if cxyz_1 is None:
            cxyz_1 = torch.cat((xyz.unsqueeze(1), self.latent_shapes_embedding(class_number)), dim=1)
            cxyz_1 = cxyz_1.reshape(xyz.shape[0], -1)

        x1 = self.main_1(cxyz_1)

        cxyz_2 = torch.cat((x1, cxyz_1), dim=1)

        x2 = self.main_2(cxyz_2)

        return x2

    @torch.no_grad()
    def reconstruct(
        self,
        latent_shapes: torch.Tensor,
        save_path: str,
        normalize: bool = True,
        check_watertight: bool = False,
        map_z_to_y: bool = False,
    ):
        self.eval()

        x = torch.linspace(
            self.configuration.MIN_BOUND,
            self.configuration.MAX_BOUND,
            self.configuration.GRID_SIZE_RECONSTRUCTION,
        )
        y = torch.linspace(
            self.configuration.MIN_BOUND,
            self.configuration.MAX_BOUND,
            self.configuration.GRID_SIZE_RECONSTRUCTION,
        )
        z = torch.linspace(
            self.configuration.MIN_BOUND,
            self.configuration.MAX_BOUND,
            self.configuration.GRID_SIZE_RECONSTRUCTION,
        )
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        xyz = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(self.configuration.DEVICE)

        results = []

        for lsi, latent_shape in enumerate(latent_shapes):
            latent_shape_flattened = latent_shape.reshape(1, -1)

            sdfs = []
            xyz_splitted = xyz.split(self.configuration.GRID_SIZE_RECONSTRUCTION * 10)

            for xyz_batch in xyz_splitted:
                latent_shape_expanded = latent_shape_flattened.expand(xyz_batch.shape[0], -1)
                cxyz_1 = torch.cat([xyz_batch, latent_shape_expanded], dim=-1)

                sdf = self.forward(None, None, cxyz_1=cxyz_1)
                sdfs.append(sdf)

            sdfs = torch.cat(sdfs, dim=0).cpu().numpy()

            if not (sdfs.min() <= self.configuration.MARCHING_CUBES_LEVEL <= sdfs.max()):
                print(f"sdf is not in the range of {self.configuration.MARCHING_CUBES_LEVEL}")
                results.append(False)
                continue

            sdf_grid = sdfs.reshape(
                self.configuration.GRID_SIZE_RECONSTRUCTION,
                self.configuration.GRID_SIZE_RECONSTRUCTION,
                self.configuration.GRID_SIZE_RECONSTRUCTION,
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

            if check_watertight and not mesh.is_watertight:
                vertices, faces = pcu.make_mesh_watertight(
                    mesh.vertices, mesh.faces, resolution=self.configuration.RECONSTRUCTION_WATERTIGHT_RESOLUTION
                )
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            if map_z_to_y:
                mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]

            mesh.export(os.path.join(save_path, f"mesh_reconstructed_{lsi}.obj"))

            results.append(True)

        self.train()

        return results
