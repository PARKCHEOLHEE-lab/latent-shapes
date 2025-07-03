import os
import sys
import torch
import skimage
import trimesh
import numpy as np
import torch.nn as nn
import point_cloud_utils as pcu

from typing import Optional, List

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from latent_shape_interpolator.src.config import Configuration


class LatentShapes(nn.Module):
    def __init__(
        self, latent_shapes: torch.Tensor, noise_min: Optional[float] = None, noise_max: Optional[float] = None
    ):
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


class SDFDecoder(nn.Module):
    def __init__(self, latent_shapes: torch.Tensor, configuration: Configuration):
        super().__init__()

        self.latent_shapes = latent_shapes
        self.configuration = configuration

        # define latent shapes embedding
        self.latent_shapes_embedding = LatentShapes(latent_shapes=self.latent_shapes)

        self.xyz_projection = nn.Linear(3 + self.configuration.K * 3, self.configuration.ATTENTION_DIM)
        self.latent_projection = nn.Linear(self.configuration.NUM_LATENT_POINTS * 3, self.configuration.ATTENTION_DIM)

        self.attention = nn.MultiheadAttention(
            embed_dim=self.configuration.ATTENTION_DIM,
            num_heads=self.configuration.NUM_HEADS,
            dropout=0.1,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(self.configuration.ATTENTION_DIM, self.configuration.ATTENTION_DIM * 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.configuration.ATTENTION_DIM * 4, self.configuration.ATTENTION_DIM),
        )

        self.layer_norm_1 = nn.LayerNorm(self.configuration.ATTENTION_DIM)
        self.layer_norm_2 = nn.LayerNorm(self.configuration.ATTENTION_DIM)

        self.first_block_in_features = (self.configuration.NUM_LATENT_POINTS + 1) * 3

        self.blocks: List[nn.Sequential]
        self.blocks = []
        for b in range(self.configuration.NUM_BLOCKS):
            blocks: List[nn.Module]
            blocks = []

            in_features = (
                self.configuration.ATTENTION_DIM
                if b == 0
                else self.configuration.ATTENTION_DIM + self.configuration.HIDDEN_DIM
            )

            blocks.extend(
                [
                    nn.Linear(in_features, self.configuration.HIDDEN_DIM),
                    getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
                ]
            )

            for _ in range(self.configuration.NUM_LAYERS):
                blocks.extend(
                    [
                        nn.Linear(self.configuration.HIDDEN_DIM, self.configuration.HIDDEN_DIM),
                        getattr(nn, self.configuration.ACTIVATION)(**self.configuration.ACTIVATION_KWARGS),
                    ]
                )

            if b + 1 == self.configuration.NUM_BLOCKS:
                blocks.extend([nn.Linear(self.configuration.HIDDEN_DIM, 1), nn.Tanh()])

            self.blocks.append(nn.Sequential(*blocks))

        self.to(self.configuration.DEVICE)
        for block in self.blocks:
            block.to(self.configuration.DEVICE)

    def forward(self, cxyz: torch.Tensor):
        xyz = cxyz[:, :3]
        latent_shape = cxyz[:, 3:]
        latent_shape_ = latent_shape.reshape(-1, self.configuration.NUM_LATENT_POINTS, 3)

        # distance between xyz and latent shape
        distance = torch.func.vmap(lambda x, y: torch.cdist(x.unsqueeze(0), y))(xyz, latent_shape_)

        # select k closest points
        _, closest_indices = torch.topk(distance, k=self.configuration.K, dim=2, largest=False)
        closest_indices = closest_indices.squeeze().unsqueeze(-1)
        latent_shape_selected = torch.gather(latent_shape_, dim=1, index=closest_indices.expand(-1, -1, 3))

        xyz_projected = self.xyz_projection(torch.cat([xyz, latent_shape_selected.flatten(1)], dim=1))
        latent_shape_projected = self.latent_projection(latent_shape)

        x_, _ = self.attention(
            query=xyz_projected,
            key=latent_shape_projected,
            value=latent_shape_projected,
        )

        x_ = self.layer_norm_1(x_ + xyz_projected)
        x_ = self.layer_norm_2(self.ff(x_) + x_)

        x = x_
        for b, block in enumerate(self.blocks):
            x = block(x)

            if b + 1 != self.configuration.NUM_BLOCKS:
                x = torch.cat([x, x_], dim=1)

        sdf_preds = x

        return sdf_preds

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
                latent_shape_bounds = (
                    torch.stack([latent_shape.amin(dim=0), latent_shape.amax(dim=0)], dim=0).cpu().numpy()
                )

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
