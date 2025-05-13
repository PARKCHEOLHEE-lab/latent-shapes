import os
import sys
import torch
import skimage
import torch.nn as nn

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from latent_shape_interpolator.src.config import Configuration


class MultiVectorEmbedding(nn.Module):
    def __init__(self, num_classes: int, num_latent_points: int, min_bound: float, max_bound: float):
        super().__init__()

        self.num_classes = num_classes
        self.num_latent_points = num_latent_points

        self.multi_vector_embedding = nn.Parameter(torch.randn(self.num_classes, self.num_latent_points, 3))
        nn.init.uniform_(self.multi_vector_embedding.data, min_bound, max_bound)

    def forward(self, class_number: torch.Tensor) -> torch.Tensor:
        return self.multi_vector_embedding[class_number]


class SDFDecoder(nn.Module):
    def __init__(self, num_classes: int, configuration: Configuration):
        super().__init__()

        self.num_classes = num_classes
        self.configuration = configuration

        self.latent_points_embedding = MultiVectorEmbedding(
            num_classes,
            self.configuration.NUM_LATENT_POINTS,
            self.configuration.MIN_BOUND,
            self.configuration.MAX_BOUND,
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
        )

        self.to(self.configuration.DEVICE)

    def forward(self, class_number, xyz, cxyz_1=None):
        if cxyz_1 is None:
            cxyz_1 = torch.cat((xyz.unsqueeze(1), self.latent_points_embedding(class_number)), dim=1)
            cxyz_1 = cxyz_1.reshape(xyz.shape[0], -1)

        x1 = self.main_1(cxyz_1)

        cxyz_2 = torch.cat((x1, cxyz_1), dim=1)

        x2 = self.main_2(cxyz_2)

        return x2

    @torch.no_grad()
    def reconstruct(self, latent_points: torch.Tensor):
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
        xyz = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        cxyz_1 = torch.cat((xyz.unsqueeze(1), latent_points), dim=1)

        sdf = self.forward(None, None, cxyz_1=cxyz_1)

        grid_sdf = sdf.reshape(
            self.configuration.GRID_SIZE_RECONSTRUCTION,
            self.configuration.GRID_SIZE_RECONSTRUCTION,
            self.configuration.GRID_SIZE_RECONSTRUCTION,
        )

        vertices, faces, _, _ = skimage.measure.marching_cubes(grid_sdf, level=0.00)

        self.train()

        # # TODO: Save to mesh

        return
