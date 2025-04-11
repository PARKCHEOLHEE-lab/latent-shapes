import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from .data import SDFDataset  # FIXME


class MultiVectorEmbedding(nn.Module):
    def __init__(self, num_classes: int, num_latent_points: int):
        super().__init__()

        self.num_classes = num_classes
        self.num_latent_points = num_latent_points

        self.embedding = nn.Parameter(torch.randn(num_classes, num_latent_points, 3))
        self.positional_encoding = None

    def forward(self, class_number: torch.Tensor) -> torch.Tensor:
        return self.embedding[class_number]


class SDFDecoder(nn.Module):
    def __init__(self, num_classes: int, num_latent_points: int):
        super().__init__()

        self.num_classes = num_classes
        self.num_latent_points = num_latent_points

        self.latent_points_embedding = MultiVectorEmbedding(num_classes, self.num_latent_points)

        self.main_1 = nn.Sequential(
            nn.Linear((self.num_latent_points + 1) * 3, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
        )

        self.main_2 = nn.Sequential(
            nn.Linear((self.num_latent_points + 1) * 3 + 512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
        )

    def forward(self, class_number, xyz, cxyz_1=None):
        if cxyz_1 is None:
            cxyz_1 = torch.cat((xyz.unsqueeze(1), self.latent_points_embedding(class_number)), dim=1)
            cxyz_1 = cxyz_1.reshape(xyz.shape[0], -1)

        x1 = self.main_1(cxyz_1)

        cxyz_2 = torch.cat((x1, cxyz_1), dim=1)

        x2 = self.main_2(cxyz_2)

        return x2


if __name__ == "__main__":
    import os

    data_path = [os.path.join("data-processed", f) for f in os.listdir("data-processed")]
    data_path.sort()

    device = "cuda"

    dataset = SDFDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    sdf_decoder = SDFDecoder(num_classes=10)
    sdf_decoder = sdf_decoder.to(device)
    sdf_decoder_optimizer = torch.optim.AdamW(
        [
            {"params": sdf_decoder.latent_points_embedding.parameters(), "lr": 1e-3},
            {"params": sdf_decoder.main_1.parameters(), "lr": 1e-5},
            {"params": sdf_decoder.main_2.parameters(), "lr": 1e-5},
        ],
    )

    epochs = 1000
    for epoch in range(1, epochs + 1):
        for xyz_batch, sdf_batch, class_number_batch, latent_points_batch, faces_batch in dataloader:
            sdf_decoder_optimizer.zero_grad()

            xyz_batch = xyz_batch.to(device)
            sdf_batch = sdf_batch.to(device)
            class_number_batch = class_number_batch.to(device)
            latent_points_batch = latent_points_batch.to(device)
            faces_batch = faces_batch.to(device)

            sdf_preds = sdf_decoder(class_number_batch, xyz_batch)
            sdf_preds = torch.clamp(sdf_preds, min=dataset.min_sdf, max=dataset.max_sdf)
            loss_sdf = torch.nn.functional.l1_loss(sdf_preds, sdf_batch.unsqueeze(-1))

            latent_points_preds = sdf_decoder.latent_points_embedding(class_number_batch)
            latent_points_target = latent_points_batch.reshape(xyz_batch.shape[0], -1)
            loss_latent_points = torch.nn.functional.l1_loss(latent_points_preds, latent_points_target)

            loss = loss_sdf + loss_latent_points

            loss.backward()
            sdf_decoder_optimizer.step()

            print(loss.item())
            break
        break
