import os
import sys
import pytz
import torch
import datetime

from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from latent_shape_interpolator.src.config import Configuration
from latent_shape_interpolator.src.data import SDFDataset
from latent_shape_interpolator.src.model import SDFDecoder


class Trainer:
    def __init__(
        self,
        sdf_decoder: SDFDecoder,
        sdf_decoder_optimizer: torch.optim.Optimizer,
        sdf_dataloader: DataLoader,
        configuration: Configuration,
        log_dir: str = None,
    ):
        self.sdf_decoder = sdf_decoder
        self.sdf_decoder_optimizer = sdf_decoder_optimizer
        self.sdf_dataloader = sdf_dataloader
        self.configuration = configuration

        self.summary_writer = None
        if log_dir is None:
            self.summary_writer = SummaryWriter(
                log_dir=os.path.join(
                    self.configuration.LOG_DIR_BASE,
                    datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%Y__%H-%M-%S"),
                )
            )

    def _create_subsets(self, subset_count: int, dataloader: DataLoader) -> List[Subset]:
        """Divide the dataloader to subsets with the number of `subset_count`

        Args:
            subset_count (int): count to divide
            dataloader (DataLoader): dataloader

        Returns:
            List[Subset]: subsets
        """

        dataloader_subsets = [dataloader]
        if subset_count > 1:
            train_loader_dataset_size = len(dataloader.dataset)
            train_loader_indices = torch.randperm(train_loader_dataset_size)

            subset_divider = train_loader_dataset_size // subset_count
            dataloader_subsets = []
            for subset_count in range(subset_count):
                subset_start = subset_count * subset_divider
                subset_end = (subset_count + 1) * subset_divider

                if subset_count == subset_count - 1:
                    subset_end = train_loader_dataset_size

                dataloader_subset = Subset(dataloader.dataset, train_loader_indices[subset_start:subset_end])

                each_dataloader = DataLoader(
                    dataset=dataloader_subset,
                    batch_size=dataloader.batch_size,
                    num_workers=int(os.cpu_count() * 0.7),
                    shuffle=True,
                    drop_last=True,
                    persistent_workers=True,
                )

                dataloader_subsets.append(each_dataloader)

        return dataloader_subsets

    def _train_each_epoch(self) -> None:
        losses = []
        losses_sdf = []
        losses_latent_points = []

        for batch_index, data in tqdm(enumerate(self.sdf_dataloader), total=len(self.sdf_dataloader)):
            xyz_batch, sdf_batch, class_number_batch, latent_points_batch, faces_batch = data

            sdf_preds = self.sdf_decoder(class_number_batch, xyz_batch)
            sdf_preds = torch.clamp(
                sdf_preds, min=self.sdf_dataloader.dataset.min_sdf, max=self.sdf_dataloader.dataset.max_sdf
            )

            loss_sdf = torch.nn.functional.l1_loss(sdf_preds, sdf_batch.unsqueeze(-1))
            loss_latent_points = torch.nn.functional.l1_loss(
                self.sdf_decoder.latent_points_embedding(class_number_batch), latent_points_batch
            )

            loss = loss_sdf + loss_latent_points
            loss /= self.configuration.ACCUMULATION_STEPS

            loss.backward()
            if (batch_index + 1) % self.configuration.ACCUMULATION_STEPS == 0:
                self.sdf_decoder_optimizer.step()
                self.sdf_decoder_optimizer.zero_grad()

            losses.append(loss.item() * self.configuration.ACCUMULATION_STEPS)
            losses_sdf.append(loss_sdf.item() * self.configuration.ACCUMULATION_STEPS)
            losses_latent_points.append(loss_latent_points.item() * self.configuration.ACCUMULATION_STEPS)

        losses_mean = sum(losses) / len(losses)
        losses_sdf_mean = sum(losses_sdf) / len(losses_sdf)
        losses_latent_points_mean = sum(losses_latent_points) / len(losses_latent_points)

        return losses_mean, losses_sdf_mean, losses_latent_points_mean

    def train(self) -> None:
        for epoch in tqdm(range(1, self.configuration.EPOCHS + 1)):
            losses_mean, losses_sdf_mean, losses_latent_points_mean = self._train_each_epoch()

            if self.summary_writer is not None:
                self.summary_writer.add_scalar("losses_mean", losses_mean, epoch)
                self.summary_writer.add_scalar("losses_sdf_mean", losses_sdf_mean, epoch)
                self.summary_writer.add_scalar("losses_latent_points_mean", losses_latent_points_mean, epoch)


if __name__ == "__main__":
    configuration = Configuration()

    data_path = []
    for folder in os.listdir(configuration.DATA_PATH):
        path = os.path.join(configuration.DATA_PATH, folder, configuration.DATA_NAME, f"{folder}.npz")
        if os.path.exists(path):
            data_path.append(path)

    data_path.sort()

    dataset = SDFDataset(data_path=data_path, configuration=configuration)
    dataloader = DataLoader(dataset=dataset, batch_size=configuration.BATCH_SIZE, shuffle=True)

    sdf_decoder = SDFDecoder(num_classes=10, configuration=configuration)
    sdf_decoder_optimizer = torch.optim.AdamW(
        [
            {"params": sdf_decoder.latent_points_embedding.parameters(), "lr": configuration.LR_LATENT_POINTS},
            {"params": sdf_decoder.main_1.parameters(), "lr": configuration.LR_DECODER},
            {"params": sdf_decoder.main_2.parameters(), "lr": configuration.LR_DECODER},
        ],
    )

    sdf_decoder_trainer = Trainer(
        sdf_decoder=sdf_decoder,
        sdf_decoder_optimizer=sdf_decoder_optimizer,
        sdf_dataloader=dataloader,
        configuration=configuration,
    )

    sdf_decoder_trainer.train()
