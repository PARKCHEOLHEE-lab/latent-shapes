import os
import sys
import pytz
import torch
import datetime

from tqdm import tqdm
from typing import List, Tuple, Optional
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
        subset_count: int = 1,
        log_dir: Optional[str] = None,
    ):
        self.sdf_decoder = sdf_decoder
        self.sdf_decoder_optimizer = sdf_decoder_optimizer
        self.sdf_dataloader = sdf_dataloader
        self.configuration = configuration
        
        self.states = {
            "epoch": 1,
            "losses_mean": torch.inf,
            "losses_sdf": torch.inf,
            "losses_latent_points": torch.inf,
        }

        self.dataloaders = self._divide_dataloader(subset_count, self.sdf_dataloader)
        
        if log_dir is None:
            log_dir = os.path.join(
                self.configuration.LOG_DIR_BASE,
                datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%Y__%H-%M-%S"),
            )

        self.summary_writer = SummaryWriter(log_dir=log_dir)

    def _divide_dataloader(self, subset_count: int, sdf_dataloader: DataLoader) -> List[DataLoader]:
        """Divide the dataloader to subsets with the number of `subset_count`

        Args:
            subset_count (int): count to divide
            dataloader (DataLoader): dataloader

        Returns:
            List[Subset]: subsets
        """

        dataloaders = [sdf_dataloader]
        if subset_count > 1:
            train_loader_dataset_size = len(sdf_dataloader.dataset)
            train_loader_indices = torch.randperm(train_loader_dataset_size)

            subset_divider = train_loader_dataset_size // subset_count
            dataloaders = []
            for count in range(subset_count):
                each_subset_start = count * subset_divider
                each_subset_end = (count + 1) * subset_divider
                each_subset = Subset(sdf_dataloader.dataset, train_loader_indices[each_subset_start:each_subset_end])
                each_dataloader = DataLoader(
                    dataset=each_subset,
                    batch_size=sdf_dataloader.batch_size,
                    num_workers=int(os.cpu_count() * 0.7),
                    drop_last=True,
                    persistent_workers=True,
                )

                dataloaders.append(each_dataloader)

        return dataloaders

    def _train_each_epoch(self, subset: DataLoader) -> Tuple[float, float, float]:
        """ """

        losses = []
        losses_sdf = []
        losses_latent_points = []

        for batch_index, data in tqdm(enumerate(subset), total=len(subset)):
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

        loss_mean = torch.tensor(losses).mean()
        loss_sdf_mean = torch.tensor(losses_sdf).mean()
        loss_latent_points_mean = torch.tensor(losses_latent_points).mean()

        return loss_mean, loss_sdf_mean, loss_latent_points_mean

    def train(self) -> None:
        torch.multiprocessing.set_start_method("spawn", force=True)

        epoch_start = self.states["epoch"]
        epoch_end = self.configuration.EPOCHS + 1

        for epoch in tqdm(range(epoch_start, epoch_end)):
            
            losses = []
            losses_sdf = []
            losses_latent_points = []
            
            for si, subset in enumerate(self.dataloaders):
                subset_loss_mean, subset_loss_sdf_mean, subset_loss_latent_points_mean = self._train_each_epoch(subset)

                losses.append(subset_loss_mean)
                losses_sdf.append(subset_loss_sdf_mean)
                losses_latent_points.append(subset_loss_latent_points_mean)

                self.summary_writer.add_scalar(f"subset_{si}_loss_mean", subset_loss_mean, epoch)
                self.summary_writer.add_scalar(f"subset_{si}_loss_sdf_mean", subset_loss_sdf_mean, epoch)
                self.summary_writer.add_scalar(
                    f"subset_{si}_loss_latent_points_mean", subset_loss_latent_points_mean, epoch
                )

            losses_mean = torch.tensor(losses).mean()
            losses_sdf_mean = torch.tensor(losses_sdf).mean()
            losses_latent_points_mean = torch.tensor(losses_latent_points).mean()

            self.summary_writer.add_scalar("loss_mean", losses_mean, epoch)
            self.summary_writer.add_scalar("loss_sdf_mean", losses_sdf_mean, epoch)
            self.summary_writer.add_scalar("loss_latent_points_mean", losses_latent_points_mean, epoch)


if __name__ == "__main__":
    configuration = Configuration()

    data_path = []
    for folder in os.listdir(configuration.DATA_PATH):
        path = os.path.join(configuration.DATA_PATH, folder, configuration.DATA_NAME, f"{folder}.npz")
        if os.path.exists(path):
            data_path.append(path)

        if len(data_path) == 1:
            break

    data_path = sorted(data_path)

    dataset = SDFDataset(data_path=data_path, configuration=configuration)
    dataloader = DataLoader(dataset=dataset, batch_size=configuration.BATCH_SIZE, shuffle=True)

    sdf_decoder = SDFDecoder(num_classes=len(data_path), configuration=configuration)
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
        subset_count=configuration.SUBSET_COUNT,
    )

    sdf_decoder_trainer.train()
