import os
import sys
import pytz
import torch
import datetime

from tqdm import tqdm
from typing import Tuple, Optional
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

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
        log_dir: Optional[str] = None,
    ):
        self.sdf_decoder = sdf_decoder
        self.sdf_decoder_optimizer = sdf_decoder_optimizer
        self.sdf_dataloader = sdf_dataloader
        self.configuration = configuration

        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.sdf_decoder_optimizer,
            factor=self.configuration.SCHEDULER_FACTOR,
            patience=self.configuration.SCHEDULER_PATIENCE,
        )

        self.states = {
            "epoch": 1,
            "loss_mean": torch.inf,
            "loss_sdf": torch.inf,
            "loss_latent_points": torch.inf,
            "state_dict_model": self.sdf_decoder.state_dict(),
            "state_dict_optimizer": self.sdf_decoder_optimizer.state_dict(),
            "state_dict_scheduler": self.scheduler.state_dict(),
        }

        if isinstance(log_dir, str):
            assert os.path.exists(log_dir)
            assert os.path.exists(os.path.join(log_dir, self.configuration.SAVE_NAME))

        if log_dir is None:
            log_dir = os.path.join(
                self.configuration.LOG_DIR_BASE,
                datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%Y__%H-%M-%S"),
            )

        self.summary_writer = SummaryWriter(log_dir=log_dir)

    @property
    def log_dir(self) -> str:
        return self.summary_writer.log_dir

    def _train_each_epoch(self) -> Tuple[float, float, float]:
        """ """

        losses = []
        losses_sdf = []
        losses_latent_points = []

        for batch_index, data in tqdm(enumerate(self.sdf_dataloader), total=len(self.sdf_dataloader)):
            xyz_batch, sdf_batch, class_number_batch, latent_points_batch, _ = data

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
        # torch.multiprocessing.set_start_method("spawn", force=True)

        epoch_start = self.states["epoch"]
        epoch_end = self.configuration.EPOCHS + 1

        for epoch in tqdm(range(epoch_start, epoch_end)):
            loss_mean, loss_sdf_mean, loss_latent_points_mean = self._train_each_epoch()

            self.scheduler.step(loss_mean)

            self.summary_writer.add_scalar("loss_mean", loss_mean, epoch)
            self.summary_writer.add_scalar("loss_sdf_mean", loss_sdf_mean, epoch)
            self.summary_writer.add_scalar("loss_latent_points_mean", loss_latent_points_mean, epoch)

            if loss_mean < self.states["loss_mean"]:
                self.states.update(
                    {
                        "epoch": epoch,
                        "loss_mean": loss_mean,
                        "loss_sdf": loss_sdf_mean,
                        "loss_latent_points": loss_latent_points_mean,
                        "state_dict_model": self.sdf_decoder.state_dict(),
                        "state_dict_optimizer": self.sdf_decoder_optimizer.state_dict(),
                        "state_dict_scheduler": self.scheduler.state_dict(),
                    }
                )
                torch.save(self.states, os.path.join(self.log_dir, self.configuration.SAVE_NAME))

            else:
                self.states = torch.load(os.path.join(self.log_dir, self.configuration.SAVE_NAME))
                self.states.update({"epoch": epoch})
                torch.save(self.states, os.path.join(self.log_dir, self.configuration.SAVE_NAME))


if __name__ == "__main__":
    configuration = Configuration()
    configuration.set_seed()

    dataset = SDFDataset(data_dir=configuration.DATA_PATH_PROCESSED, configuration=configuration, data_slicer=5)
    dataloader = DataLoader(dataset=dataset, batch_size=configuration.BATCH_SIZE, shuffle=True)

    sdf_decoder = SDFDecoder(num_classes=len(dataset.data_path), configuration=configuration)
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
