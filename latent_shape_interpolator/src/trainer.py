import os
import sys
import pytz
import torch
import datetime

from tqdm import tqdm
from typing import Tuple, Optional
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
        sdf_dataset: SDFDataset,
        configuration: Configuration,
        pretrained_dir: Optional[str] = None,
    ):
        self.sdf_decoder = sdf_decoder
        self.sdf_decoder_optimizer = sdf_decoder_optimizer
        self.sdf_dataset = sdf_dataset
        self.configuration = configuration

        assert None not in (
            self.sdf_dataset.train_dataset,
            self.sdf_dataset.train_dataloader,
            self.sdf_dataset.validation_dataset,
            self.sdf_dataset.validation_dataloader,
        )

        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.sdf_decoder_optimizer,
            factor=self.configuration.SCHEDULER_FACTOR,
            patience=self.configuration.SCHEDULER_PATIENCE,
        )

        # default states
        self.states = {
            "epoch": 1,
            "loss_mean": torch.inf,
            "loss_sdf": torch.inf,
            "loss_latent_points": torch.inf,
            "loss_mean_val": torch.inf,
            "loss_sdf_val": torch.inf,
            "loss_latent_points_val": torch.inf,
            "loss_mean_weighted_sum": torch.inf,
            "state_dict_model": self.sdf_decoder.state_dict(),
            "state_dict_optimizer": self.sdf_decoder_optimizer.state_dict(),
            "state_dict_scheduler": self.scheduler.state_dict(),
            "configuration": self.configuration.to_dict(),
        }

        # load and set states from log_dir
        if isinstance(pretrained_dir, str) and os.path.exists(pretrained_dir):
            states_path = os.path.join(pretrained_dir, self.configuration.SAVE_NAME)
            assert os.path.exists(states_path)

            self.states = torch.load(states_path)
            self.sdf_decoder.load_state_dict(self.states["state_dict_model"])
            self.sdf_decoder_optimizer.load_state_dict(self.states["state_dict_optimizer"])
            self.scheduler.load_state_dict(self.states["state_dict_scheduler"])

        # create new log_dir
        elif pretrained_dir is None:
            pretrained_dir = os.path.join(
                self.configuration.LOG_DIR_BASE,
                datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%m-%d-%Y__%H-%M-%S"),
            )

        self.summary_writer = SummaryWriter(log_dir=pretrained_dir)

    @property
    def log_dir(self) -> str:
        return self.summary_writer.log_dir

    @torch.no_grad()
    def _evaluate_each_epoch(self) -> Tuple[float, float, float]:
        """ """

        # set to evaluation mode
        self.sdf_decoder.eval()

        losses_val = []
        losses_sdf_val = []
        losses_latent_points_val = []

        for _, data in tqdm(
            enumerate(self.sdf_dataset.validation_dataloader), total=len(self.sdf_dataset.validation_dataloader)
        ):
            xyz_batch, sdf_batch, class_number_batch, latent_points_batch, _ = data

            sdf_preds = self.sdf_decoder(class_number_batch, xyz_batch)
            sdf_preds = torch.clamp(sdf_preds, min=self.sdf_dataset.min_sdf, max=self.sdf_dataset.max_sdf)

            loss_sdf = torch.nn.functional.l1_loss(sdf_preds, sdf_batch.unsqueeze(-1))
            loss_latent_points = torch.nn.functional.l1_loss(
                self.sdf_decoder.latent_points_embedding(class_number_batch), latent_points_batch
            )

            loss = loss_sdf + loss_latent_points

            losses_val.append(loss.item())
            losses_sdf_val.append(loss_sdf.item())
            losses_latent_points_val.append(loss_latent_points.item())

        loss_mean_val = torch.tensor(losses_val).mean()
        loss_sdf_mean_val = torch.tensor(losses_sdf_val).mean()
        loss_latent_points_mean_val = torch.tensor(losses_latent_points_val).mean()

        # re-set to training mode
        self.sdf_decoder.train()

        return loss_mean_val, loss_sdf_mean_val, loss_latent_points_mean_val

    def _train_each_epoch(self) -> Tuple[float, float, float]:
        """ """

        losses = []
        losses_sdf = []
        losses_latent_points = []

        for batch_index, data in tqdm(
            enumerate(self.sdf_dataset.train_dataloader), total=len(self.sdf_dataset.train_dataloader)
        ):
            xyz_batch, sdf_batch, class_number_batch, latent_points_batch, _ = data

            sdf_preds = self.sdf_decoder(class_number_batch, xyz_batch)
            sdf_preds = torch.clamp(sdf_preds, min=self.sdf_dataset.min_sdf, max=self.sdf_dataset.max_sdf)

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
        epoch_start = self.states["epoch"]
        epoch_end = self.configuration.EPOCHS + 1

        for epoch in tqdm(range(epoch_start, epoch_end)):
            loss_mean, loss_sdf_mean, loss_latent_points_mean = self._train_each_epoch()
            loss_mean_val, loss_sdf_mean_val, loss_latent_points_mean_val = self._evaluate_each_epoch()

            loss_mean_weighted_sum = (
                loss_mean * self.configuration.LOSS_TRAIN_WEIGHT
                + loss_mean_val * self.configuration.LOSS_VALIDATION_WEIGHT
            )

            self.summary_writer.add_scalar("loss_mean", loss_mean, epoch)
            self.summary_writer.add_scalar("loss_sdf_mean", loss_sdf_mean, epoch)
            self.summary_writer.add_scalar("loss_latent_points_mean", loss_latent_points_mean, epoch)

            self.summary_writer.add_scalar("loss_mean_val", loss_mean_val, epoch)
            self.summary_writer.add_scalar("loss_sdf_mean_val", loss_sdf_mean_val, epoch)
            self.summary_writer.add_scalar("loss_latent_points_mean_val", loss_latent_points_mean_val, epoch)

            self.summary_writer.add_scalar("loss_mean_weighted_sum", loss_mean_weighted_sum, epoch)

            self.scheduler.step(loss_mean_weighted_sum)

            if loss_mean_weighted_sum < self.states["loss_mean_weighted_sum"]:
                self.states.update(
                    {
                        "epoch": epoch,
                        "loss_mean": loss_mean,
                        "loss_sdf": loss_sdf_mean,
                        "loss_latent_points": loss_latent_points_mean,
                        "loss_mean_val": loss_mean_val,
                        "loss_sdf_val": loss_sdf_mean_val,
                        "loss_latent_points_val": loss_latent_points_mean_val,
                        "loss_mean_weighted_sum": loss_mean_weighted_sum,
                        "state_dict_model": self.sdf_decoder.state_dict(),
                        "state_dict_optimizer": self.sdf_decoder_optimizer.state_dict(),
                        "state_dict_scheduler": self.scheduler.state_dict(),
                        "configuration": self.configuration.to_dict(),
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

    sdf_dataset = SDFDataset.create_dataset(
        data_dir=configuration.DATA_PATH_PROCESSED, configuration=configuration, data_slicer=5
    )

    sdf_decoder = SDFDecoder(num_classes=sdf_dataset.num_classes, configuration=configuration)
    sdf_decoder_optimizer = getattr(torch.optim, configuration.OPTIMIZER)(
        [
            {"params": sdf_decoder.latent_points_embedding.parameters(), "lr": configuration.LR_LATENT_POINTS},
            {"params": sdf_decoder.main_1.parameters(), "lr": configuration.LR_DECODER},
            {"params": sdf_decoder.main_2.parameters(), "lr": configuration.LR_DECODER},
        ],
    )

    sdf_decoder_trainer = Trainer(
        sdf_decoder=sdf_decoder,
        sdf_decoder_optimizer=sdf_decoder_optimizer,
        sdf_dataset=sdf_dataset,
        configuration=configuration,
        pretrained_dir="latent_shape_interpolator/runs/05-04-2025__17-32-53",
    )

    sdf_decoder_trainer.train()
