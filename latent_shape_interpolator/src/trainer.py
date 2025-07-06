import os
import sys
import pytz
import torch
import shutil
import inspect
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

        self.sdf_decoder_module = self.sdf_decoder.module if self.configuration.USE_MULTI_GPUS else self.sdf_decoder

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
            "epoch": 0,
            "loss_mean": torch.inf,
            "loss_mean_val": torch.inf,
            "loss_sdf_val": torch.inf,
            "loss_latent_shapes_val": torch.inf,
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

            print(f"Loaded states successfully from `{pretrained_dir}`")

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

        for _, data in tqdm(
            enumerate(self.sdf_dataset.validation_dataloader), total=len(self.sdf_dataset.validation_dataloader)
        ):
            xyz_batch, sdf_batch, class_number_batch, latent_shapes_batch_r, _ = data

            latent_shapes_batch = self.sdf_decoder_module.latent_shapes_embedding(class_number_batch)
            latent_shapes_batch = latent_shapes_batch.reshape(latent_shapes_batch.shape[0], -1)

            # add noise
            latent_shapes_batch += -self.configuration.LATENT_POINTS_NOISE + torch.rand_like(latent_shapes_batch) * (
                self.configuration.LATENT_POINTS_NOISE - (-self.configuration.LATENT_POINTS_NOISE)
            )

            cxyz = torch.cat((xyz_batch, latent_shapes_batch), dim=1)

            sdf_preds = self.sdf_decoder(cxyz)

            sdf_preds = torch.clamp(sdf_preds, min=-self.configuration.CLAMP, max=self.configuration.CLAMP)
            sdf_batch = torch.clamp(sdf_batch, min=-self.configuration.CLAMP, max=self.configuration.CLAMP)

            loss = torch.nn.functional.l1_loss(sdf_preds, sdf_batch.unsqueeze(-1))
            
            if self.configuration.USE_SHAPE_LOSS:
                loss = loss + torch.nn.functional.mse_loss(
                    latent_shapes_batch.reshape(-1, self.configuration.NUM_LATENT_POINTS, 3), 
                    latent_shapes_batch_r
                )

            losses_val.append(loss.item())

        loss_mean_val = torch.tensor(losses_val).mean()

        # re-set to training mode
        self.sdf_decoder.train()

        return loss_mean_val

    def _train_each_epoch(self) -> Tuple[float, float, float]:
        """ """

        losses = []

        for batch_index, data in tqdm(
            enumerate(self.sdf_dataset.train_dataloader), total=len(self.sdf_dataset.train_dataloader)
        ):
            xyz_batch, sdf_batch, class_number_batch, latent_shapes_batch_r, _ = data

            latent_shapes_batch = self.sdf_decoder_module.latent_shapes_embedding(class_number_batch)
            latent_shapes_batch = latent_shapes_batch.reshape(latent_shapes_batch.shape[0], -1)

            # add noise
            latent_shapes_batch += -self.configuration.LATENT_POINTS_NOISE + torch.rand_like(latent_shapes_batch) * (
                self.configuration.LATENT_POINTS_NOISE - (-self.configuration.LATENT_POINTS_NOISE)
            )

            cxyz = torch.cat((xyz_batch, latent_shapes_batch), dim=1)

            sdf_preds = self.sdf_decoder(cxyz)

            sdf_preds = torch.clamp(sdf_preds, min=-self.configuration.CLAMP, max=self.configuration.CLAMP)
            sdf_batch = torch.clamp(sdf_batch, min=-self.configuration.CLAMP, max=self.configuration.CLAMP)

            loss = torch.nn.functional.l1_loss(sdf_preds, sdf_batch.unsqueeze(-1))

            if self.configuration.USE_SHAPE_LOSS:
                loss = loss + torch.nn.functional.mse_loss(
                    latent_shapes_batch.reshape(-1, self.configuration.NUM_LATENT_POINTS, 3), 
                    latent_shapes_batch_r
                )
            
            loss = loss / self.configuration.ACCUMULATION_STEPS

            loss.backward()
            if (batch_index + 1) % self.configuration.ACCUMULATION_STEPS == 0 or (batch_index + 1) == len(
                self.sdf_dataset.train_dataloader
            ):
                self.sdf_decoder_optimizer.step()
                self.sdf_decoder_optimizer.zero_grad()

            losses.append(loss.item() * self.configuration.ACCUMULATION_STEPS)

        loss_mean = torch.tensor(losses).mean()

        return loss_mean

    def train(self) -> None:
        # copy used configuration
        config_path = inspect.getfile(Configuration)
        shutil.copy(config_path, os.path.join(self.log_dir, os.path.basename(config_path)))

        epoch_start = self.states["epoch"] + 1
        epoch_end = self.configuration.EPOCHS + 1

        for epoch in tqdm(range(epoch_start, epoch_end)):
            loss_mean = self._train_each_epoch()
            loss_mean_val = self._evaluate_each_epoch()

            loss_mean_weighted_sum = (
                loss_mean * self.configuration.LOSS_TRAIN_WEIGHT
                + loss_mean_val * self.configuration.LOSS_VALIDATION_WEIGHT
            )

            self.summary_writer.add_scalar("loss_mean", loss_mean, epoch)
            self.summary_writer.add_scalar("loss_mean_val", loss_mean_val, epoch)
            self.summary_writer.add_scalar("loss_mean_weighted_sum", loss_mean_weighted_sum, epoch)

            self.scheduler.step(loss_mean_weighted_sum)

            if epoch == 1 or epoch % self.configuration.RECONSTRUCTION_INTERVAL == 0:
                
                latent_shapes_batch = self.sdf_dataset.latent_shapes[
                    torch.randperm(self.sdf_dataset.num_classes)[: self.configuration.RECONSTRUCTION_COUNT]
                ]
                
                reconstruction_results = self.sdf_decoder_module.reconstruct(
                    latent_shapes_batch,
                    save_path=self.log_dir,
                    normalize=True,
                    check_watertight=False,
                    add_noise=False,
                    rescale=True,
                    epoch=epoch,
                )

                if reconstruction_results.count(None) == latent_shapes_batch.shape[0]:
                    print(f"All reconstructions failed at epoch {epoch}")

            if loss_mean_weighted_sum < self.states["loss_mean_weighted_sum"]:
                print(
                    f"""
                    states updated:
                        loss_mean_weighted_sum: {loss_mean_weighted_sum}
                        self.states["loss_mean_weighted_sum"]: {self.states["loss_mean_weighted_sum"]}
                    """
                )

                self.states.update(
                    {
                        "epoch": epoch,
                        "loss_mean": loss_mean,
                        "loss_mean_val": loss_mean_val,
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
