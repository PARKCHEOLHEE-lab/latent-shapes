import os
import sys
import pytz
import torch
import shutil
import datetime

from tqdm import tqdm
from typing import Tuple, Optional
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

if os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from latent_shapes.src.config import Configuration
from latent_shapes.src.data import SDFDataset
from latent_shapes.src.model import SDFDecoder, LatentShapes


class Trainer:
    """trainer for the decoder, and the latent shape"""

    def __init__(
        self,
        latent_shapes: LatentShapes,
        latent_shapes_optimizer: torch.optim.Optimizer,
        sdf_decoder: SDFDecoder,
        sdf_decoder_optimizer: torch.optim.Optimizer,
        sdf_dataset: SDFDataset,
        configuration: Configuration,
        pretrained_dir: Optional[str] = None,
    ):
        self.latent_shapes = latent_shapes
        self.latent_shapes_optimizer = latent_shapes_optimizer
        self.sdf_decoder = sdf_decoder
        self.sdf_decoder_optimizer = sdf_decoder_optimizer
        self.sdf_dataset = sdf_dataset
        self.configuration = configuration

        self.sdf_decoder_module = (
            self.sdf_decoder.module
            if self.configuration.USE_MULTI_GPUS and torch.cuda.device_count() >= 2
            else self.sdf_decoder
        )

        assert None not in (
            self.sdf_dataset.train_dataset,
            self.sdf_dataset.train_dataloader,
            self.sdf_dataset.validation_dataset,
            self.sdf_dataset.validation_dataloader,
        )

        self.scheduler_decoder = lr_scheduler.ReduceLROnPlateau(
            self.sdf_decoder_optimizer,
            factor=self.configuration.SCHEDULER_FACTOR,
            patience=self.configuration.SCHEDULER_PATIENCE,
        )

        self.scheduler_latent_shapes = lr_scheduler.ReduceLROnPlateau(
            self.latent_shapes_optimizer,
            factor=self.configuration.SCHEDULER_FACTOR,
            patience=self.configuration.SCHEDULER_PATIENCE,
        )

        # initialize states
        self.states = {
            "epoch": 0,
            "loss_mean": torch.inf,
            "loss_mean_val": torch.inf,
            "loss_shape_mean": torch.inf,
            "loss_mean_weighted_sum": torch.inf,
            "state_dict_latent_shapes": self.latent_shapes.state_dict(),
            "state_dict_latent_shapes_optimizer": self.latent_shapes_optimizer.state_dict(),
            "state_dict_scheduler_latent_shapes": self.scheduler_latent_shapes.state_dict(),
            "state_dict_decoder": self.sdf_decoder.state_dict(),
            "state_dict_decoder_optimizer": self.sdf_decoder_optimizer.state_dict(),
            "state_dict_scheduler_decoder": self.scheduler_decoder.state_dict(),
            "configuration": self.configuration.to_dict(),
        }

        # load and set states from log_dir
        if isinstance(pretrained_dir, str) and os.path.exists(pretrained_dir):
            states_path = os.path.join(pretrained_dir, self.configuration.SAVE_NAME)
            assert os.path.exists(states_path)

            self.states = torch.load(states_path)
            self.latent_shapes.load_state_dict(self.states["state_dict_latent_shapes"])
            self.latent_shapes_optimizer.load_state_dict(self.states["state_dict_latent_shapes_optimizer"])
            self.scheduler_latent_shapes.load_state_dict(self.states["state_dict_scheduler_latent_shapes"])
            self.sdf_decoder.load_state_dict(self.states["state_dict_decoder"])
            self.sdf_decoder_optimizer.load_state_dict(self.states["state_dict_decoder_optimizer"])
            self.scheduler_decoder.load_state_dict(self.states["state_dict_scheduler_decoder"])

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
    def _evaluate_each_epoch(self) -> Tuple[float, float]:
        """evaluation method for each epoch

        Returns:
            Tuple[float, float]: validaion loss for the decoder, and the latent shape
        """

        # set to evaluation mode
        self.sdf_decoder.eval()

        losses_val = []
        losses_shape_val = []

        iterator_val = tqdm(
            enumerate(self.sdf_dataset.validation_dataloader), total=len(self.sdf_dataset.validation_dataloader)
        )

        for _, data in iterator_val:
            xyz_batch, sdf_batch, class_number_batch, latent_shapes_batch_r, _ = data

            latent_shapes_batch = self.latent_shapes(class_number_batch)
            latent_shapes_batch = latent_shapes_batch.reshape(latent_shapes_batch.shape[0], -1)

            cxyz = torch.cat((xyz_batch, latent_shapes_batch), dim=1)

            sdf_preds = self.sdf_decoder(cxyz)

            sdf_preds = torch.clamp(sdf_preds, min=-self.configuration.CLAMP, max=self.configuration.CLAMP)
            sdf_batch = torch.clamp(sdf_batch, min=-self.configuration.CLAMP, max=self.configuration.CLAMP)

            loss = torch.nn.functional.l1_loss(sdf_preds, sdf_batch.unsqueeze(-1))
            loss_shape = torch.nn.functional.mse_loss(self.latent_shapes(class_number_batch), latent_shapes_batch_r)

            losses_val.append(loss.item())
            losses_shape_val.append(loss_shape.item())

        loss_mean_val = torch.tensor(losses_val).mean().item()
        losses_shape_mean_val = torch.tensor(losses_shape_val).mean().item()

        # re-set to training mode
        self.sdf_decoder.train()

        return loss_mean_val, losses_shape_mean_val

    def _train_each_epoch(self) -> Tuple[float, float]:
        """training method for each epoch

        Returns:
            Tuple[float, float]: training loss for the decoder, and the latent shape
        """

        losses = []
        losses_shape = []

        iterator_train = tqdm(
            enumerate(self.sdf_dataset.train_dataloader), total=len(self.sdf_dataset.train_dataloader)
        )

        for batch_index, data in iterator_train:
            xyz_batch, sdf_batch, class_number_batch, latent_shapes_batch_r, _ = data

            latent_shapes_batch = self.latent_shapes(class_number_batch)
            latent_shapes_batch = latent_shapes_batch.reshape(latent_shapes_batch.shape[0], -1)

            cxyz = torch.cat((xyz_batch, latent_shapes_batch), dim=1)

            sdf_preds = self.sdf_decoder(cxyz)

            sdf_preds = torch.clamp(sdf_preds, min=-self.configuration.CLAMP, max=self.configuration.CLAMP)
            sdf_batch = torch.clamp(sdf_batch, min=-self.configuration.CLAMP, max=self.configuration.CLAMP)

            loss_shape = torch.nn.functional.mse_loss(self.latent_shapes(class_number_batch), latent_shapes_batch_r)

            loss_shape.backward()
            self.latent_shapes_optimizer.step()
            self.latent_shapes_optimizer.zero_grad()

            loss = torch.nn.functional.l1_loss(sdf_preds, sdf_batch.unsqueeze(-1))
            loss = loss / self.configuration.ACCUMULATION_STEPS

            loss.backward()
            if (batch_index + 1) % self.configuration.ACCUMULATION_STEPS == 0 or (batch_index + 1) == len(
                self.sdf_dataset.train_dataloader
            ):
                self.sdf_decoder_optimizer.step()
                self.sdf_decoder_optimizer.zero_grad()

            losses.append(loss.item() * self.configuration.ACCUMULATION_STEPS)
            losses_shape.append(loss_shape.item())

        loss_mean = torch.tensor(losses).mean().item()
        loss_shape_mean = torch.tensor(losses_shape).mean().item()

        return loss_mean, loss_shape_mean

    def _save_state_dicts(
        self, epoch: int, loss_mean: float, loss_mean_val: float, loss_mean_weighted_sum: float, loss_shape_mean: float
    ) -> None:
        """save state dicts if improved

        Args:
            epoch (int): current epoch.
            loss_mean (float): mean training loss for the decoder
            loss_mean_val (float): mean validation loss for the decoder
            loss_mean_weighted_sum (float): weighted sum of training and validation losses
            loss_shape_mean (float): mean training loss for the latent shape
        """

        if loss_shape_mean < self.states["loss_shape_mean"]:
            print(
                f"""
                latentshapes states updated:
                    loss_shape_mean: {loss_shape_mean}
                    self.states["loss_shape_mean"]: {self.states["loss_shape_mean"]}
                """
            )

            self.states.update(
                {
                    "epoch": epoch,
                    "loss_shape_mean": loss_shape_mean,
                    "state_dict_latent_shapes": self.latent_shapes.state_dict(),
                    "state_dict_latent_shapes_optimizer": self.latent_shapes_optimizer.state_dict(),
                    "state_dict_scheduler_latent_shapes": self.scheduler_latent_shapes.state_dict(),
                    "configuration": self.configuration.to_dict(),
                }
            )

            torch.save(self.states, os.path.join(self.log_dir, self.configuration.SAVE_NAME))

        if loss_mean_weighted_sum < self.states["loss_mean_weighted_sum"]:
            print(
                f"""
                decoder states updated:
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
                    "state_dict_decoder": self.sdf_decoder.state_dict(),
                    "state_dict_decoder_optimizer": self.sdf_decoder_optimizer.state_dict(),
                    "state_dict_scheduler_decoder": self.scheduler_decoder.state_dict(),
                    "configuration": self.configuration.to_dict(),
                }
            )
            torch.save(self.states, os.path.join(self.log_dir, self.configuration.SAVE_NAME))

        else:
            self.states = torch.load(os.path.join(self.log_dir, self.configuration.SAVE_NAME))
            self.states.update({"epoch": epoch})
            torch.save(self.states, os.path.join(self.log_dir, self.configuration.SAVE_NAME))

    def _copy_srcs(self) -> None:
        """copy used srcs"""

        shutil.copytree(
            src=max(sys.path, key=lambda f: f.split("/")[-1] == "src"),
            dst=self.log_dir,
            dirs_exist_ok=True,
            ignore=lambda _, files: [f for f in files if f == "__pycache__"],
        )

    def train(self) -> None:
        """main method"""

        self._copy_srcs()

        epoch_start = self.states["epoch"] + 1
        epoch_end = self.configuration.EPOCHS + 1

        for epoch in tqdm(range(epoch_start, epoch_end)):
            loss_mean, loss_shape_mean = self._train_each_epoch()
            loss_mean_val, loss_shape_mean_val = self._evaluate_each_epoch()

            loss_mean_weighted_sum = (
                loss_mean * self.configuration.LOSS_TRAIN_WEIGHT
                + loss_mean_val * self.configuration.LOSS_VALIDATION_WEIGHT
            )

            self.summary_writer.add_scalar("loss_mean", loss_mean, epoch)
            self.summary_writer.add_scalar("loss_shape_mean", loss_shape_mean, epoch)
            self.summary_writer.add_scalar("loss_mean_val", loss_mean_val, epoch)
            self.summary_writer.add_scalar("loss_shape_mean_val", loss_shape_mean_val, epoch)
            self.summary_writer.add_scalar("loss_mean_weighted_sum", loss_mean_weighted_sum, epoch)

            self.scheduler_decoder.step(loss_mean_weighted_sum)
            self.scheduler_latent_shapes.step(loss_shape_mean)

            self._save_state_dicts(epoch, loss_mean, loss_mean_val, loss_mean_weighted_sum, loss_shape_mean)

            # reconstruction with loaded state dicts
            if epoch == 1 or epoch % self.configuration.RECONSTRUCTION_INTERVAL == 0:
                _latent_shapes = LatentShapes(latent_shapes=self.sdf_dataset.latent_shapes)

                _sdf_decoder = SDFDecoder(
                    configuration=self.configuration,
                )

                _states = torch.load(os.path.join(self.log_dir, self.configuration.SAVE_NAME))
                _latent_shapes.load_state_dict(_states["state_dict_latent_shapes"])
                _sdf_decoder.load_state_dict(_states["state_dict_decoder"])

                latent_shapes_batch_embedding = _latent_shapes(
                    torch.randperm(self.sdf_dataset.num_classes)[: self.configuration.RECONSTRUCTION_COUNT]
                )

                _sdf_decoder.reconstruct(
                    latent_shapes_batch_embedding,
                    save_path=self.log_dir,
                    normalize=True,
                    check_watertight=False,
                    add_noise=False,
                    rescale=False,
                    map_z_to_y=False,
                    additional_title="loaded_emb",
                )
