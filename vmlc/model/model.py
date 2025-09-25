"""
PyTorch Lightning module to handle sequence-to-sequence model
"""

import math
from typing import Any, Dict, List, Literal, Tuple

import pytorch_lightning as pl
import torch

from vmlc.model.metrics import evaluate_acc
from vmlc.model.modules import SeqToSeqTransformer


class LitModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        PAD_idx_input: int,
        PAD_idx_output: int,
        nlayers_encoder: int = 5,
        nlayers_decoder: int = 3,
        nhead: int = 8,
        dropout_p: float = 0.1,
        ff_mult: int = 4,
        activation: str = "gelu",
        grid_size: Tuple[int, int] = (10, 10),
        patch_size: Tuple[int, int] = (2, 2),
        num_support: int = 18,
        batch_size: int = 32,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        lr_end_factor: float = 0.05,
        nepochs: int = 10,
        max_length: int = 100,
        all_zero_token_idx: int = 0,
        zero_token_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = SeqToSeqTransformer(
            hidden_size=hidden_size,
            output_size=output_size,
            PAD_idx_input=PAD_idx_input,
            PAD_idx_output=PAD_idx_output,
            nlayers_encoder=nlayers_encoder,
            nlayers_decoder=nlayers_decoder,
            nhead=nhead,
            dropout_p=dropout_p,
            ff_mult=ff_mult,
            activation=activation,
            grid_size=grid_size,
            patch_size=patch_size,
            num_support=num_support,
        )

        # Create a weight vector for the loss such that the all-zero token gets lower weight.
        # All tokens initially have a weight of 1.0.
        class_weights = torch.ones(output_size)
        class_weights[all_zero_token_idx] = zero_token_weight

        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=PAD_idx_output, weight=class_weights
        )

        self.train_predictions: List[Any] = []
        self.validation_predictions: List[Any] = []

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass through the model.
        """
        target_shift = batch["yq_sos"]
        return self.model(target_shift, batch)

    def on_fit_start(self):
        """
        Set the number of steps per epoch based on the training dataloader.
        """
        if self.trainer.train_dataloader is not None:
            dataset_size = len(self.trainer.train_dataloader.dataset)
            self.nstep_epoch_estimate = math.ceil(
                dataset_size / self.hparams.batch_size
            )
        else:
            raise ValueError("Trainer does not have a training dataloader!")

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for the model.

        Args:
            batch (Dict[str, Any]): Input batch containing input and target sequences.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss for the batch.
        """
        target_batches = batch["yq_eos"]
        model_logits = self(batch)
        logits_flat = model_logits.reshape(-1, model_logits.shape[-1])

        # compute loss
        train_loss = self.loss_fn(logits_flat, target_batches.reshape(-1))
        self.log(
            "train/train_loss",
            train_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,  # type: ignore
            sync_dist=True,
        )

        # compute accuracy
        acc_dict = evaluate_acc(
            batch=batch,
            model_logits=model_logits,
            max_length=self.hparams.max_length,  # type: ignore
            tokenizer=self.trainer.datamodule.tokenizer,  # type: ignore
        )
        self.log_acc_metrics(
            acc_dict=acc_dict, batch_size=self.hparams.batch_size, mode="train"  # type: ignore
        )

        if batch_idx == 0:
            self.train_predictions = acc_dict["samples_pred"]

        return train_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Perform a validation step, computing validation loss, log-likelihood and accuracy metrics.
        """
        target_batches = batch["yq_eos"]
        model_logits = self(batch)
        logits_flat = model_logits.reshape(-1, model_logits.shape[-1])

        # compute loss
        val_loss = self.loss_fn(logits_flat, target_batches.reshape(-1))
        self.log(
            "val/val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,  # type: ignore
            sync_dist=True,
        )

        # Compute and log accuracy metrics
        acc_dict = evaluate_acc(
            batch=batch,
            model_logits=model_logits,
            max_length=self.hparams.max_length,  # type: ignore
            tokenizer=self.trainer.datamodule.tokenizer,  # type: ignore
        )
        self.log_acc_metrics(
            acc_dict=acc_dict, batch_size=self.hparams.batch_size, mode="val"  # type: ignore
        )
        self.validation_predictions.extend(acc_dict["samples_pred"])

        return {"val_loss": val_loss, **acc_dict}

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Perform a test step, computing test loss, log-likelihood and accuracy.
        """
        target_batches = batch["yq_eos"]
        model_logits = self(batch)
        logits_flat = model_logits.reshape(-1, model_logits.shape[-1])

        # compute loss
        test_loss = self.loss_fn(logits_flat, target_batches.reshape(-1))

        self.log(
            "test/test_loss",
            test_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,  # type: ignore
            sync_dist=True,
        )

        # Compute and log accuracy metrics
        acc_dict = evaluate_acc(
            batch=batch,
            model_logits=model_logits,
            max_length=self.hparams.max_length,  # type: ignore
            tokenizer=self.trainer.datamodule.tokenizer,  # type: ignore
        )
        self.log_acc_metrics(
            acc_dict=acc_dict, batch_size=self.hparams.batch_size, mode="test"  # type: ignore
        )

        return {"test_loss": test_loss, **acc_dict}

    def log_acc_metrics(
        self, acc_dict: Dict, batch_size: int, mode: Literal["train", "val", "test"]
    ) -> None:
        """
        Logs accuracy metrics to the tensorboard logger.

        Args:
            acc_dict (Dict): A dictionary containing accuracy metrics.
            batch_size (int): Batch size.
            mode (Literal["train", "val", "test"]): Mode to log metrics in.

        Returns:
            None
        """
        self.log(
            f"{mode}/{mode}_query_acc",
            acc_dict["query_acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{mode}/{mode}_copy_acc",
            acc_dict["copy_acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{mode}/{mode}_full_episode_acc",
            acc_dict["full_episode_acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{mode}/{mode}_full_episode_color_acc",
            acc_dict["full_episode_color_acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{mode}/{mode}_full_episode_shape_acc",
            acc_dict["full_episode_shape_acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{mode}/{mode}_copy_color_acc",
            acc_dict["copy_color_acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{mode}/{mode}_copy_shape_acc",
            acc_dict["copy_shape_acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{mode}/{mode}_query_color_acc",
            acc_dict["query_color_acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            f"{mode}/{mode}_query_shape_acc",
            acc_dict["query_shape_acc"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
            sync_dist=True,
        )

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate schedulers for the model.

        Returns:
            tuple: A tuple containing the optimizer and a list of scheduler configurations.
        """
        total_steps = self.trainer.estimated_stepping_batches

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            weight_decay=self.hparams.weight_decay,
        )

        # Warm-up: linearly increase LR from 0.001*1e-4 to 0.001 over one epoch
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=int(total_steps * 0.1),
        )

        # After warm-up: linearly decay LR from 0.001 to 0.001*0.05 (i.e. 5e-5) over the remaining epochs
        scheduler_epoch = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.hparams.lr_end_factor,
            total_iters=self.hparams.nepochs - 1,
        )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler_warmup,
                    "interval": "step",  # Step-level scheduler
                    "frequency": 1,
                    "name": "warmup_scheduler",
                },
                {
                    "scheduler": scheduler_epoch,
                    "interval": "epoch",  # Epoch-level scheduler
                    "frequency": 1,
                    "name": "epoch_scheduler",
                },
            ],
        )
