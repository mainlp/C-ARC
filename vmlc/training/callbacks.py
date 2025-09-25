import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from vmlc.model.eval_plots import display_console_pred, display_error_pred


def get_checkpoint_callback(dirpath="model_checkpoints/"):
    return ModelCheckpoint(
        dirpath=dirpath,
        filename="lit_model-{epoch:02d}-{val_loss:.4f}",
        monitor="val/val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )


class PlotCallback(pl.Callback):
    def __init__(
        self,
        plot_every_n_epochs: int = 5,
        savedir: str = "plots",
        with_copy_task: bool = True,
        include_error_plots: bool = True,
    ):
        """
        Callback to plot validation results every N epochs.
        """
        self.plot_every_n_epochs = plot_every_n_epochs
        self.savedir = savedir
        self.with_copy_task = with_copy_task
        self.include_error_plots = include_error_plots

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            trainer.current_epoch % self.plot_every_n_epochs == 0
            and trainer.current_epoch != 0
        ):
            pl_module.print(f"Generating train plots for epoch {trainer.current_epoch}")
            predictions = pl_module.train_predictions
            if predictions:
                plot_name = f"epoch_{trainer.current_epoch}.png"

                display_console_pred(
                    predictions,
                    os.path.join(self.savedir, f"train_sample_{plot_name}"),
                    self.with_copy_task,
                )
                if self.include_error_plots:
                    display_error_pred(
                        predictions,
                        os.path.join(self.savedir, f"train_error_{plot_name}"),
                    )
            else:
                pl_module.print("No valid samples found for plotting.")
        pl_module.train_predictions = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Runs after validation ends. If the current epoch is a plotting epoch,
        retrieves stored predictions generates plots, and then clears stored predictions.
        """
        if trainer.current_epoch != 0:
            pl_module.print(f"Generating val plots for epoch {trainer.current_epoch}")

            predictions = pl_module.validation_predictions
            if predictions:
                plot_name = f"epoch_{trainer.current_epoch}.png"

                display_console_pred(
                    predictions,
                    os.path.join(self.savedir, f"val_sample_{plot_name}"),
                    self.with_copy_task,
                )
                if self.include_error_plots:
                    display_error_pred(
                        predictions,
                        os.path.join(self.savedir, f"val_error_{plot_name}"),
                    )
            else:
                pl_module.print("No valid samples found for plotting.")
        pl_module.validation_predictions = []
