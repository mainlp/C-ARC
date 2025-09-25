"""
Train MLC model.
"""

import argparse
import logging
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from vmlc.data_prep.datamodule import LitDataModule
from vmlc.data_prep.patch_tokenizer import PatchTokenizer
from vmlc.model.model import LitModel
from vmlc.training.callbacks import PlotCallback, get_checkpoint_callback
from vmlc.utils import set_seed, setup_logging


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line input arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Training script for VMLC algorithm")

    # General config
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the data split.",
    )
    parser.add_argument(
        "--data_file_name",
        type=str,
        required=True,
        help="The name of the data files.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default="wandb",
        help="Directory containing the wandb files.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="train_vmlc",
        help="Directory containing the wandb files.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="experimental_results/models/plots",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--plot_freq",
        type=int,
        default=0,
        help="Number of epochs between validation plots. Set to 0 to disable plotting.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Used when performing hyperparameter search. If specified, don't save model checkpoints.",
    )

    # Data config
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum generated sequence length.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=2,
        help="Size of each patch for patch-based processing.",
    )
    parser.add_argument(
        "--p_noise",
        type=float,
        default=0.001,
        help="Probability to replace output grids uniformly to ensure decoder-robustness.",
    )

    # Train config
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25,
        help="number of episodes per batch.",
    )
    parser.add_argument(
        "--batch_hold_update",
        type=int,
        default=2,
        help="update the weights after this many batches (default=1).",
    )
    parser.add_argument(
        "--nepochs",
        type=int,
        default=50,
        help="number of training epochs.",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=10,
        help="Validate every nth epoch.",
    )
    parser.add_argument(
        "--no_copy_task",
        default=True,
        action="store_false",
        dest="with_copy_task",
        help="Do not include copy task.",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate.")
    parser.add_argument(
        "--lr_end_factor",
        type=float,
        default=0.05,
        help="factor X for decrease learning rate linearly from 1.0*lr to X*lr across training.",
    )
    parser.add_argument(
        "--matmul_precision",
        type=str,
        default="medium",
        choices=[None, "medium", "high"],
        help="Matmul precision for GPU optimization.",
    )
    parser.add_argument(
        "--overfit_batches",
        type=float,
        default=0.0,
        help="Overfit on a small portion of the data for debugging.",
    )
    parser.add_argument(
        "--resume_from_ckpt",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from. If not provided, training starts from scratch.",
    )

    # Model config
    parser.add_argument(
        "--nlayers_encoder", type=int, default=3, help="number of layers for encoder."
    )
    parser.add_argument(
        "--nlayers_decoder", type=int, default=3, help="number of layers for decoder."
    )
    parser.add_argument("--emb_size", type=int, default=128, help="size of embedding.")
    parser.add_argument(
        "--ff_mult",
        type=int,
        default=6,
        help="multiplier for size of the fully-connected layer in transformer.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout applied to embeddings and transformer.",
    )
    parser.add_argument(
        "--zero_token_weight",
        type=float,
        default=0.2,
        help="Loss weight for zero patch token.",
    )
    parser.add_argument(
        "--act",
        type=str,
        default="gelu",
        choices=["relu", "gelu"],
        help="activation function in the fully-connected layer of the transformer (relu or gelu).",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    """
    Main Script
    """
    args = parse_arguments()

    setup_logging(args.verbose)
    set_seed(args.seed)
    pl.seed_everything(args.seed)

    if args.matmul_precision is not None:
        torch.set_float32_matmul_precision(args.matmul_precision)

    # Initialize tokenizer
    grid_size = 10

    tokenizer = PatchTokenizer(
        patch_size=args.patch_size, max_value=9, grid_size=grid_size
    )
    output_size = tokenizer.vocab_size
    PAD_idx = tokenizer.PAD_idx
    zero_idx = tokenizer.zero_token_id

    # Initialize the DataModule
    data_module = LitDataModule(
        data_dir=args.data_dir,
        data_file_name=args.data_file_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        p_noise=args.p_noise,
        with_copy_task=args.with_copy_task,
        num_workers=4,
        prefetch_factor=2,
    )
    data_module.setup()
    num_support = len(data_module.train_dataset[0]["xs"])
    logging.info("Data loaded.")

    # Initialize the model
    model = LitModel(
        hidden_size=args.emb_size,
        output_size=output_size,
        PAD_idx_input=PAD_idx,
        PAD_idx_output=PAD_idx,
        nlayers_encoder=args.nlayers_encoder,
        nlayers_decoder=args.nlayers_decoder,
        ff_mult=args.ff_mult,
        dropout_p=args.dropout,
        activation=args.act,
        grid_size=(grid_size, grid_size),
        patch_size=(args.patch_size, args.patch_size),
        num_support=num_support,
        lr=args.lr,
        lr_end_factor=args.lr_end_factor,
        nepochs=args.nepochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        all_zero_token_idx=zero_idx,
        zero_token_weight=args.zero_token_weight,
    )
    logging.info("Model initialized.")

    # Initialize Callbacks
    wandb_logger = WandbLogger(
        entity="your_entity",
        project="MLC",
        save_dir=args.wandb_dir,
        name=args.wandb_name,
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = get_checkpoint_callback(dirpath=args.checkpoint_dir)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, lr_monitor]

    if args.plot_freq > 0:
        os.makedirs(args.plot_dir, exist_ok=True)
        callbacks.append(
            PlotCallback(args.plot_freq, args.plot_dir, args.with_copy_task)
        )

    # Configure Trainer
    trainer = pl.Trainer(
        max_epochs=args.nepochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        enable_checkpointing=False if args.sweep else True,
        callbacks=None if args.sweep else callbacks,
        log_every_n_steps=10,
        logger=wandb_logger,
        accumulate_grad_batches=args.batch_hold_update,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        overfit_batches=args.overfit_batches,
    )

    # Train
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_ckpt)

    # Test
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
