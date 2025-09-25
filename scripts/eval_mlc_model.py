"""
Evaluate trained MLC model on test dataset.
"""

import argparse
import logging
import os

import pytorch_lightning as pl
import torch

from vmlc.data_prep.datamodule import LitTestDataModule
from vmlc.data_prep.patch_tokenizer import PatchTokenizer
from vmlc.model.model import LitModel
from vmlc.utils import save_dict_to_json, set_seed, setup_logging


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line input arguments for model evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MLC model on the test set."
    )

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file to load.",
    )
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
        help="The name of the data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the test evaluation results.",
    )

    # Eval configs
    parser.add_argument(
        "--batch_size", type=int, default=25, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--test_queries",
        type=int,
        default=-1,
        help="Number of queries to evaluate on.",
    )
    parser.add_argument(
        "--p_noise",
        type=float,
        default=0.0,
        help="Probability to apply noise to input patches.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=2,
        help="Patch size used by the tokenizer.",
    )
    parser.add_argument(
        "--no_copy_task",
        action="store_false",
        dest="with_copy_task",
        default=True,
        help="Do not include copy task (by default, copy task is enabled).",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function for evaluation.
    """
    args = parse_arguments()

    # Set up logging and seed
    setup_logging(verbosity=args.verbose)
    pl.seed_everything(args.seed)
    set_seed(args.seed)

    # Initialize the tokenizer (using grid_size 10 as in training)
    grid_size = 10
    tokenizer = PatchTokenizer(
        patch_size=args.patch_size, max_value=9, grid_size=grid_size
    )

    # Initialize the DataModule for evaluation and prepare the test data
    data_module = LitTestDataModule(
        data_dir=args.data_dir,
        data_file_name=args.data_file_name,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        p_noise=args.p_noise,
        with_copy_task=args.with_copy_task,
        test_queries=args.test_queries,
        num_workers=4,
        prefetch_factor=2,
    )
    data_module.setup()
    logging.info("Data module set up for evaluation.")

    # Load the trained model from the provided checkpoint
    model = LitModel.load_from_checkpoint(args.checkpoint_path)
    logging.info(f"Model loaded from checkpoint: {args.checkpoint_path}")

    # Configure the PyTorch Lightning trainer for evaluation
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
    )

    # Evaluate the model on the test set
    test_results = trainer.test(model, datamodule=data_module)
    result = (
        test_results[0]
        if isinstance(test_results, list) and test_results
        else test_results
    )
    logging.info(f"Results:\n{result}")

    # Save evaluation results
    output_file = os.path.join(args.output_dir, "model_accuracy.json")
    save_dict_to_json(data=dict(result), file_path=output_file)


if __name__ == "__main__":
    main()
