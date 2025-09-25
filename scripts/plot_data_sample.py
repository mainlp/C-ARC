"""
Plot a specific data sample based on the sample's index.
"""

import argparse
import logging
import os

from vmlc.utils import load_jsonl, plot_sample, setup_logging


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Fetch CLI arguments
    parser = argparse.ArgumentParser("Data sample plotting.")

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The path to where the data is stored.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="experimental_results/data/plots",
        help="The path to where the plots should be created.",
    )

    # Dataset configs
    parser.add_argument(
        "--indicees", type=int, nargs="+", required=True, help="The indicees to plot."
    )
    parser.add_argument(
        "--max_columns", type=int, required=True, help="How many columns to plot."
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to orchestrate the execution flow.
    """
    args = parse_arguments()
    setup_logging(args.verbose)

    # get data
    file_name = args.data_path.split(".jsonl")[0].split("/")[-1]
    episodes = load_jsonl(file_path=args.data_path)
    episode_len = len(episodes)
    logging.info(f"{episode_len} episodes loaded.")

    # plot samples
    for index in args.indicees:
        assert index >= 0 and index < episode_len, f"Invalid index: {index}."
        episode_to_plot = episodes[index]

        plot_sample(
            episode_to_plot,
            os.path.join(args.plot_dir, f"{file_name}_episode_{index}.png"),
            max_columns=args.max_columns,
        )
        logging.info(f"Plot for episode {index} created.")


if __name__ == "__main__":
    main()
