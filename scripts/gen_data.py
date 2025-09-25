"""
Generate datasets based on visual grammar.
"""

import argparse
import os
from typing import Any, Callable, Dict, List, Set, Tuple

from tqdm import tqdm

from vmlc.utils import (
    load_yaml_file,
    plot_sample,
    save_dict_to_json,
    save_dicts_as_jsonl,
    set_seed,
    setup_logging,
)
from vmlc.visual_grammar import (
    gen_episode,
    grow,
    mirror,
    remove_row_column,
    rotation,
    set_color,
    translation,
)

TRANSFORMATIONS: Dict[str, Callable] = {
    "translation": translation,
    "mirror": mirror,
    "rotation": rotation,
    "set_color": set_color,
    "remove": remove_row_column,
    "grow": grow,
}


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Fetch CLI arguments
    parser = argparse.ArgumentParser("Data generation.")

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--seed", type=int, default=1860, help="Random generator seed.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The path to where the data should be stored.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="experimental_results/data/plots",
        help="The path to where the plots should be created.",
    )
    parser.add_argument(
        "--plot_freq",
        type=int,
        default=0,
        help="Frequency with which to generate plots of data.",
    )

    # Dataset configs
    parser.add_argument(
        "--num_samples", type=int, default=10e5, help="Number of dataset samples."
    )

    # Grid Configs
    parser.add_argument(
        "--num_primitives", type=int, default=4, help="Number of primitive examples."
    )
    parser.add_argument(
        "--num_func_compositions",
        type=int,
        default=4,
        help="Number of function composition examples.",
    )
    parser.add_argument(
        "--num_queries", type=int, default=14, help="Number of queries."
    )
    parser.add_argument(
        "--grid_size", type=int, nargs="+", default=[10, 10], help="Grid size."
    )
    parser.add_argument(
        "--max_tries",
        type=int,
        default=500,
        help="Number of tries to generate mapping for shape.",
    )

    return parser.parse_args()


def gen_samples(
    transformations: List[Callable],
    transformations_kwarg_options: Dict[str, Dict],
    num_samples: int,
    num_primitives: int,
    num_func_compositions: int,
    num_queries: int,
    hashset: Set[str],
    grid_size: Tuple[int, int] = (10, 10),
    max_tries: int = 500,
    plot_freq: int = 0,
    plot_dir: str = "experimental_results/data/plots",
    data_dir: str = "data",
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Generates samples for training, validation, or testing.

    Args:
        transformations (List[Callable]): List of possible transformations.
        transformations_kwarg_options (Dict[str, Dict]): Dictionary with possible transformation kwargs.
        num_samples (int): The number of samples to generate.
        num_primitives (int): The number of primitive examples.
        num_func_compositions (int): The number of function composition examples.
        num_queries (int): The number of queries.
        hashset (Set[str]): A set of transformation hashes.
        grid_size (Tuple[int, int], optional): The size of the grid. Defaults to (10, 10).
        max_tries (int, optional): The maximum number of tries to generate a mapping for a shape. Defaults to 500.
        plot_freq (int, optional): The frequency of plotting samples. Defaults to 0.
        plot_dir (str, optional): The directory to store plots. Defaults to "experimental_results/data/plots".

    Returns:
        Tuple[List[dict], Set[str]]: A tuple containing episodes and updated hashset.
    """
    episode_list: List[Dict[str, Any]] = []

    for sample_num in tqdm(range(num_samples), desc="Generating Episodes"):
        episode, hashset = gen_episode(
            transformations=transformations,
            transformations_kwarg_options=transformations_kwarg_options,
            num_primitives=num_primitives,
            num_func_compositions=num_func_compositions,
            num_queries=num_queries,
            hashset=hashset,
            grid_size=grid_size,
            num_tries=max_tries,
        )
        episode_list.append(episode)

        if plot_freq > 0 and sample_num % plot_freq == 0:
            plot_sample(
                episode,
                os.path.join(plot_dir, f"episode_{sample_num}.png"),
            )

        if sample_num % 1000 == 0:
            save_dicts_as_jsonl(
                data=episode_list,
                filepath=os.path.join(data_dir, "all_episodes.jsonl"),
            )

    return episode_list, hashset


def main() -> None:
    """
    Main function to orchestrate the execution flow.
    """
    args = parse_arguments()

    setup_logging(args.verbose)
    set_seed(args.seed)

    # hash sets for dataset distribution
    hashset: Set[str] = set()

    # get data
    assert (
        len(args.grid_size) == 2
    ), f"grid_size argument must be tuple (height, width). Not: {args.grid_size}"

    # get data config file
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    config_file_path = os.path.join(
        os.path.dirname(current_dir), "vmlc", "configs", "data_config.yaml"
    )
    data_config = load_yaml_file(config_file_path)

    # save metadata
    meta_data: Dict[str, Any] = {"data_config": data_config, "args": vars(args)}
    save_dict_to_json(
        data=meta_data,
        file_path=os.path.join(args.data_dir, "config_all_episodes.json"),
    )

    # get transformations
    transformations = [
        TRANSFORMATIONS[transformation_name]
        for transformation_name in data_config["transformations"]
    ]
    transformations_kwarg_options = {
        transformation_name: transformation_kwargs
        for transformation_name, transformation_kwargs in data_config[
            "transformations"
        ].items()
    }

    # generate episodes
    episodes, hashset = gen_samples(
        transformations=transformations,
        transformations_kwarg_options=transformations_kwarg_options,
        num_samples=int(args.num_samples),
        num_primitives=args.num_primitives,
        num_func_compositions=args.num_func_compositions,
        num_queries=args.num_queries,
        hashset=hashset,
        grid_size=args.grid_size,
        max_tries=args.max_tries,
        plot_freq=args.plot_freq,
        plot_dir=args.plot_dir,
        data_dir=args.data_dir,
    )

    # save data
    save_dicts_as_jsonl(
        data=episodes,
        filepath=os.path.join(args.data_dir, "all_episodes.jsonl"),
    )


if __name__ == "__main__":
    main()
