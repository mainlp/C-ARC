"""
Generate static dataset based on six statically defined transformations that don't change between episodes.
"""

import argparse
import os
from itertools import product
from typing import Any, Callable, Dict, List, Set, Tuple

from tqdm import tqdm

from vmlc.utils import (
    plot_sample,
    save_dict_to_json,
    save_dicts_as_jsonl,
    set_seed,
    setup_logging,
)
from vmlc.visual_grammar import (
    Group,
    gen_static_episode_data,
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


def shape_based_transformations() -> Tuple[List[str], List[Dict[str, Any]]]:
    shape_transformation_shapes = ["00010212", "0110112021"]
    shape_transformation_dicts = [
        {
            "transformation": TRANSFORMATIONS["grow"],
            "name": "grow",
            "transformation_kwargs": {"direction": "-y"},
        },
        {
            "transformation": TRANSFORMATIONS["translation"],
            "name": "translation",
            "transformation_kwargs": {"direction": "y", "step_size": 1},
        },
    ]
    return shape_transformation_shapes, shape_transformation_dicts


def color_based_transformations() -> Tuple[List[int], List[Dict[str, Any]]]:
    color_transformation_colors = [2, 6]
    color_transformation_dicts = [
        {
            "transformation": TRANSFORMATIONS["rotation"],
            "name": "rotation",
            "transformation_kwargs": {"rotation_type": 0},
        },
        {
            "transformation": TRANSFORMATIONS["mirror"],
            "name": "mirror",
            "transformation_kwargs": {"direction": "x"},
        },
    ]
    return color_transformation_colors, color_transformation_dicts


def indicator_based_transformations() -> (
    Tuple[List[int], List[Group], List[Dict[str, Any]]]
):
    indicator_transformation_colors = [1, 3]
    indicator_transformation_groups = [
        Group(coordinates=[(8, 4)], color=indicator_transformation_colors[0]),
        Group(coordinates=[(0, 7)], color=indicator_transformation_colors[1]),
    ]
    indicator_transformation_dicts = [
        {
            "transformation": TRANSFORMATIONS["translation"],
            "name": "translation",
            "transformation_kwargs": {"direction": "x", "step_size": 1},
        },
        {
            "transformation": TRANSFORMATIONS["grow"],
            "name": "grow",
            "transformation_kwargs": {"direction": "-x"},
        },
    ]
    return (
        indicator_transformation_colors,
        indicator_transformation_groups,
        indicator_transformation_dicts,
    )


def gen_static_samples(
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

    shape_transformation_shapes, shape_transformation_dicts = (
        shape_based_transformations()
    )
    color_transformation_colors, color_transformation_dicts = (
        color_based_transformations()
    )
    (
        indicator_transformation_colors,
        indicator_transformation_groups,
        indicator_transformation_dicts,
    ) = indicator_based_transformations()

    free_colors = [
        color_id
        for color_id in range(1, 10)
        if color_id not in color_transformation_colors + indicator_transformation_colors
    ]

    unique_combinations = list(
        product([0, 1], [0, 1], [0, 1])
    )  # get unique indicee combinations

    for sample_num, idx_combination in tqdm(
        enumerate(unique_combinations), desc="Generating Static Episodes"
    ):
        shape_idx = idx_combination[0]
        other_shape_idx = 0 if shape_idx == 1 else 1
        color_idx = idx_combination[1]
        indicator_idx = idx_combination[2]
        transformation_mapping = {
            "shape_transformation": shape_transformation_dicts[shape_idx],
            "color_transformation": color_transformation_dicts[color_idx],
            "indicator_transformation": indicator_transformation_dicts[indicator_idx],
        }
        color_transformation_color = color_transformation_colors[color_idx]
        shape_transformation_shape = shape_transformation_shapes[shape_idx]
        other_shape_transformation_shape = shape_transformation_shapes[other_shape_idx]
        indicator_group = indicator_transformation_groups[indicator_idx]
        indicator_color = indicator_transformation_colors[indicator_idx]

        # hash
        color_hash = (
            str(color_transformation_color)
            + str(color_transformation_dicts[color_idx]["name"])
            + str(color_transformation_dicts[color_idx]["transformation_kwargs"])
        )

        indicator_hash = (
            str(indicator_color)
            + "00"
            + str(indicator_transformation_dicts[indicator_idx]["name"])
            + str(
                indicator_transformation_dicts[indicator_idx]["transformation_kwargs"]
            )
        )

        shape_hash = (
            shape_transformation_shape
            + str(shape_transformation_dicts[shape_idx]["name"])
            + str(shape_transformation_dicts[shape_idx]["transformation_kwargs"])
        )
        full_hash = color_hash + indicator_hash + shape_hash

        sample_hash = {
            "color_hash": color_hash,
            "shape_hash": shape_hash,
            "indicator_hash": indicator_hash,
            "full_hash": full_hash,
        }

        episode, hashset = gen_static_episode_data(
            transformation_mapping=transformation_mapping,
            free_colors=free_colors,
            color_transformation_color=color_transformation_color,
            shape_transformation_shape=shape_transformation_shape,
            indicator_group=indicator_group,
            sample_hash=sample_hash,
            num_primitives=num_primitives,
            num_func_compositions=num_func_compositions,
            num_queries=num_queries,
            hashset=hashset,
            grid_size=grid_size,
            num_tries=max_tries,
            forbidden_shapes=[other_shape_transformation_shape],
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
    data_config = {
        "transformations": {
            "translation": {"direction": ["x", "y"], "step_size": [1]},
            "mirror": {"direction": ["x"]},
            "rotation": {"rotation_type": [0]},
            "grow": {"direction": ["x", "y"]},
        }
    }

    # save metadata
    meta_data: Dict[str, Any] = {"data_config": data_config, "args": vars(args)}
    save_dict_to_json(
        data=meta_data,
        file_path=os.path.join(args.data_dir, "config_all_episodes.json"),
    )

    # generate episodes
    episodes, hashset = gen_static_samples(
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
