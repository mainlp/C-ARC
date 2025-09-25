"""
Script to pre-process data for static data systematicity experiment.
We split the data such that each sample consists only of an (input, output) tuple (no few-shot examples).
"""

import argparse
import logging
import os
import random
from typing import Any, Dict, List, Tuple

from vmlc.utils import (
    load_jsonl,
    save_dict_to_json,
    save_dicts_as_jsonl,
    set_seed,
    setup_logging,
)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Fetch CLI arguments
    parser = argparse.ArgumentParser("Split data for systematicity experiment.")

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
        "--data_path",
        type=str,
        required=True,
        help="The directory to where the data is stored.",
    )

    # Dataset configs
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of dataset to use samples. Takes all if 0.",
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=0,
        help="Number of training samples after filtering. Takes all if 0.",
    )
    parser.add_argument(
        "--num_test_samples",
        type=int,
        default=0,
        help="Number of test samples after filtering. Takes all if 0.",
    )

    # Sample Configs
    parser.add_argument(
        "--frac_test_compositions",
        type=float,
        default=0.2,
        help="Fraction of unique test compositions.",
    )

    return parser.parse_args()


def extract_transformations(
    sample: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Extracts the shape, color, and indicator transformations from a sample.

    Args:
        sample: A dictionary expected to have a "meta_data" key containing transformation info.

    Returns:
        A tuple containing the shape, color, and indicator transformation dictionaries.

    Raises:
        KeyError: If "meta_data" or any of the transformation keys are missing.
    """
    meta_data = sample.get("meta_data")
    if not isinstance(meta_data, dict):
        raise KeyError(f"Missing or invalid 'meta_data' in sample: {sample}")
    try:
        shape_transform = meta_data["shape_transformation"]
        color_transform = meta_data["color_transformation"]
        indicator_transform = meta_data["indicator_transformation"]
    except KeyError as e:
        raise KeyError(f"Missing transformation key in sample: {sample}") from e
    return shape_transform, color_transform, indicator_transform


def merge_update(original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    For each key in the update dictionary:
      - If the key exists in the original:
          - If both values are dictionaries, merge them recursively.
          - Otherwise, convert non-list values to lists and append the update value,
            ensuring no duplicate values are added.
      - If the key does not exist in the original, add it. If its value is not a list,
        wrap it in a list.

    Parameters:
        original (Dict[str, Any]): The original dictionary.
        update (Dict[str, Any]): The dictionary containing update values.

    Returns:
        Dict[str, Any]: The merged dictionary.
    """
    for key, update_value in update.items():
        if key in original:
            original_value = original[key]
            if isinstance(original_value, dict) and isinstance(update_value, dict):
                merge_update(original_value, update_value)
            else:
                # Ensure both values are lists
                if not isinstance(original_value, list):
                    original_value = [original_value]
                if not isinstance(update_value, list):
                    update_value = [update_value]
                for item in update_value:
                    if item not in original_value:
                        original_value.append(item)
                original[key] = original_value
        else:
            original[key] = (
                update_value if isinstance(update_value, list) else [update_value]
            )
    return original


def update_func_kwargs(
    func_kwargs: Dict[str, Dict[str, Any]],
    func_type: str,
    transformation: Dict[str, Any],
) -> None:
    """
    Update the function kwargs dictionary for a given function type using merge_update.

    Args:
        func_kwargs: Dictionary mapping function types to their kwargs.
        func_type: The key representing the function type.
        transformation: A transformation dict that should contain a "kwargs" key.
    """
    if func_type not in func_kwargs:
        func_kwargs[func_type] = {}
    kwargs_update = transformation.get("kwargs", {})
    func_kwargs[func_type] = merge_update(func_kwargs[func_type], kwargs_update)


def create_sample(meta_data: Dict[str, Any], query: Any) -> Dict[str, Any]:
    """
    Creates a sample dict for a given query and metadata.

    Args:
        meta_data: Metadata from the episode.
        query: A study example or query.

    Returns:
        A dictionary representing the sample.
    """
    return {"study_examples": [], "queries": [query], "meta_data": meta_data}


def add_study_examples(
    episode: Dict[str, Any], target_samples: List[Dict[str, Any]]
) -> None:
    """
    Adds study example samples from both primitive functions and function compositions to the target list.

    Args:
        episode: An episode dict containing study examples.
        target_samples: The list where samples should be appended.
    """
    for key, study_example_list in episode.get("primitive_functions", {}).items():
        for study_example in study_example_list:
            target_samples.append(create_sample(episode["meta_data"], study_example))
    for key, study_example_list in episode.get("function_compositions", {}).items():
        for study_example in study_example_list:
            target_samples.append(create_sample(episode["meta_data"], study_example))


def add_queries(episode: Dict[str, Any], target_samples: List[Dict[str, Any]]) -> None:
    """
    Adds query samples from the episode to the target list.

    Args:
        episode: An episode dict containing queries.
        target_samples: The list where samples should be appended.
    """
    for query in episode.get("queries", []):
        target_samples.append(create_sample(episode["meta_data"], query))


def process_episode(
    episode: Dict[str, Any],
    study_target: List[Dict[str, Any]],
    query_target: List[Dict[str, Any]],
    composition_target: List[List[str]],
    kwargs_target: Dict[str, Dict[str, Any]],
) -> None:
    """
    Processes an episode by extracting transformation info, updating function compositions and kwargs,
    and adding samples.

    Args:
        episode: The episode to process.
        study_target: List to which study examples will be added.
        query_target: List to which queries will be added.
        composition_target: List where the function composition (types) will be appended.
        kwargs_target: Dictionary mapping function types to their kwargs; will be updated.
    """
    shape_trans, color_trans, indicator_trans = extract_transformations(episode)
    shape_type = shape_trans.get("type")
    color_type = color_trans.get("type")
    indicator_type = indicator_trans.get("type")
    func_types = [shape_type, color_type, indicator_type]
    composition_target.append(func_types)  # type: ignore

    update_func_kwargs(kwargs_target, shape_type, shape_trans)  # type: ignore
    update_func_kwargs(kwargs_target, color_type, color_trans)  # type: ignore
    update_func_kwargs(kwargs_target, indicator_type, indicator_trans)  # type: ignore

    # Add study examples (from both primitive functions and function compositions).
    add_study_examples(episode, study_target)
    add_queries(episode, query_target)


def train_val_test_split(
    episodes: List[Dict[str, Any]],
    frac_test_compositions: float,
    num_train_samples: int = 0,
    num_test_samples: int = 0,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[List[str]],
    Dict[str, Dict[str, Any]],
    List[List[str]],
    Dict[str, Dict[str, Any]],
]:
    """
    Splits the provided episodes into training and testing sets, processes each episode,
    and creates sample dictionaries along with function compositions and kwargs.

    In this setup, for test episodes, study examples are merged into the training samples,
    while queries become test samples. The validation set is set equal to the test set.

    Args:
        episodes: List of episode dictionaries.
        frac_test_compositions: Fraction of episodes to allocate to the test set.
        num_train_samples: Optional limit on the number of training samples.
        num_test_samples: Optional limit on the number of test samples.

    Returns:
        A tuple containing:
          - train_samples (List[Dict[str, Any]])
          - test_samples (List[Dict[str, Any]])
          - val_samples (List[Dict[str, Any]]) (same as test_samples)
          - train_func_compositions (List[List[str]])
          - train_func_kwargs (Dict[str, Dict[str, Any]])
          - test_func_compositions (List[List[str]])
          - test_func_kwargs (Dict[str, Dict[str, Any]])
    """
    num_episodes = len(episodes)
    num_test_episodes = int(num_episodes * frac_test_compositions)
    num_train_episodes = num_episodes - num_test_episodes

    random.shuffle(episodes)
    train_episodes = episodes[:num_train_episodes]
    test_episodes = episodes[num_train_episodes:]

    train_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    train_func_compositions: List[List[str]] = []
    test_func_compositions: List[List[str]] = []

    train_func_kwargs: Dict[str, Dict[str, Any]] = {}
    test_func_kwargs: Dict[str, Dict[str, Any]] = {}

    # Process train episodes: both study examples and queries go to training samples.
    for episode in train_episodes:
        process_episode(
            episode,
            study_target=train_samples,
            query_target=train_samples,
            composition_target=train_func_compositions,
            kwargs_target=train_func_kwargs,
        )

    # Process test episodes: study examples go to training samples, queries to test samples.
    for episode in test_episodes:
        process_episode(
            episode,
            study_target=train_samples,
            query_target=test_samples,
            composition_target=test_func_compositions,
            kwargs_target=test_func_kwargs,
        )

    # Optionally limit the number of samples.
    if 0 < num_train_samples < len(train_samples):
        train_samples = train_samples[:num_train_samples]
    logging.info(f"{len(train_samples)} train samples used.")

    if 0 < num_test_samples < len(test_samples):
        test_samples = test_samples[:num_test_samples]
    logging.info(f"{len(test_samples)} test samples used.")

    # In this experiment, validation samples are the same as test samples.
    val_samples = test_samples

    return (
        train_samples,
        val_samples,
        test_samples,
        train_func_compositions,
        train_func_kwargs,
        test_func_compositions,
        test_func_kwargs,
    )


def main() -> None:
    """
    Main function to orchestrate the execution flow.
    """
    args = parse_arguments()

    setup_logging(args.verbose)
    set_seed(args.seed)

    # Load episodes.
    assert args.num_samples >= 0, f"Invalid number of samples: {args.num_samples}"
    episodes = load_jsonl(file_path=args.data_path)
    num_episodes = len(episodes)
    logging.info(f"{num_episodes} episodes loaded.")

    if args.num_samples == 0 or args.num_samples > num_episodes:
        args.num_samples = num_episodes
    episodes = episodes[: args.num_samples]
    logging.info(f"Using first {len(episodes)} episodes...")

    # Split episodes into train, validation, and test samples.
    (
        train_samples,
        val_samples,
        test_samples,
        train_func_compositions,
        train_func_kwargs,
        test_func_compositions,
        test_func_kwargs,
    ) = train_val_test_split(
        episodes=episodes,
        frac_test_compositions=args.frac_test_compositions,
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
    )

    # Save metadata.
    data_dir = os.path.dirname(os.path.dirname(args.data_path))
    split_dir = os.path.join(data_dir, f"static_split_seed_{args.seed}")
    save_dict_to_json(
        data={
            "train_func_compositions": train_func_compositions,
            "test_func_compositions": test_func_compositions,
            "train_func_kwargs": train_func_kwargs,
            "test_func_kwargs": test_func_kwargs,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "num_test_samples": len(test_samples),
            "args": vars(args),
        },
        file_path=os.path.join(
            split_dir, f"config_static_systematicity_seed_{args.seed}.json"
        ),
    )

    # Save the split datasets.
    save_dicts_as_jsonl(
        data=train_samples,
        filepath=os.path.join(
            split_dir, f"train_static_systematicity_seed_{args.seed}.jsonl"
        ),
    )
    save_dicts_as_jsonl(
        data=val_samples,
        filepath=os.path.join(
            split_dir, f"val_static_systematicity_seed_{args.seed}.jsonl"
        ),
    )
    save_dicts_as_jsonl(
        data=test_samples,
        filepath=os.path.join(
            split_dir, f"test_static_systematicity_seed_{args.seed}.jsonl"
        ),
    )


if __name__ == "__main__":
    main()
