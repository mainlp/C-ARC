"""
Script to pre-process data for systematicity experiment.
"""

import argparse
import itertools
import logging
import os
import random
from collections import Counter
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

from tqdm import tqdm

from vmlc.utils import (
    load_jsonl,
    load_yaml_file,
    save_dict_to_json,
    save_dicts_as_jsonl,
    set_seed,
    setup_logging,
)
from vmlc.visual_grammar import (
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

    # Sample Configs
    parser.add_argument(
        "--frac_test_compositions",
        type=float,
        default=0.2,
        help="Fraction of unique test compositions.",
    )
    parser.add_argument(
        "--min_func_examples",
        type=int,
        default=10,
        help="Minimum number of training samples that represent function (with specific kwargs).",
    )
    parser.add_argument(
        "--no_shuffle_study_examples",
        default=True,
        action="store_false",
        dest="shuffle_study_examples",
        help="Do not shuffle study examples.",
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
    parser.add_argument(
        "--num_primitives",
        type=int,
        default=3,
        help="Number of primitive function examples.",
    )
    parser.add_argument(
        "--num_compositions", type=int, default=3, help="Number of primitive examples."
    )
    parser.add_argument(
        "--num_queries", type=int, default=10, help="Number of queries for training."
    )
    parser.add_argument(
        "--num_query_few_shots",
        type=int,
        default=0,
        help="Number of query few-shot examples.",
    )

    return parser.parse_args()


def split_triplet_compositions(
    elements: List[str], frac_test_compositions: float
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    Generate all possible unique triplet combinations from the provided list of elements and
    randomly split them into training and testing compositions.

    Args:
        elements (List[str]): A list of strings representing the elements.
        num_test_comps (int): The number of test compositions desired. This value must be between 0
                              and the total number of possible triplet combinations.

    Returns:
        Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
            - train_compositions: A list of triplets (tuples) used for training.
            - test_compositions: A list of triplets (tuples) used for testing, of length `num_test_comps`.

    Raises:
        ValueError: If `num_test_comps` is negative or exceeds the total number of triplet combinations.
    """
    all_triplets = list(itertools.combinations(elements, 3))
    total_triplets = len(all_triplets)
    num_test_comps = int(frac_test_compositions * total_triplets)

    # Validate the number of test compositions requested.
    if num_test_comps < 0 or num_test_comps > total_triplets:
        raise ValueError(
            f"num_test_comps must be between 0 and {total_triplets} (inclusive). "
            f"Provided value: {num_test_comps}."
        )
    random.shuffle(all_triplets)
    test_compositions = all_triplets[:num_test_comps]
    train_compositions = all_triplets[num_test_comps:]

    return train_compositions, test_compositions


def filter_num_examples(
    valid_sample: Dict,
    num_primitive_functions: int,
    num_func_compositions: int,
    num_query_functions: int,
    num_query_few_shots: int = 0,
    shuffle: bool = True,
) -> Dict[str, Dict]:
    """
    Filters the given sample based on the given number of examples for each type
    of example (primitive_functions, function_compositions, queries). The sample
    is modified in-place.

    Args:
        valid_sample (Dict): The sample to be filtered.
        num_primitive_functions (int): The number of primitive function examples to be kept.
        num_func_compositions (int): The number of function composition examples to be kept.
        num_query_functions (int): The number of query function examples to be kept.
        num_query_few_shots (int, Optional): The number of query few shot examples. Defaults to 0.
        shuffle (bool, Optional): Whether to shuffle order of study examples. Defaults to False.
    """
    final_sample: Dict[str, Any] = {"study_examples": []}

    for key, value in valid_sample.items():
        if key == "meta_data":
            final_sample[key] = value
        elif key == "primitive_functions":
            for indicator, examples in valid_sample[key].items():
                final_sample["study_examples"].extend(
                    examples[:num_primitive_functions]
                )
        elif key == "function_compositions":
            for indicator, examples in valid_sample[key].items():
                final_sample["study_examples"].extend(examples[:num_func_compositions])
        elif key == "queries":
            final_sample[key] = valid_sample[key][:num_query_functions]
            if num_query_few_shots > 0:
                assert num_query_functions + num_query_few_shots <= len(
                    valid_sample[key]
                ), "Not enough queries for query and query few-shot specifications!"
                final_sample["study_examples"].extend(
                    valid_sample[key][
                        num_query_functions : num_query_functions + num_query_few_shots
                    ]
                )
        else:
            raise ValueError(f"Unexpected key: {key}")

    if shuffle:
        random.shuffle(final_sample["study_examples"])

    return final_sample


def filter_for_func_kwargs(
    func_kwargs: Dict[str, Any], allowable_function_kwargs: Dict[str, Any]
) -> bool:
    """
    Checks if the given function kwargs are valid based on the allowable kwargs.

    Args:
        func_kwargs (Dict[str, Any]): The function kwargs to be checked.
        allowable_function_kwargs (Dict[str, Any]): The allowable function kwargs.

    Returns:
        bool: True if the function kwargs are valid, False otherwise.
    """
    for arg_name, arg_value in func_kwargs.items():
        if arg_name == "grid_size":
            continue

        if arg_value not in allowable_function_kwargs[arg_name]:
            return False

    return True


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


def is_valid_triplet(
    shape_type: str,
    color_type: str,
    indicator_type: str,
    valid_triplets_set: Set[FrozenSet[str]],
) -> bool:
    """
    Checks if the set of transformation types forms a valid triplet.

    Args:
        shape_type: The type of the shape transformation.
        color_type: The type of the color transformation.
        indicator_type: The type of the indicator transformation.
        valid_triplets_set: A set of valid frozensets of transformation types.

    Returns:
        True if the triplet is valid; False otherwise.
    """
    sample_triplet = frozenset({shape_type, color_type, indicator_type})
    return sample_triplet in valid_triplets_set


def are_valid_function_kwargs(
    func_names: List[str],
    transforms: List[Dict[str, Any]],
    allowable_function_kwargs: Dict[str, Dict],
) -> bool:
    """
    Validates that the keyword arguments for each function meet the allowable criteria.

    Args:
        func_names: A list of function names (as strings).
        transforms: A list of transformation dictionaries corresponding to each function.
        allowable_function_kwargs: A mapping from function names to their allowed kwargs.

    Returns:
        True if all function kwargs are valid; False otherwise.
    """
    for func_name, transform in zip(func_names, transforms):
        func_kwargs = transform.get("kwargs", {})
        allowed_kwargs = allowable_function_kwargs.get(func_name, {})

        if not filter_for_func_kwargs(
            func_kwargs=func_kwargs, allowable_function_kwargs=allowed_kwargs
        ):
            return False
    return True


def update_and_check_func_kwargs_counter(
    mode: Literal["train", "test"],
    func_kwargs_counter: Counter[str],
    func_names: List[str],
    transforms: List[Dict[str, Any]],
    min_func_examples: int,
) -> bool:
    """
    Updates or checks the function keyword arguments counter based on the mode.

    In 'train' mode, the counter is updated with the current sample's kwargs label.
    In 'test' mode, the function checks if each kwargs label meets the minimum examples.

    Args:
        mode: Either "train" or "test".
        func_kwargs_counter: A Counter tracking occurrences of function kwargs labels.
        func_names: A list of function names (as strings) for the current sample.
        transforms: A list of transformation dictionaries corresponding to each function.
        min_func_examples: The minimum required examples for a function's kwargs in test mode.

    Returns:
        True if the sample meets the primitive function requirements; False otherwise.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    valid_primitives = True
    for func_name, transform in zip(func_names, transforms):
        func_kwargs = transform.get("kwargs", {})

        # Create a label by joining the function name with its kwargs key-value pairs.
        func_kwarg_label = f"{func_name} " + " ".join(
            f"{key}_{value}" for key, value in func_kwargs.items()
        )
        if mode == "train":
            func_kwargs_counter[func_kwarg_label] += 1
        elif mode == "test":
            if func_kwargs_counter[func_kwarg_label] < min_func_examples:
                valid_primitives = False
        else:
            raise ValueError(f"Invalid mode: {mode}")
    return valid_primitives


@overload
def filter_samples_by_composition(
    samples: List[Dict[str, Any]],
    valid_triplets: List[Tuple[str, str, str]],
    num_primitive_functions: int,
    num_func_compositions: int,
    num_query_functions: int,
    allowable_function_kwargs: Dict[str, Dict],
    mode: Literal["test"],
    num_query_few_shots: int = 0,
    shuffle_study_examples: bool = True,
    min_func_examples: int = 0,
    func_kwargs_counter: Optional[Counter[str]] = None,
) -> List[Dict[str, Any]]: ...


@overload
def filter_samples_by_composition(
    samples: List[Dict[str, Any]],
    valid_triplets: List[Tuple[str, str, str]],
    num_primitive_functions: int,
    num_func_compositions: int,
    num_query_functions: int,
    allowable_function_kwargs: Dict[str, Dict],
    mode: Literal["train"],
    num_query_few_shots: int = 0,
    shuffle_study_examples: bool = True,
    min_func_examples: int = 0,
    func_kwargs_counter: Optional[Counter[str]] = None,
) -> Tuple[List[Dict[str, Any]], Counter[str]]: ...


def filter_samples_by_composition(
    samples: List[Dict[str, Any]],
    valid_triplets: List[Tuple[str, str, str]],
    num_primitive_functions: int,
    num_func_compositions: int,
    num_query_functions: int,
    allowable_function_kwargs: Dict[str, Dict],
    mode: Literal["train", "test"],
    num_query_few_shots: int = 0,
    shuffle_study_examples: bool = True,
    min_func_examples: int = 0,
    func_kwargs_counter: Optional[Counter[str]] = None,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Counter[str]]]:
    """
    Filters samples based on valid transformation compositions (triplets), function kwargs, and function primitives.

    Args:
        samples: A list of sample dictionaries. Each sample must contain a "meta_data" key with the expected structure.
        valid_triplets: A list of valid transformation triplets, where each triplet is a tuple of three strings.
        num_primitive_functions: Number of primitive functions required.
        num_func_compositions: Number of function compositions required.
        num_query_functions: Number of query functions required.
        allowable_function_kwargs: A mapping from function names to allowed keyword arguments.
        shuffle_study_examples: Shuffle study examples.
        mode: Operation mode, either "train" or "test".
        num_query_few_shots: The number of query few shot examples. Defaults to 0.
        min_func_examples: Minimum required examples for a function's kwargs in test mode (default is 0).
        func_kwargs_counter: Optional counter for tracking function kwargs occurrences (if None, a new Counter is created).

    Returns:
        In 'train' mode, a tuple of (filtered_samples, updated func_kwargs_counter).
        In 'test' mode, just the list of filtered samples.

    Raises:
        KeyError: If a sample is missing expected keys or structure.
        ValueError: If an invalid mode is provided.
    """
    if func_kwargs_counter is None:
        func_kwargs_counter = Counter()

    valid_triplets_set: Set[FrozenSet[str]] = {
        frozenset(triplet) for triplet in valid_triplets
    }
    filtered_samples: List[Dict[str, Any]] = []

    for sample in tqdm(samples, desc=f"Filter {mode} data"):
        try:
            shape_transform, color_transform, indicator_transform = (
                extract_transformations(sample)
            )
        except KeyError as e:
            raise e

        shape_type = shape_transform.get("type")
        color_type = color_transform.get("type")
        indicator_type = indicator_transform.get("type")
        if shape_type is None or color_type is None or indicator_type is None:
            raise KeyError(f"Missing transformation type in sample: {sample}")

        # Check if the transformation triplet is valid.
        if not is_valid_triplet(
            shape_type, color_type, indicator_type, valid_triplets_set
        ):
            continue

        # Check if the transformation kwargs are valid.
        func_names_in_sample = [shape_type, color_type, indicator_type]
        transforms = [shape_transform, color_transform, indicator_transform]
        if not are_valid_function_kwargs(
            func_names_in_sample, transforms, allowable_function_kwargs
        ):
            continue

        # Update counter (or check in test mode) and ensure valid primitive functions.
        if not update_and_check_func_kwargs_counter(
            mode,
            func_kwargs_counter,
            func_names_in_sample,
            transforms,
            min_func_examples,
        ):
            continue

        # Further filter the sample based on the required number of examples.
        valid_sample = filter_num_examples(
            valid_sample=sample,
            num_primitive_functions=num_primitive_functions,
            num_func_compositions=num_func_compositions,
            num_query_functions=num_query_functions,
            num_query_few_shots=num_query_few_shots,
            shuffle=shuffle_study_examples,
        )
        filtered_samples.append(valid_sample)

    if mode == "train":
        return filtered_samples, func_kwargs_counter
    elif mode == "test":
        return filtered_samples
    else:
        raise ValueError(f"Invalid mode: {mode}")


def train_val_test_split(
    train_samples: List[Dict],
    test_samples: List[Dict],
    num_train_samples: int = 0,
    num_test_samples: int = 0,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Splits the provided samples into training, validation, and testing sets.

    Args:
        train_samples (List[Dict]): A list of dictionaries, each representing a sample for training.
        test_samples (List[Dict]): A list of dictionaries, each representing a sample for testing.
        num_train_samples (int, optional): The number of training samples to use. Defaults to 0.
        num_test_samples (int, optional): The number of test samples to use. Defaults to 0.

    Returns:
        Tuple[List[Dict], List[Dict], List[Dict]]: A tuple containing the training, validation, and testing sets.
    """
    if num_train_samples > 0 and num_train_samples < len(train_samples):
        train_samples = train_samples[:num_train_samples]
    logging.info(f"{len(train_samples)} train episodes used.")

    if num_test_samples > 0 and num_test_samples < len(test_samples):
        test_samples = test_samples[:num_test_samples]
    logging.info(f"{len(test_samples)} test episodes used.")

    test_size = len(test_samples) // 2
    val_samples = test_samples[:test_size]
    test_samples = test_samples[test_size:]

    return train_samples, val_samples, test_samples


def main() -> None:
    """
    Main function to orchestrate the execution flow.
    """
    args = parse_arguments()

    setup_logging(args.verbose)
    set_seed(args.seed)

    # get data config file
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    config_file_path = os.path.join(
        os.path.dirname(current_dir), "vmlc", "configs", "systematicity_data.yaml"
    )
    data_config = load_yaml_file(config_file_path)

    # configure split
    train_function_kwargs = data_config["train"]
    test_function_kwargs = data_config["test"]
    assert list(train_function_kwargs.keys()) == list(
        test_function_kwargs.keys()
    ), "For systematicity, we want to have the same primitive functions for training and testing!"

    train_func_compositions, test_func_compositions = split_triplet_compositions(
        list(train_function_kwargs.keys()),
        frac_test_compositions=args.frac_test_compositions,
    )

    # get data
    assert args.num_samples >= 0, f"Invalid number of samples: {args.num_samples}"

    episodes = load_jsonl(file_path=args.data_path)
    num_episodes = len(episodes)
    logging.info(f"{num_episodes} episodes loaded.")

    if args.num_samples == 0 or args.num_samples > num_episodes:
        args.num_samples = num_episodes
    episodes = episodes[: args.num_samples]
    logging.info(f"Use first {len(episodes)} episodes...")

    # filter train and test sets
    all_train_samples, func_kwargs_counter = filter_samples_by_composition(
        samples=episodes,
        valid_triplets=train_func_compositions,
        num_primitive_functions=args.num_primitives,
        num_func_compositions=args.num_compositions,
        num_query_functions=args.num_queries,
        allowable_function_kwargs=train_function_kwargs,
        mode="train",
        num_query_few_shots=args.num_query_few_shots,
        shuffle_study_examples=args.shuffle_study_examples,
    )
    logging.info(f"{len(all_train_samples)} train episodes found.")

    all_test_samples = filter_samples_by_composition(
        samples=episodes,
        valid_triplets=test_func_compositions,
        num_primitive_functions=args.num_primitives,
        num_func_compositions=args.num_compositions,
        num_query_functions=args.num_queries,
        allowable_function_kwargs=test_function_kwargs,
        mode="test",
        num_query_few_shots=args.num_query_few_shots,
        shuffle_study_examples=args.shuffle_study_examples,
        min_func_examples=args.min_func_examples,
        func_kwargs_counter=func_kwargs_counter,
    )
    logging.info(f"{len(all_test_samples)} test episodes found.")

    train_samples, val_samples, test_samples = train_val_test_split(
        train_samples=all_train_samples,
        test_samples=all_test_samples,
        num_train_samples=args.num_train_samples,
        num_test_samples=args.num_test_samples,
    )

    # save metadata
    data_dir = os.path.dirname(os.path.dirname(args.data_path))
    split_dir = os.path.join(data_dir, f"split_seed_{args.seed}")
    save_dict_to_json(
        data={
            "train_func_compositions": train_func_compositions,
            "test_func_compositions": test_func_compositions,
            "func_freq": func_kwargs_counter,
            "train_func_kwargs": train_function_kwargs,
            "test_func_kwargs": test_function_kwargs,
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "num_test_samples": len(test_samples),
            "args": vars(args),
        },
        file_path=os.path.join(
            split_dir, f"config_systematicity_seed_{args.seed}.json"
        ),
    )

    # save split
    save_dicts_as_jsonl(
        data=train_samples,
        filepath=os.path.join(split_dir, f"train_systematicity_seed_{args.seed}.jsonl"),
    )
    save_dicts_as_jsonl(
        data=val_samples,
        filepath=os.path.join(split_dir, f"val_systematicity_seed_{args.seed}.jsonl"),
    )
    save_dicts_as_jsonl(
        data=test_samples,
        filepath=os.path.join(split_dir, f"test_systematicity_seed_{args.seed}.jsonl"),
    )


if __name__ == "__main__":
    main()
