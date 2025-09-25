"""
Utility functions.
"""

import json
import logging
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

# Get the existing logger
logger = logging.getLogger(__name__)


def setup_logging(verbosity: int):
    """
    Set up logging based on the verbosity level.

    Args:
    verbosity (int): Verbosity level from command line arguments.
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def set_seed(seed: int = 0) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generation. Defaults to 0.

    Returns:
        None
    """
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_dict_to_json(data: Dict, file_path: str) -> None:
    """
    Saves a given dictionary to a JSON file.

    Parameters:
    - data (dict): The dictionary to be saved.
    - file_path (str): The path (including file name) where the JSON file will be saved.

    Raises:
    - Exception: Propagates any exceptions that occur during file creation or JSON serialization.

    This function will create the directory path if it does not exist.
    It handles exceptions related to file writing and JSON serialization by raising them.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Failed to save dictionary to JSON: {file_path}. Error: {e}")
        raise e


def save_dicts_as_jsonl(data: List[Dict], filepath: str) -> None:
    """
    Saves a list of dictionaries as a JSON Lines file.

    Args:
        data (List[Dict]): The list of dictionaries to be saved.
        filepath (str): The path to the JSON Lines file, including the filename.
                        The folder structure is created if it doesn't exist.

    Raises:
        ValueError: If the data is not a list or contains non-dictionary elements.
        IOError: If there are issues writing to the file.
    """
    # Validate input data
    if not isinstance(data, List) or not all(isinstance(item, Dict) for item in data):
        raise ValueError("Data must be a list of dictionaries.")

    # Ensure the directory exists
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Write data to .jsonl file
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            for item in data:
                json_record = json.dumps(item, ensure_ascii=False)
                file.write(json_record + "\n")
    except IOError as e:
        raise IOError(f"Failed to write to file {filepath}: {e}")


def load_json(file_path: str) -> dict:
    """
    Load a JSON file and return a dictionary or list.

    Args:
        file_path (str): The path to the JSON file to be loaded.

    Returns:
        dict or list: The JSON object loaded from the file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {file_path}")
        raise


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON Lines (JSONL) file and return a list of dictionaries.

    This function includes error handling to log messages if the file doesn't exist or if a line contains invalid JSON.

    Args:
        file_path (str): The path to the JSONL file to be loaded.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
        represents a JSON object from a single line of the JSONL file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If a line in the file is not valid JSON.
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    json_line = json.loads(line.strip())
                    data.append(json_line)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON at line {line_number} in {file_path}")
                    raise
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

    return data


def load_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Reads a nested YAML file and returns its contents as a nested dictionary.

    This function attempts to open and parse the YAML file specified by `file_path`.
    If the file is empty or only contains comments, an empty dictionary is returned.
    The function ensures that the YAML content is a dictionary; otherwise, a ValueError
    is raised.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        Dict[str, Any]: A nested dictionary representing the YAML file's data.

    Raises:
        FileNotFoundError: If the file at `file_path` does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
        ValueError: If the parsed YAML content is not a dictionary.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            if data is None:
                # Empty file or file with only comments; return an empty dictionary
                return {}
            if not isinstance(data, dict):
                raise ValueError(
                    "The YAML file does not contain a top-level dictionary."
                )
            return data
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(f"File not found: {file_path}") from fnf_error
    except yaml.YAMLError as yaml_error:
        raise yaml.YAMLError(f"Error parsing YAML file: {file_path}") from yaml_error


def extract_input_output_grids(
    grid_list: List[List[np.ndarray]],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extracts the input and output grids from a list of lists of grids.

    Args:
        grid_list (List[List[np.ndarray]]): A list of lists of grids, where each inner
            list contains two grids: the input grid and the output grid.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: A tuple containing two lists: the input
            grids and the output grids.
    """
    input_grids: List[np.ndarray] = []
    output_grids: List[np.ndarray] = []

    for grids in grid_list:
        assert len(grids) == 2, f"Invalid number of grids: {len(grids)}"
        input_grids.append(np.array(grids[0]))
        output_grids.append(np.array(grids[1]))

    return input_grids, output_grids


def parse_data(episode: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
    """
    Parses an episode to extract input and output grids for study examples and queries.

    Args:
        episode (Dict[str, Any]): A dictionary containing 'study_examples' and 'queries',
            each of which is a list of lists of grids.

    Returns:
        Dict[str, List[np.ndarray]]: A dictionary with keys 'xs', 'ys', 'xq', 'yq',
            representing lists of input and output grids extracted from study examples
            and queries, respectively.
    """
    study_examples = episode["study_examples"]
    queries = episode["queries"]

    xs, ys = extract_input_output_grids(study_examples)
    xq, yq = extract_input_output_grids(queries)

    return {
        "xs": xs,
        "ys": ys,
        "xq": xq,
        "yq": yq,
    }


def extract(include: Union[List[bool], np.ndarray], arr: List[Any]) -> List[Any]:
    """
    Create a new list using only the elements of `arr` that correspond to True in `include`.

    Args:
        include (Union[List[bool], np.ndarray]): A boolean array indicating which elements to include.
        arr (List[Any]): The array from which elements are extracted.

    Returns:
        List[Any]: A new list containing the extracted elements.
    """
    assert len(include) == len(arr)
    return [a for idx, a in enumerate(arr) if include[idx]]


def pattern_to_matrix(pattern: List[int]) -> np.ndarray:
    """
    Convert a list of integers into a 10x10 matrix, with specific constraints on the values.

    Args:
        pattern (List[int]): A list of integers.

    Returns:
        np.ndarray: A 10x10 numpy array.
    """
    try:
        pattern = [int(x) if x is not None else -1 for x in pattern]
    except ValueError:
        pattern = [0] * 100

    if len(pattern) < 100:
        pattern = [0] * 100
    elif len(pattern) > 100:
        pattern = pattern[:100]

    matrix = np.array(pattern).reshape((10, 10))

    return matrix


def flatten_matrices(
    list_of_lists: List[List[List[Optional[int]]]],
) -> List[List[Optional[int]]]:
    """
    Flattens a list of lists of lists.

    Args:
        list_of_lists ( List[List[List[Optional[int]]]]):
            A list containing lists of lists to be flattened.

    Returns:
        List[List[Optional[int]]]: A flattened list of lists where each element is an integer.
    """
    return [
        [item for inner_list in sublist for item in inner_list]
        for sublist in list_of_lists
    ]


def flip(p_head: float = 0.5) -> bool:
    """
    Return True with probability p_head.

    Args:
        p_head (float, optional): Probability of returning True. Defaults to 0.5.

    Returns:
        bool: True with probability p_head, else False.
    """
    return random.random() < p_head


def add_response_noise(
    yq_item: np.ndarray, p_noise: float, grid_size: int = 10
) -> np.ndarray:
    """
    Add noise to the output sequence by randomly replacing symbols.

    Args:
        yq_item (np.ndarray): List of output symbols for the response to a single command (excluding EOS).
        p_noise (float): Probability of lapse (uniform draw) for a particular emission.
        langs (Lang): Language object for translating symbols.

    Returns:
        np.ndarray: Output sequence with noise added.
    """
    assert len(yq_item.shape) == 2, f"invalid shape of grid! {yq_item.shape}"
    yq_item = deepcopy(yq_item)
    for row in range(grid_size):
        for col in range(grid_size):
            if flip(p_noise):
                new_value = random.randint(0, 9)
                yq_item[row, col] = new_value
    return yq_item
