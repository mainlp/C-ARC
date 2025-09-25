"""
Utility functions for our visual grammar.
"""

import random
from itertools import combinations
from typing import Callable, Dict, List, Tuple

import numpy as np


def get_coordinate_range(
    coordinates: List[Tuple[int, int]]
) -> Tuple[int, int, int, int]:
    """
    Get the range of coordinates in the list.

    Parameters:
        coordinates (list of tuples): List of (x, y) coordinates.

    Returns:
        tuple: (min_x, max_x, min_y, max_y)
    """

    x_coords, y_coords = zip(*coordinates)
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    return min_x, max_x, min_y, max_y


def shape2coordinates(shape: str) -> List[Tuple[int, int]]:
    """
    Converts a binary string into a list of tuples, each containing two consecutive digits.

    Parameters:
        binary_string (str): A string of binary digits (e.g., "011011").

    Returns:
        list: A list of tuples where each tuple contains two consecutive digits.
    """
    # Initialize an empty list to store the tuples
    tuple_list = []

    # Iterate over the string, taking two characters at a time
    for i in range(0, len(shape), 2):
        # Create a tuple from two consecutive characters
        tuple_pair = (int(shape[i]), int(shape[i + 1]))
        # Append the tuple to the list
        tuple_list.append(tuple_pair)

    return tuple_list


def coordinates2shape(coordinates: List[Tuple[int, int]]) -> str:
    """
    Normalize and sort the coordinates.

    Parameters:
        coordinates (list of tuples): List of (x, y) coordinates.

    Returns:
        str: A string representation of the normalized and sorted coordinates.
    """
    min_x, _, min_y, _ = get_coordinate_range(coordinates)

    # Normalize coordinates by subtracting min_x and min_y
    normalized_coordinates = [(x - min_x, y - min_y) for x, y in coordinates]

    # Sort coordinates by x first and then by y
    sorted_coordinates = sorted(
        normalized_coordinates, key=lambda coord: (coord[0], coord[1])
    )

    # Convert sorted coordinates to the desired string format
    coordinates_string = "".join(f"{x}{y}" for x, y in sorted_coordinates)

    return coordinates_string


def find_groups(x: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    Finds connected components in the input matrix.

    Parameters:
    x (np.ndarray): A 2D NumPy array where connected components are to be identified.

    Returns:
    groups list[list[tuple[int, int]]]: A list of groups, where each group is a list of tuples representing the coordinates of connected cells.
    """

    def dfs(row, col, group):
        if (
            row < 0
            or col < 0
            or row >= rows
            or col >= cols
            or x[row, col] == 0
            or (row, col) in visited
        ):
            return
        visited.add((row, col))
        group.append((row, col))
        dfs(row - 1, col - 1, group)
        dfs(row - 1, col, group)
        dfs(row - 1, col + 1, group)
        dfs(row, col - 1, group)
        dfs(row, col + 1, group)
        dfs(row + 1, col - 1, group)
        dfs(row + 1, col, group)
        dfs(row + 1, col + 1, group)

    rows, cols = x.shape
    positions = set((i, j) for i in range(rows) for j in range(cols))
    groups: List[List[Tuple[int, int]]] = []
    visited: set[Tuple[int, int]] = set()

    for position in positions:
        if x[position] != -1 and position not in visited:
            group: List[Tuple[int, int]] = []
            dfs(position[0], position[1], group)
            if group:
                groups.append(group)

    return groups


def shape2size(shape: str) -> int:
    """
    Calculates the size of a shape based on its coordinates.

    Args:
        shape (str): A string representation of the shape's coordinates.

    Returns:
        int: The number of coordinates in the shape.
    """
    coordinates = shape2coordinates(shape)
    return len(coordinates)


def random_translation_kwargs() -> Dict:
    """
    Generates a random translation direction and step.

    Returns:
        Dict: A dictionary containing the direction ('x' or 'y') and the step (an integer from -5 to 5, excluding 0).
    """
    direction = random.choice(["x", "y"])
    step_size = random.choice(list(range(-5, 0)) + list(range(1, 6)))
    return {"direction": direction, "step_size": step_size}


def random_mirror_kwargs() -> Dict:
    """
    Generates a random mirror direction.

    Returns:
        Dict: A dictionary containing the randomly chosen mirror direction ('x' or 'y').
    """
    mirror_direction = random.choice(["x", "y"])
    return {"direction": mirror_direction}


def random_rotation_kwargs() -> Dict:
    """
    Generates a random rotation type.

    Returns:
        Dict: A dictionary containing the randomly chosen rotation type.
    """
    return {"rotation_type": random.choice([0, 1, 2, 3, 4, 5, 6, 7])}


def random_set_color_kwargs() -> Dict:
    """
    Generates a random set color keyword argument.

    Returns:
        Dict: A dictionary containing a randomly chosen new color.
    """
    return {"new_color": random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])}


def random_reshape_kwargs() -> Dict:
    """
    Generates a random reshape keyword argument.

    Returns:
        Dict: A dictionary containing a randomly chosen shape type.
    """
    return {"shape_type": random.choice(["cross", "x", "square", "flower"])}


def random_grow_kwargs() -> Dict:
    """
    Generates a random grow keyword argument.

    Returns:
        Dict: An empty dictionary representing the grow keyword argument.
    """
    return {"direction": random.choice(["-x", "+x", "-y", "+y"])}


def random_remove_row_column_kwargs() -> Dict:
    """
    Generates a random grow keyword argument.

    Returns:
        Dict: An empty dictionary representing the grow keyword argument.
    """
    return {"direction": random.choice(["-x", "+x", "-y", "+y"])}


def random_sample_kwargs(kwarg_options: Dict) -> Dict:
    """
    Randomly samples keyword arguments from a dictionary of possible values.

    Args:
        kwarg_options (Dict): A dictionary where the keys are the names of the keyword arguments and the values are lists of possible values.

    Returns:
        Dict: A dictionary containing a randomly chosen value for each keyword argument.
    """
    final_kwargs: Dict = {}
    for kwarg_name, kwarg_values in kwarg_options.items():
        final_kwargs[kwarg_name] = random.choice(kwarg_values)

    return final_kwargs


def train_test_transformation_compositions(
    callables: List[Callable],
) -> Tuple[List[Tuple[Callable, Callable]], List[Tuple[Callable, Callable]]]:
    """
    Randomly splits pairs of callables into two lists such that each callable appears exactly 2 times in a list.

    Args:
        callables (List[Callable]): List of callable functions.

    Returns:
        Tuple[List[Tuple[Callable, Callable]], List[Tuple[Callable, Callable]]]: Two lists of callable pairs.
    """
    # Generate all unique pairs of callables
    all_pairs = list(combinations(callables, 2))

    # Shuffle the pairs randomly
    random.shuffle(all_pairs)

    # Initialize counts for each callable
    count_in_list1 = {c: 0 for c in callables}

    # Initialize the two lists for storing the pairs
    test_callables: List[Tuple[Callable, Callable]] = []
    train_callables: List[Tuple[Callable, Callable]] = []

    for pair in all_pairs:
        c1, c2 = pair

        # Try to add the pair to List 1 if both callables appear fewer than 3 times
        if count_in_list1[c1] < 2 and count_in_list1[c2] < 2:
            test_callables.append(pair)
            count_in_list1[c1] += 1
            count_in_list1[c2] += 1
        else:
            train_callables.append(pair)

    # Check that the split satisfies the constraints
    assert all(
        count == 2 for count in count_in_list1.values()
    ), f"Some callable appears fewer than 2 times in test callables:\n{test_callables}"

    return train_callables, test_callables
