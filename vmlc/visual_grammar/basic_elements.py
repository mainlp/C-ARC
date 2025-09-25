"""
Basic elements of the visual grammar.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from vmlc.utils import coordinates2shape, get_coordinate_range, shape2coordinates


@dataclass
class Group:
    """
    A group of connected coordinates with the same color in a given grid.
    """

    coordinates: List[Tuple[int, int]] = field(default_factory=list)
    color: int = 0

    def __eq__(self, other):
        return (
            isinstance(other, Group)
            and self.coordinates == other.coordinates
            and self.color == other.color
        )

    def __hash__(self):
        return hash((tuple(self.coordinates), self.color))

    @property
    def shape(self) -> str:
        if not self.coordinates:
            return ""

        return coordinates2shape(self.coordinates)

    @property
    def height(self) -> int:
        if not self.coordinates:
            return 0

        min_x, max_x, _, _ = get_coordinate_range(self.coordinates)
        return max_x - min_x + 1

    @property
    def width(self) -> int:
        if not self.coordinates:
            return 0

        _, _, min_y, max_y = get_coordinate_range(self.coordinates)
        return max_y - min_y + 1


def coordinate_overlap(
    group_list: List[Group],
    grid_size: Tuple[int, int] = (10, 10),
) -> bool:
    """
    Checks for overlapping coordinates among a list of groups within a given grid size.

    Args:
        group_list (List[Group]): A list of groups to check for overlapping coordinates.
        grid_size (Tuple[int, int]): The size of the grid. Defaults to (10, 10).

    Returns:
        bool: True if any overlapping coordinates are found, False otherwise.
    """
    overall_set: set[Tuple[int, int]] = set()
    for group in group_list:
        intermediate_set: set[Tuple[int, int]] = set()
        for coord in group.coordinates:
            x, y = coord
            if coord in overall_set:
                return True

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    new_x, new_y = x + dx, y + dy
                    new_coord = (new_x, new_y)
                    if 0 <= new_x < grid_size[0] and 0 <= new_y < grid_size[1]:
                        if new_coord not in overall_set:
                            intermediate_set.add(new_coord)
        overall_set.update(intermediate_set)
    return False


def init_grid(grid_size: Tuple[int, int]) -> np.ndarray:
    """
    Initializes a grid of zeros with the given grid size.

    Args:
        grid_size (Tuple[int, int]): The size of the grid.

    Returns:
        numpy.ndarray: A grid of zeros with shape grid_size and dtype int.
    """
    return np.zeros(grid_size, dtype=int)


def set_groups_in_grid(grid: np.ndarray, groups: List[Group]) -> np.ndarray:
    """
    Sets a group of coordinates with a specified color in a given grid.

    Args:
        grid (np.ndarray): A 2D NumPy array representing the grid.
        group (List[Group]): Groups of coordinates with a specified color.

    Returns:
        np.ndarray: The modified grid with the group of coordinates set to the specified color.
    """
    copied = grid.copy()
    for group in groups:
        for coordinate in group.coordinates:
            x_coordinate = coordinate[0]
            y_coordinate = coordinate[1]

            assert (
                x_coordinate < grid.shape[0]
            ), f"x_coordinate {x_coordinate} exceeds grid zero-dimension of size: {grid.shape[0]}"
            assert (
                y_coordinate < grid.shape[1]
            ), f"x_coordinate {y_coordinate} exceeds grid one-dimension of size: {grid.shape[1]}"

            copied[x_coordinate, y_coordinate] = group.color

    return copied


def set_groups_in_input_output_grids(
    input_output_groups: List[Tuple[List[Group], List[Group]]],
    grid_size: Tuple[int, int] = (10, 10),
    to_list: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Fills grids with input and output groups.

    Args:
        input_output_groups (List[Tuple[List[Group], List[Group]]]): A list of tuples containing input and output groups.
        grid_size (Tuple[int, int], optional): The size of the grid. Defaults to (10, 10).

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: A list of tuples containing input and output grids.
    """
    grid = init_grid(grid_size)
    grid_list: List[Tuple[np.ndarray, np.ndarray]] = []

    for input_groups, output_groups in input_output_groups:
        input_grid = set_groups_in_grid(grid, input_groups)
        output_grid = set_groups_in_grid(grid, output_groups)

        if to_list:
            input_grid = input_grid.tolist()
            output_grid = output_grid.tolist()
        grid_list.append((input_grid, output_grid))

    return grid_list


def find_groups(grid: np.ndarray) -> list[Group]:
    """
    Finds groups of connected coordinates with the same color in a given grid.

    Args:
        grid (np.ndarray): A 2D NumPy array representing the grid, where each element is a color value.

    Returns:
        list[Group]: A list of groups, where each group is a collection of connected coordinates with the same color.
    """

    def extract_group(row: int, col: int, group: Group):
        if (
            row < 0
            or col < 0
            or row >= rows
            or col >= cols
            or grid[row, col] == 0
            or (row, col) in visited
        ):
            return
        visited.add((row, col))
        group.coordinates.append((row, col))
        group.color = grid[row, col]
        extract_group(row - 1, col - 1, group)
        extract_group(row - 1, col, group)
        extract_group(row - 1, col + 1, group)
        extract_group(row, col - 1, group)
        extract_group(row, col + 1, group)
        extract_group(row + 1, col - 1, group)
        extract_group(row + 1, col, group)
        extract_group(row + 1, col + 1, group)

    rows, cols = grid.shape
    positions = set((i, j) for i in range(rows) for j in range(cols))
    groups: List[Group] = []
    visited: set[Tuple[int, int]] = set()

    for position in positions:
        if grid[position] != -1 and position not in visited:
            group: Group = Group()
            extract_group(position[0], position[1], group)
            if group:
                groups.append(group)

    return groups


def generate_group(
    color: int = 0,
    size: int = 0,
    shape: str = "",
    grid_size: Tuple[int, int] = (10, 10),
    x_step: int = 0,
    y_step: int = 0,
) -> Group:
    """
    Generates a group with a block of specified size and color.

    Parameters:
    color (int): The color to fill the block (1-9). If 0 or out of range, a random color is chosen.
    size (int): The number of cells in the block (1-9). If 0 or out of range, a random size is chosen.
    shape (str): The shape of the block. If an empty string, a random shape is generated.
    grid_size (Tuple[int, int]): The size of the grid (grid_size x grid_size). Defaults to (10, 10).
    x_step (int): The x-step for the block. Defaults to 0.
    y_step (int): The y-step for the block. Defaults to 0.

    Returns:
    Group: The generated group if successful, or None if the block cannot be created.
    """
    # Ensure color and size are within the valid range (1-9)
    if color < 1 or color > 9:
        logging.warning(
            "Color is outside specified integer range 1-9! Select color randomly from [1, 9]."
        )
        color = random.randint(1, 9)
    if size < 1 or size > 9:
        logging.warning(
            "Size is outside specified integer range 1-9! Select size randomly from [1, 9]."
        )
        size = random.randint(1, 9)

    grid_height = grid_size[0]
    grid_width = grid_size[1]

    if shape == "":
        # Randomly generate group
        group = Group(color=color)

        # Randomly choose a starting point for the block
        start_x, start_y = random.randint(0, grid_height - 1), random.randint(
            0, grid_width - 1
        )
        current = (start_x, start_y)
        group.coordinates.append(current)

        moves = 0  # Track the number of moves made

        # Continue to expand the block until it reaches the desired size or max moves are reached
        while len(group.coordinates) < size and moves < 4 * size:
            neighbors: List[Tuple[int, int]] = []

            # Find all valid neighboring cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = current[0] + dx, current[1] + dy
                    if 0 <= nx < grid_height and 0 <= ny < grid_width:
                        neighbors.append((nx, ny))

            # If there are no valid neighbors, break the loop
            if not neighbors:
                break

            # Randomly choose a neighboring cell
            next_current = random.choice(neighbors)

            # If the chosen neighbor is not already in the block, add it
            if next_current not in group.coordinates:
                current = next_current
                group.coordinates.append(current)
            else:
                current = next_current

            moves += 1  # Increment the move counter

        # Return the grid if the block has the desired size, otherwise return None
        if len(group.coordinates) == size:
            return group
        else:
            generate_group(color=color, size=size)
    else:
        coordinates = shape2coordinates(shape)
        group = Group(color=color, coordinates=coordinates)

        start_x, start_y = random.randint(
            max(0, 0 - x_step),
            min(grid_height - group.height, grid_height - group.height - x_step),
        ), random.randint(
            max(0, 0 - y_step),
            min(grid_width - group.width, grid_width - group.width - y_step),
        )

        # shift to starting position
        for i, (x, y) in enumerate(group.coordinates):
            group.coordinates[i] = (start_x + x, start_y + y)

    return group
