"""
Basic transformations (translation, rotation, etc.).
"""

import copy
from typing import Literal, Tuple

from vmlc.utils import get_coordinate_range
from vmlc.visual_grammar import Group


def translation(
    group: Group,
    step_size: int,
    direction: Literal["x", "y"],
    grid_size: Tuple[int, int] = (10, 10),
) -> Group | None:
    """
    Translates a group of coordinates in a grid by a specified step size in a given direction.

    Parameters:
        group (Group): The group of coordinates to be translated.
        step_size (int): The number of steps to translate the group.
        direction (Literal["x", "y"]): The direction of translation, either "x" or "y".
        grid_size (Tuple[int, int], optional): The size of the grid. Defaults to (10, 10).

    Returns:
        Group | None: The translated group if successful, or None if the translation exceeds the grid boundaries.
    """
    grid_height = grid_size[0]
    grid_width = grid_size[1]

    new_group = Group(color=group.color)

    for row, col in group.coordinates:
        if direction == "x":
            if (
                col + step_size >= grid_width or col + step_size < 0
            ):  # exceeds array bounds
                return None
            else:
                new_group.coordinates.append((row, col + step_size))
        elif direction == "y":
            if (
                row + step_size >= grid_height or row + step_size < 0
            ):  # exceeds array bounds
                return None
            else:
                new_group.coordinates.append((row + step_size, col))

    return new_group


def mirror(
    group: Group,
    direction: Literal["x", "y"],
    grid_size: Tuple[int, int] = (10, 10),
) -> Group | None:
    """
    Mirrors a group of coordinates in a specified direction.

    Parameters:
        group (Group): The group of coordinates to be mirrored.
        direction (Literal["x", "y"]): The direction of mirroring, either "x" or "y".

    Returns:
        Group: The mirrored group of coordinates.
    """
    grid_height = grid_size[0]
    grid_width = grid_size[1]
    min_x, max_x, min_y, max_y = get_coordinate_range(group.coordinates)
    new_group = Group(color=group.color)

    if direction == "y":
        for row, col in group.coordinates:
            new_col = max_y - (col - min_y)
            if new_col >= grid_width or new_col < 0:  # exceeds array bounds
                return None
            else:
                new_group.coordinates.append((row, new_col))
    if direction == "x":
        for row, col in group.coordinates:
            new_row = max_x - (row - min_x)
            if new_row >= grid_height or new_row < 0:  # exceeds array bounds
                return None
            else:
                new_group.coordinates.append((new_row, col))

    # check for symmetric objects
    if set(group.coordinates) == set(new_group.coordinates):
        return None

    return new_group


def rotation(
    group: Group,
    rotation_type: Literal[0, 1, 2, 3, 4, 5, 6, 7] = 0,
    grid_size: Tuple[int, int] = (10, 10),
) -> Group | None:
    """
    Rotates a block within a grid based on the specified rotation type.

    Parameters:
        group (Group): The group of coordinates to be rotated.
        rotation_type (Literal[0, 1, 2, 3, 4, 5, 6, 7]): The type of rotation to apply.
        grid_size (Tuple[int, int]): The size of the grid.

    Returns:
        Group | None: The rotated group of coordinates, or None if the rotation is out of bounds.
    """
    grid_height = grid_size[0]
    grid_width = grid_size[1]
    min_x, max_x, min_y, max_y = get_coordinate_range(group.coordinates)

    new_group = Group(color=group.color)

    for row, col in group.coordinates:
        # Rotate around the top-left corner of the bounding box (anticlockwise)
        if rotation_type == 1:
            new_row = min_x - (col - min_y)
            new_col = min_y + (row - min_x)

        # Rotate around the top-right corner of the bounding box (clockwise)
        elif rotation_type == 2:
            new_row = min_x - (max_y - col)
            new_col = max_y - (row - min_x)

        # Rotate around the top-right corner of the bounding box (anticlockwise)
        elif rotation_type == 3:
            new_row = min_x + max_y - col
            new_col = max_y + row - min_x

        # Rotate around the bottom-left corner of the bounding box (clockwise)
        elif rotation_type == 4:
            new_row = max_x + col - min_y
            new_col = min_y + max_x - row

        # Rotate around the bottom-left corner of the bounding box (anticlockwise)
        elif rotation_type == 5:
            new_row = max_x - (col - min_y)
            new_col = min_y - (max_x - row)

        # Rotate around the bottom-right corner of the bounding box (clockwise)
        elif rotation_type == 6:
            new_row = max_x + col - max_y
            new_col = max_y - (row - max_x)

        # Rotate around the bottom-right corner of the bounding box (anticlockwise)
        elif rotation_type == 7:
            new_row = max_x - (col - max_y)
            new_col = max_y + (row - max_x)
        else:
            # Rotate around the top-left corner of the bounding box (clockwise)
            new_row = min_x + col - min_y
            new_col = min_y - (row - min_x)

        # Check if the new coordinates are out of bounds
        if 0 <= new_row < grid_height and 0 <= new_col < grid_width:
            new_group.coordinates.append((new_row, new_col))
        else:
            return None

    # check for symmetric objects
    if set(group.coordinates) == set(new_group.coordinates):
        return None

    return new_group


def set_color(
    group: Group, new_color: int = 0, grid_size: Tuple[int, int] = (10, 10)
) -> Group:
    """
    Modifies the provided Group by setting its color to a new color.

    Args:
        group (Group): The group to be modified.
        new_color (int): The group's new color.

    Returns:
        Group: A new Group with the new colors.
    """
    coordinates = copy.copy(group.coordinates)
    new_group = Group(coordinates=coordinates, color=new_color)
    return new_group


def reshape(
    group: Group,
    shape_type: Literal["cross", "x", "square", "flower"] = "x",
    grid_size: Tuple[int, int] = (10, 10),
) -> Group | None:
    """
    Reshapes a given Group into a specified shape.

    Args:
        group (Group): The group to be reshaped.
        shape_type (Literal["cross", "x", "square", "flower"]): The type of shape to reshape into. Defaults to "x".
        grid_size (Tuple[int, int]): The size of the grid. Defaults to (10, 10).

    Returns:
        Group | None: A new Group with the reshaped coordinates, or None if any coordinate is out of bounds.
    """
    # Dictionary of shapes
    shapes = {
        "cross": [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        "x": [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],
        "square": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)],
        "flower": [(0, 1), (1, 0), (1, 2), (2, 1)],
    }
    assert (
        shape_type in shapes
    ), f"Invalid shape_type: {shape_type}! Must be one of the following: 'cross', 'x', 'square', 'flower'."

    # Get the grid dimensions
    grid_height, grid_width = grid_size

    # Get the minimum x and y coordinates from the current group's coordinates
    min_x, _, min_y, _ = get_coordinate_range(group.coordinates)

    # Create a new group with the same color as the original group
    new_group = Group(color=group.color)

    # Get the shape coordinates based on the shape_type
    shape_coords = shapes[shape_type]

    # Transform the shape coordinates and check if they are within bounds
    for i, j in shape_coords:
        new_row = min_x + i
        new_col = min_y + j

        # Check if the new coordinates are within the grid bounds
        if 0 <= new_row < grid_height and 0 <= new_col < grid_width:
            new_group.coordinates.append((new_row, new_col))
        else:
            return None  # If any coordinate is out of bounds, return None

    return new_group


def grow(
    group: Group,
    direction: Literal["-x", "+x", "-y", "+y"],
    grid_size: Tuple[int, int] = (10, 10),
) -> Group | None:
    """
    Grows a given group of coordinates by one unit in all four directions (up, down, left, right).

    Parameters:
        group (Group): The group of coordinates to be grown.
        grid_size (Tuple[int, int]): The size of the grid (height, width) that the group is bound to. Defaults to (10, 10).

    Returns:
        Group | None: A new Group with the grown coordinates or None if any new coordinate is out of bounds.
    """
    considered_coordinates = {
        "-x": [(0, 0), (-1, 0)],
        "+x": [(0, 0), (1, 0)],
        "-y": [(0, 0), (0, -1)],
        "+y": [(0, 0), (0, 1)],
    }
    assert (
        direction in considered_coordinates
    ), f"Invalid direction: {direction}! Must be one of the following: '-x', '+x', '-y', '+y'."

    new_group = Group(color=group.color)
    coordinates = copy.copy(group.coordinates)

    for coord in coordinates:
        x, y = coord
        for dxdy in considered_coordinates[direction]:
            new_x, new_y = x + dxdy[0], y + dxdy[1]
            new_coord = (new_x, new_y)
            if new_coord not in new_group.coordinates:
                if 0 <= new_x < grid_size[0] and 0 <= new_y < grid_size[1]:
                    new_group.coordinates.append(new_coord)
                else:
                    return None

    return new_group


def remove_row_column(
    group: Group,
    direction: Literal["-x", "+x", "-y", "+y"],
    grid_size: Tuple[int, int] = (10, 10),
) -> Group | None:
    """
    Removes a row or column from a given group of coordinates based on the specified direction.

    Parameters:
        group (Group): The group of coordinates from which to remove a row or column.
        direction (Literal["-x", "+x", "-y", "+y"]): The direction in which to remove a row or column.
        grid_size (Tuple[int, int], optional): The size of the grid (height, width) that the group is bound to. Defaults to (10, 10).

    Returns:
        Group | None: A new Group with the removed row or column or None if the group is empty after removal.
    """
    coordinates = copy.copy(group.coordinates)
    new_group = Group(color=group.color)
    if "x" in direction:
        if direction == "-x":
            skip_x_coord = min([coord[0] for coord in coordinates])
        else:
            skip_x_coord = max([coord[0] for coord in coordinates])

        for coord in coordinates:
            if coord[0] != skip_x_coord:
                new_group.coordinates.append(coord)
    else:
        if direction == "-y":
            skip_y_coord = min([coord[1] for coord in coordinates])
        else:
            skip_y_coord = max([coord[1] for coord in coordinates])

        for coord in coordinates:
            if coord[1] != skip_y_coord:
                new_group.coordinates.append(coord)

    if len(new_group.coordinates) == 0:
        return None

    return new_group
