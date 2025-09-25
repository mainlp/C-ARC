"""
Primitive transformations (color-based, shape-based transforms, etc.).
"""

import copy
import random
from typing import Callable, List, Tuple

from vmlc.utils import shape2size
from vmlc.visual_grammar import Group, coordinate_overlap, generate_group


def color_shape_based_input_outputs(
    allowed_colors: List[int],
    shape: str,
    transformation: Callable[..., Group | None],
    transformation_kwargs: dict,
    forbidden_shape: str = "",
    num_pairs: int = 3,
    num_groups: int = 1,
    grid_size: Tuple[int, int] = (10, 10),
) -> List[Tuple[List[Group], List[Group]]]:
    """
    Generates a specified number of input-output pairs of study examples based on
    the given shape, transformation, and grid size.

    Parameters:
        allowed_colors (List[int]): A list of allowed colors for the input groups.
        shape (str): The shape of the input groups.
        transformation (Callable[..., Group]): A transformation function to apply to
            the input groups.
        transformation_kwargs (dict): Keyword arguments for the transformation function.
        forbidden_shape (str): A shape that the input groups should not have. Defaults to "".
        num_pairs (int): The number of input-output pairs to generate. Defaults to 3.
        grid_size (Tuple[int, int]): The size of the grid. Defaults to (10, 10).

    Returns:
        List[Tuple[List[Group], List[Group]]]: A list of input-output pairs of study examples.
    """
    input_outputs: List[Tuple[List[Group], List[Group]]] = []

    while len(input_outputs) < num_pairs:
        list_in_groups: List[Group] = []
        list_out_groups: List[Group] = []

        while len(list_out_groups) < num_groups:
            color = random.choice(allowed_colors)
            if shape == "":
                size = random.randint(1, grid_size[0] - 1)
            else:
                size = shape2size(shape)

            in_group = generate_group(
                color=color, size=size, shape=shape, grid_size=grid_size
            )
            if in_group.shape == forbidden_shape or coordinate_overlap(
                list_in_groups + [in_group]
            ):
                continue

            out_group = transformation(group=in_group, **transformation_kwargs)
            if out_group is not None and not coordinate_overlap(
                list_out_groups + [out_group]
            ):
                list_in_groups.append(in_group)
                list_out_groups.append(out_group)

        input_outputs.append((list_in_groups, list_out_groups))

    return input_outputs


def gen_input_outputs(
    allowed_colors: List[int],
    shape: str,
    transformation: Callable[..., Group | None],
    transformation_kwargs: dict,
    indicators: List[Group],
    forbidden_shapes: List[str] = [""],
    num_pairs: int = 3,
    num_groups: int = 1,
    grid_size: Tuple[int, int] = (10, 10),
    max_tries: int = 500,
) -> List[Tuple[List[Group], List[Group]]] | None:
    """
    Generates a specified number of input-output pairs of study examples based on
    the given shape, transformation, and grid size.

    Parameters:
        allowed_colors (List[int]): A list of allowed colors for the input groups.
        shape (str): The shape of the input groups.
        transformation (Callable[..., Group]): A transformation function to apply to
            the input groups.
        transformation_kwargs (dict): Keyword arguments for the transformation function.
        forbidden_shape (str): A shape that the input groups should not have. Defaults to "".
        num_pairs (int): The number of input-output pairs to generate. Defaults to 3.
        grid_size (Tuple[int, int]): The size of the grid. Defaults to (10, 10).
        max_tries (int): The maximum number of tries to generate the input-output pairs.
            Defaults to 50.

    Returns:
        List[Tuple[List[Group], List[Group]]]: A list of input-output pairs of study examples.
    """
    tries = 0
    num_indicators = len(indicators)
    input_outputs: List[Tuple[List[Group], List[Group]]] = []

    while len(input_outputs) < num_pairs:
        list_in_groups: List[Group] = copy.copy(indicators)
        list_out_groups: List[Group] = copy.copy(indicators)

        while len(list_out_groups) < num_groups + num_indicators:
            tries += 1
            if tries > max_tries:
                return None

            color = random.choice(allowed_colors)
            if shape == "":
                size = random.randint(1, grid_size[0] - 1)
            else:
                size = shape2size(shape)

            in_group = generate_group(
                color=color, size=size, shape=shape, grid_size=grid_size
            )
            if in_group.shape in forbidden_shapes or coordinate_overlap(
                list_in_groups + [in_group]
            ):
                continue

            out_group = transformation(group=in_group, **transformation_kwargs)
            if out_group is not None and not coordinate_overlap(
                list_out_groups + [out_group]
            ):
                list_in_groups.append(in_group)
                list_out_groups.append(out_group)

        input_outputs.append((list_in_groups, list_out_groups))

    return input_outputs
