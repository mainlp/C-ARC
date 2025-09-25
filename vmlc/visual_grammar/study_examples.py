"""
Functions to generate (input, output)-pairs of study examples.
"""

import copy
import logging
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

from vmlc.utils import (
    random_grow_kwargs,
    random_mirror_kwargs,
    random_remove_row_column_kwargs,
    random_rotation_kwargs,
    random_sample_kwargs,
    random_set_color_kwargs,
    random_translation_kwargs,
    shape2size,
)
from vmlc.visual_grammar import (
    Group,
    composition,
    gen_indicator_shapes,
    gen_input_outputs,
    generate_group,
    grow,
    mirror,
    remove_row_column,
    rotation,
    set_color,
    set_groups_in_input_output_grids,
    translation,
)

RANDOM_TRANSFORMATION_KWARGS = {
    translation: random_translation_kwargs,
    mirror: random_mirror_kwargs,
    rotation: random_rotation_kwargs,
    set_color: random_set_color_kwargs,
    grow: random_grow_kwargs,
    remove_row_column: random_remove_row_column_kwargs,
}

TRANSFORMATION_NAMES = {
    translation: "translation",
    mirror: "mirror",
    rotation: "rotation",
    set_color: "set_color",
    grow: "grow",
    remove_row_column: "remove",
}


def color_transformation_config(
    colors: List[int],
    kwarg_options: Dict,
) -> Tuple[int, Dict, List[int]]:
    """
    Returns a tuple containing a randomly chosen color from the input list,
    the corresponding transformation keyword arguments, and the updated list
    of colors after removing the chosen color.

    Args:
        colors (List[int]): A list of colors to choose from.
        kwarg_options (Dict): A dictionary of suitable possible function kwargs.

    Returns:
        Tuple[int, Dict, List[int]]: A tuple containing the chosen color, the
            transformation keyword arguments, and the updated list of colors.
    """
    colors = copy.copy(colors)
    color_transformation_color = random.choice(colors)
    colors.remove(color_transformation_color)
    color_transformation_kwargs = random_sample_kwargs(kwarg_options=kwarg_options)

    return color_transformation_color, color_transformation_kwargs, colors


def indicator_transformation_config(
    colors: List[int],
    kwarg_options: Dict,
) -> Tuple[int, str, int, Dict, List[int]]:
    """
    Returns a tuple of configuration parameters for an indicator transformation.

    Args:
        colors (List[int]): A list of colors to choose from.
        mode (Literal["train", "val", "test"]): The mode of the transformation.
        kwarg_options (Dict): A dictionary of suitable possible function kwargs.

    Returns:
        Tuple[int, str, int, Dict, List[int]]: A tuple containing the indicator color,
        the indicator shape, the size, the indicator transformation kwargs, and the remaining colors.
    """
    indicator_color = random.choice(colors)
    colors.remove(indicator_color)

    indicator_shapes = gen_indicator_shapes()
    indicator_shape = random.choice(indicator_shapes)
    size = shape2size(indicator_shape)

    indicator_transformation_kwargs = random_sample_kwargs(kwarg_options=kwarg_options)

    return (
        indicator_color,
        indicator_shape,
        size,
        indicator_transformation_kwargs,
        colors,
    )


def shape_transformation_config(
    colors: List[int],
    color_transformation_kwargs: Dict,
    indicator_shape: str,
    kwarg_options: Dict,
    grid_size: Tuple[int, int],
) -> Tuple[str, Dict]:
    """
    Returns a tuple containing a randomly chosen shape transformation shape and its corresponding transformation keyword arguments.

    Args:
        colors (List[int]): A list of colors to choose from.
        color_transformation_kwargs (Dict): The transformation keyword arguments for the color transformation.
        indicator_shape (str): The shape of the indicator group.
        kwarg_options (Dict): A dictionary of suitable possible function kwargs.
        grid_size (Tuple[int, int]): The size of the grid.

    Returns:
        Tuple[str, Dict]: A tuple containing the shape transformation shape and its transformation keyword arguments.
    """
    shape_transformation_kwargs = random_sample_kwargs(kwarg_options=kwarg_options)

    while shape_transformation_kwargs == color_transformation_kwargs:
        shape_transformation_kwargs = random_sample_kwargs(kwarg_options=kwarg_options)

    shape_transformation_shape = indicator_shape
    while shape_transformation_shape == indicator_shape:
        size = random.randint(1, 9)
        reference_group = generate_group(
            color=random.choice(colors), size=size, shape="", grid_size=grid_size
        )
        shape_transformation_shape = reference_group.shape

    return shape_transformation_shape, shape_transformation_kwargs


def transformation_config(
    hashset: Set[str],
    color_transformation_name: str,
    shape_transformation_name: str,
    indicator_transformation_name: str,
    transformations_kwarg_options: Dict[str, Dict],
    grid_size: Tuple[int, int] = (10, 10),
) -> Tuple[List[int], int, str, Group, dict, dict, dict, dict]:
    """
    Returns a tuple of configuration parameters for transformations.

    Args:
        hashset (Set[str]): A set of existing hashes.
        color_transformation (Callable[..., Group]): A callable that generates a color transformation.
        shape_transformation (Callable[..., Group]): A callable that generates a shape transformation.
        indicator_transformation (Callable[..., Group]): A callable that generates an indicator transformation.
        grid_size (Tuple[int, int], optional): The size of the grid. Defaults to (10, 10).

    Returns:
        Tuple[List[int], int, str, Group, dict, dict, dict, dict]: A tuple containing the remaining colors,
        the translation color, the translation shape, the indicator group, the color transformation kwargs,
        the shape transformation kwargs, the indicator transformation kwargs, and the sample hash.

    Raises:
        ValueError: If the number of tries to find a unique transformation hash exceeds 10000.

    """
    # --- check hash ---
    tries: int = 0
    full_hash: str = ""

    while full_hash == "" or full_hash in hashset:
        initial_colors = list(range(1, 10))

        color_transformation_color, color_transformation_kwargs, colors = (
            color_transformation_config(
                colors=initial_colors,
                kwarg_options=transformations_kwarg_options[color_transformation_name],
            )
        )
        color_hash = (
            str(color_transformation_color)
            + str(color_transformation_name)
            + str(color_transformation_kwargs)
        )

        (
            indicator_color,
            indicator_shape,
            size,
            indicator_transformation_kwargs,
            colors,
        ) = indicator_transformation_config(
            colors=colors,
            kwarg_options=transformations_kwarg_options[indicator_transformation_name],
        )
        indicator_hash = (
            str(indicator_color)
            + indicator_shape
            + str(indicator_transformation_name)
            + str(indicator_transformation_kwargs)
        )

        shape_transformation_shape, shape_transformation_kwargs = (
            shape_transformation_config(
                colors=colors,
                color_transformation_kwargs=color_transformation_kwargs,
                indicator_shape=indicator_shape,
                kwarg_options=transformations_kwarg_options[shape_transformation_name],
                grid_size=grid_size,
            )
        )
        shape_hash = (
            shape_transformation_shape
            + str(shape_transformation_name)
            + str(shape_transformation_kwargs)
        )

        # full hash
        full_hash = color_hash + indicator_hash + shape_hash

        tries += 1
        if tries > 10000:
            logging.warning(
                f"Number of tries {tries} to find unique transformation hash exceeds 10000!"
            )

    # ---- indicator ----
    indicator_group = generate_group(
        color=indicator_color, size=size, shape=indicator_shape, grid_size=grid_size
    )

    # ---- record hash ----
    sample_hash = {
        "color_hash": color_hash,
        "shape_hash": shape_hash,
        "indicator_hash": indicator_hash,
        "full_hash": full_hash,
    }

    return (
        colors,
        color_transformation_color,
        shape_transformation_shape,
        indicator_group,
        color_transformation_kwargs,
        shape_transformation_kwargs,
        indicator_transformation_kwargs,
        sample_hash,
    )


def map_transformations(
    transformations: List[Callable],
) -> Dict[str, Any]:
    """
    Randomly selects transformations from a list of transformations and maps
    them to specified categories ('color_transformation', 'shape_transformation', 'indicator_transformation').

    Args:
        transformations (List[Callable]): A list of transformation functions.

    Returns:
        Dict[str, Callable]: A dictionary that maps indicators to selected transformation functions.
    """
    transformations = copy.copy(transformations)

    # Extract first transformation composition
    random.shuffle(transformations)
    shape_transform = transformations.pop()
    color_transform = transformations.pop()
    indicator_transform = transformations.pop()

    # Map the shuffled categories to the selected transformations
    transformation_mapping = {
        "shape_transformation": {
            "transformation": shape_transform,
            "name": TRANSFORMATION_NAMES[shape_transform],
        },
        "color_transformation": {
            "transformation": color_transform,
            "name": TRANSFORMATION_NAMES[color_transform],
        },
        "indicator_transformation": {
            "transformation": indicator_transform,
            "name": TRANSFORMATION_NAMES[indicator_transform],
        },
    }

    return transformation_mapping


def primitive_composition_params(
    first_transformation_specifier: Literal[
        "color_transformation", "shape_transformation", "indicator_transformation"
    ],
    second_transformation_specified: Literal[
        "color_transformation", "shape_transformation", "indicator_transformation"
    ],
    color_transformation_color: int,
    free_colors: List[int],
    shape_transformation_shape: str,
    indicator_group: Group,
) -> Tuple[List[int], str, str, List[Group]]:
    """
    This function generates parameters for a primitive composition based on the given transformation specifiers.

    Args:
        first_transformation_specifier (Literal["color_transformation", "shape_transformation", "indicator_transformation"]):
            The specifier for the first transformation.
        second_transformation_specified (Literal["color_transformation", "shape_transformation", "indicator_transformation"]):
            The specifier for the second transformation.
        color_transformation_color (int): The color for the color transformation.
        free_colors (List[int]): A list of free colors.
        shape_transformation_shape (str): The shape for the shape transformation.
        indicator_group (Group): The group for the indicator transformation.

    Returns:
        Tuple[List[int], str, str, List[Group]]: A tuple containing the function colors, shape, forbidden shape, and indicators.
    """
    if (
        first_transformation_specifier == "color_transformation"
        or second_transformation_specified == "color_transformation"
    ):
        function_colors = [color_transformation_color]
    else:
        function_colors = free_colors

    if (
        first_transformation_specifier == "shape_transformation"
        or second_transformation_specified == "shape_transformation"
    ):
        function_shape = shape_transformation_shape
        function_forbidden_shape = ""
    else:
        function_shape = ""
        function_forbidden_shape = shape_transformation_shape

    if (
        first_transformation_specifier == "indicator_transformation"
        or second_transformation_specified == "indicator_transformation"
    ):
        function_indicators = [indicator_group]
    else:
        function_indicators = []

    return (
        function_colors,
        function_shape,
        function_forbidden_shape,
        function_indicators,
    )


def primitive_params(
    transformation_specifier: Literal[
        "color_transformation", "shape_transformation", "indicator_transformation"
    ],
    color_transformation_color: int,
    free_colors: List[int],
    shape_transformation_shape: str,
    indicator_group: Group,
) -> Tuple[List[int], str, str, List[Group]]:
    """
    Returns a tuple of parameters based on the given transformation specifier.

    Args:
        transformation_specifier (Literal["color_transformation", "shape_transformation", "indicator_transformation"]):
            The specifier for the transformation.
        color_transformation_color (int): The color for the color transformation.
        free_colors (List[int]): A list of free colors.
        shape_transformation_shape (str): The shape for the shape transformation.
        indicator_group (Group): The group for the indicator transformation.

    Returns:
        Tuple[List[int], str, str, List[Group]]: A tuple containing the function colors, shape, forbidden shape, and indicators.
    """
    if transformation_specifier == "color_transformation":
        function_colors = [color_transformation_color]
    else:
        function_colors = free_colors

    if transformation_specifier == "shape_transformation":
        function_shape = shape_transformation_shape
        function_forbidden_shape = ""
    else:
        function_shape = ""
        function_forbidden_shape = shape_transformation_shape

    if transformation_specifier == "indicator_transformation":
        function_indicators = [indicator_group]
    else:
        function_indicators = []

    return (
        function_colors,
        function_shape,
        function_forbidden_shape,
        function_indicators,
    )


def has_unique_groups(query_groups: List[Tuple[List["Group"], Any]]) -> bool:
    """
    Checks if all input groups in the query groups are unique.

    Args:
        query_groups (List[Tuple[List[Group], any]]): A list of query groups, where each query group is
            a tuple containing:
            - A list of `Group` objects (input).
            - An associated value (ignored here).

    Returns:
        bool: True if all input groups are unique, False otherwise.
    """
    seen = set()

    for input, _ in query_groups:
        input_groups = frozenset((tuple(g.coordinates), g.color) for g in input)

        if input_groups in seen:
            return False
        seen.add(input_groups)

    return True


def gen_primitive_groups(
    transformation_mapping: Dict[str, Dict],
    transformation_specifier: Literal[
        "color_transformation", "shape_transformation", "indicator_transformation"
    ],
    color_transformation_color: int,
    free_colors: List[int],
    shape_transformation_shape: str,
    indicator_group: Group,
    num_primitives: int,
    grid_size: Tuple[int, int],
    num_tries: int,
    forbidden_shapes: Optional[List[str]] = None,
) -> List[Tuple[List[Group], List[Group]]] | None:
    """
    Generates a specified number of primitive groups based on the given transformation specifier.

    Args:
        transformation_mapping (Dict[str, Dict]): A dictionary mapping transformation names
            to dictionaries containing the transformation function and its keyword arguments.
        transformation_specifier (Literal["color_transformation", "shape_transformation", "indicator_transformation"]):
            The specifier for the transformation.
        color_transformation_color (int): The color for the color transformation.
        free_colors (List[int]): A list of free colors.
        shape_transformation_shape (str): The shape for the shape transformation.
        indicator_group (Group): The group for the indicator transformation.
        num_primitives (int): The number of primitives to generate.
        grid_size (Tuple[int, int]): The size of the grid.
        num_tries (int): The maximum number of tries to generate the input-output pairs.

    Returns:
        List[Tuple[List[Group], List[Group]]] | None: A list of tuples, where each tuple contains
            a list of input groups and a list of output groups. If no unique primitive groups can be found within the maximum number of tries, None is returned.
    """
    if forbidden_shapes is None:
        primitive_forbidden_shapes: List[str] = []
    else:
        primitive_forbidden_shapes = forbidden_shapes.copy()
    (
        primitive_colors,
        primitive_shape,
        primitive_forbidden_shape,
        primitive_indicators,
    ) = primitive_params(
        transformation_specifier=transformation_specifier,
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
    )
    primitive_forbidden_shapes.append(primitive_forbidden_shape)

    primitive_one_groups = gen_input_outputs(
        allowed_colors=primitive_colors,
        shape=primitive_shape,
        transformation=transformation_mapping[transformation_specifier][
            "transformation"
        ],
        transformation_kwargs=transformation_mapping[transformation_specifier][
            "transformation_kwargs"
        ],
        indicators=primitive_indicators,
        forbidden_shapes=primitive_forbidden_shapes,
        num_pairs=num_primitives,
        grid_size=grid_size,
        max_tries=num_tries,
    )
    return primitive_one_groups


def gen_function_groups(
    transformation_mapping: Dict[str, Dict],
    first_transformation_specifier: Literal[
        "color_transformation", "shape_transformation", "indicator_transformation"
    ],
    second_transformation_specifier: Literal[
        "color_transformation", "shape_transformation", "indicator_transformation"
    ],
    color_transformation_color: int,
    free_colors: List[int],
    shape_transformation_shape: str,
    indicator_group: Group,
    num_function_examples: int,
    grid_size: Tuple[int, int],
    num_tries: int,
    forbidden_shapes: Optional[List[str]] = None,
) -> List[Tuple[List[Group], List[Group]]] | None:
    """
    Generates input-output pairs of study examples for a given primitive composition.

    Args:
        transformation_mapping (Dict[str, Dict]): A dictionary mapping transformation names
            to dictionaries containing the transformation function and its keyword arguments.
        first_transformation_specifier (str): The specifier for the first transformation in the
            composition.
        second_transformation_specifier (str): The specifier for the second transformation in the
            composition.
        color_transformation_color (int): The color for the color transformation.
        free_colors (List[int]): A list of free colors.
        shape_transformation_shape (str): The shape for the shape transformation.
        indicator_group (Group): The group for the indicator transformation.
        num_function_examples (int): The number of function groups to generate.
        grid_size (Tuple[int, int]): The size of the grid.
        num_tries (int): The maximum number of tries to generate the input-output pairs.

    Returns:
        List[Tuple[List[Group], List[Group]]] | None: A list of input-output pairs of study examples.
            If no unique function groups can be found within the maximum number of tries, None is returned.
    """
    if forbidden_shapes is None:
        function_forbidden_shapes: List[str] = []
    else:
        function_forbidden_shapes = forbidden_shapes.copy()
    (
        function_colors,
        function_shape,
        function_forbidden_shape,
        function_indicators,
    ) = primitive_composition_params(
        first_transformation_specifier=first_transformation_specifier,
        second_transformation_specified=second_transformation_specifier,
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
    )
    function_forbidden_shapes.append(function_forbidden_shape)

    function_groups = gen_input_outputs(
        allowed_colors=function_colors,
        shape=function_shape,
        transformation=composition,
        transformation_kwargs={
            "transformation_one": transformation_mapping[
                first_transformation_specifier
            ]["transformation"],
            "transformation_one_kwargs": transformation_mapping[
                first_transformation_specifier
            ]["transformation_kwargs"],
            "transformation_two": transformation_mapping[
                second_transformation_specifier
            ]["transformation"],
            "transformation_two_kwargs": transformation_mapping[
                second_transformation_specifier
            ]["transformation_kwargs"],
        },
        indicators=function_indicators,
        forbidden_shapes=function_forbidden_shapes,
        num_pairs=num_function_examples,
        grid_size=grid_size,
        max_tries=num_tries,
    )

    return function_groups


def gen_query_groups(
    transformation_mapping: Dict[str, Dict],
    color_transformation_color: int,
    shape_transformation_shape: str,
    indicator_group: Group,
    num_queries: int,
    grid_size: Tuple[int, int],
    num_tries: int,
) -> List[Tuple[List[Group], List[Group]]] | None:
    """
    Generates query groups for a given set of transformations.

    Args:
        transformation_mapping (Dict[str, Dict]): A dictionary mapping transformation names
            to dictionaries containing the transformation function and its keyword arguments.
        color_transformation_color (int): The color for the color transformation.
        shape_transformation_shape (str): The shape for the shape transformation.
        indicator_group (Group): The group for the indicator transformation.
        num_queries (int): The number of queries to generate.
        grid_size (Tuple[int, int]): The size of the grid.
        num_tries (int): The maximum number of tries to generate the input-output pairs.

    Returns:
        List[Tuple[List[Group], List[Group]]] | None: A list of tuples, where each tuple contains
            a list of input groups and a list of output groups. If no unique query groups
            can be found within the maximum number of tries, None is returned.
    """
    function_composition_kwargs = {
        "transformation_one": composition,
        "transformation_one_kwargs": {
            "transformation_one": transformation_mapping["shape_transformation"][
                "transformation"
            ],
            "transformation_one_kwargs": transformation_mapping["shape_transformation"][
                "transformation_kwargs"
            ],
            "transformation_two": transformation_mapping["color_transformation"][
                "transformation"
            ],
            "transformation_two_kwargs": transformation_mapping["color_transformation"][
                "transformation_kwargs"
            ],
        },
        "transformation_two": transformation_mapping["indicator_transformation"][
            "transformation"
        ],
        "transformation_two_kwargs": transformation_mapping["indicator_transformation"][
            "transformation_kwargs"
        ],
    }

    query_groups = gen_input_outputs(
        allowed_colors=[color_transformation_color],
        shape=shape_transformation_shape,
        transformation=composition,
        transformation_kwargs=function_composition_kwargs,
        indicators=[indicator_group],
        forbidden_shapes=[""],
        num_pairs=num_queries,
        grid_size=grid_size,
        max_tries=num_tries,
    )

    return query_groups


def gen_episode(
    transformations: List[Callable],
    transformations_kwarg_options: Dict[str, Dict],
    num_primitives: int,
    num_queries: int,
    num_func_compositions: int,
    hashset: Set,
    grid_size: Tuple[int, int] = (10, 10),
    num_tries: int = 500,
) -> Tuple[Dict, Set]:
    """
    Generates study examples for a given mode, including level one and level two examples.

    Args:
        transformations (List[Callable]): The list of possible transformations.
        transformations_kwarg_options (Dict[str, Dict]): A dictionary with possible kwargs for each transformation.
        num_primitives (int): The number of primitive function examples to generate.
        num_func_compositions (int): The number of function composition examples to generate.
        num_queries (int): The number of queries to generate.
        hashset (Set): A set of hashes to ensure uniqueness.
        mode (Literal["train", "val", "test"]): The mode of generation, either "train", "val" or "test".
        grid_size (Tuple[int, int], optional): The size of the grid. Defaults to (10, 10).
        num_tries (int, optional): The number of attempts to generate examples. Defaults to 500.

    Returns:
        Tuple[dict, Set]: A tuple containing episodes and the updated hashset.
    """
    # --- select transformations ---
    transformation_mapping = map_transformations(transformations)

    shape_transformation_name = transformation_mapping["shape_transformation"]["name"]
    color_transformation_name = transformation_mapping["color_transformation"]["name"]
    indicator_transformation_name = transformation_mapping["indicator_transformation"][
        "name"
    ]

    # --- hash check ---
    (
        free_colors,
        color_transformation_color,
        shape_transformation_shape,
        indicator_group,
        color_transformation_kwargs,
        shape_transformation_kwargs,
        indicator_transformation_kwargs,
        sample_hash,
    ) = transformation_config(
        hashset=hashset,
        color_transformation_name=color_transformation_name,
        shape_transformation_name=shape_transformation_name,
        indicator_transformation_name=indicator_transformation_name,
        transformations_kwarg_options=transformations_kwarg_options,
        grid_size=grid_size,
    )

    transformation_mapping["color_transformation"][
        "transformation_kwargs"
    ] = color_transformation_kwargs
    transformation_mapping["shape_transformation"][
        "transformation_kwargs"
    ] = shape_transformation_kwargs
    transformation_mapping["indicator_transformation"][
        "transformation_kwargs"
    ] = indicator_transformation_kwargs

    # ---- complete function composition (query) ----
    query_groups = gen_query_groups(
        transformation_mapping=transformation_mapping,
        color_transformation_color=color_transformation_color,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_queries=num_queries,
        grid_size=grid_size,
        num_tries=num_tries,
    )
    if query_groups is None or not has_unique_groups(query_groups):
        return gen_episode(
            transformations=transformations,
            transformations_kwarg_options=transformations_kwarg_options,
            num_primitives=num_primitives,
            num_func_compositions=num_func_compositions,
            num_queries=num_queries,
            hashset=hashset,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    query_examples = set_groups_in_input_output_grids(query_groups, grid_size)

    # ---- function 1 examples ----
    shape_color_groups = gen_function_groups(
        transformation_mapping=transformation_mapping,
        first_transformation_specifier="shape_transformation",
        second_transformation_specifier="color_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_function_examples=num_func_compositions,
        grid_size=grid_size,
        num_tries=num_tries,
    )
    if shape_color_groups is None:
        return gen_episode(
            transformations=transformations,
            transformations_kwarg_options=transformations_kwarg_options,
            num_primitives=num_primitives,
            num_func_compositions=num_func_compositions,
            num_queries=num_queries,
            hashset=hashset,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    shape_color_examples = set_groups_in_input_output_grids(
        shape_color_groups, grid_size
    )

    # ---- function 2 examples ----
    shape_indicator_groups = gen_function_groups(
        transformation_mapping=transformation_mapping,
        first_transformation_specifier="shape_transformation",
        second_transformation_specifier="indicator_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_function_examples=num_func_compositions,
        grid_size=grid_size,
        num_tries=num_tries,
    )
    if shape_indicator_groups is None:
        return gen_episode(
            transformations=transformations,
            transformations_kwarg_options=transformations_kwarg_options,
            num_primitives=num_primitives,
            num_func_compositions=num_func_compositions,
            num_queries=num_queries,
            hashset=hashset,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    shape_indicator_examples = set_groups_in_input_output_grids(
        shape_indicator_groups, grid_size
    )

    # ---- function 3 examples ----
    color_indicator_groups = gen_function_groups(
        transformation_mapping=transformation_mapping,
        first_transformation_specifier="color_transformation",
        second_transformation_specifier="indicator_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_function_examples=num_func_compositions,
        grid_size=grid_size,
        num_tries=num_tries,
    )
    if color_indicator_groups is None:
        return gen_episode(
            transformations=transformations,
            transformations_kwarg_options=transformations_kwarg_options,
            num_primitives=num_primitives,
            num_func_compositions=num_func_compositions,
            num_queries=num_queries,
            hashset=hashset,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    color_indicator_examples = set_groups_in_input_output_grids(
        color_indicator_groups, grid_size
    )

    # ---- primitive 1 examples ----
    primitive_shape_transform_groups = gen_primitive_groups(
        transformation_mapping=transformation_mapping,
        transformation_specifier="shape_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_primitives=num_primitives,
        grid_size=grid_size,
        num_tries=num_tries,
    )
    if primitive_shape_transform_groups is None:
        return gen_episode(
            transformations=transformations,
            transformations_kwarg_options=transformations_kwarg_options,
            num_primitives=num_primitives,
            num_func_compositions=num_func_compositions,
            num_queries=num_queries,
            hashset=hashset,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    primitive_shape_examples = set_groups_in_input_output_grids(
        primitive_shape_transform_groups, grid_size
    )

    # ---- primitive 2 examples ----
    primitive_color_groups = gen_primitive_groups(
        transformation_mapping=transformation_mapping,
        transformation_specifier="color_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_primitives=num_primitives,
        grid_size=grid_size,
        num_tries=num_tries,
    )
    if primitive_color_groups is None:
        return gen_episode(
            transformations=transformations,
            transformations_kwarg_options=transformations_kwarg_options,
            num_primitives=num_primitives,
            num_func_compositions=num_func_compositions,
            num_queries=num_queries,
            hashset=hashset,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    primitive_color_examples = set_groups_in_input_output_grids(
        primitive_color_groups, grid_size
    )

    # ---- primitive 3 examples ----
    primitive_indicator_groups = gen_primitive_groups(
        transformation_mapping=transformation_mapping,
        transformation_specifier="indicator_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_primitives=num_primitives,
        grid_size=grid_size,
        num_tries=num_tries,
    )
    if primitive_indicator_groups is None:
        return gen_episode(
            transformations=transformations,
            transformations_kwarg_options=transformations_kwarg_options,
            num_primitives=num_primitives,
            num_func_compositions=num_func_compositions,
            num_queries=num_queries,
            hashset=hashset,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    primitive_indicator_examples = set_groups_in_input_output_grids(
        primitive_indicator_groups, grid_size
    )

    episode = {
        "primitive_functions": {
            "shape_transformation": primitive_shape_examples,
            "color_transformation": primitive_color_examples,
            "indicator_transformation": primitive_indicator_examples,
        },
        "function_compositions": {
            "shape_color_transformation": shape_color_examples,
            "shape_indicator_transformation": shape_indicator_examples,
            "color_indicator_transformation": color_indicator_examples,
        },
        "queries": query_examples,
        "meta_data": {
            "shape_transformation": {
                "type": shape_transformation_name,
                "kwargs": shape_transformation_kwargs,
            },
            "color_transformation": {
                "type": color_transformation_name,
                "kwargs": color_transformation_kwargs,
            },
            "indicator_transformation": {
                "type": indicator_transformation_name,
                "kwargs": indicator_transformation_kwargs,
            },
        },
    }
    hashset.add(sample_hash["full_hash"])

    return episode, hashset


def gen_static_episode_data(
    transformation_mapping: Dict[str, Dict[str, Any]],
    free_colors: List[int],
    color_transformation_color: int,
    shape_transformation_shape: str,
    indicator_group: Group,
    sample_hash: Dict[str, str],
    num_primitives: int,
    num_queries: int,
    num_func_compositions: int,
    hashset: Set,
    forbidden_shapes: Optional[List[str]] = None,
    grid_size: Tuple[int, int] = (10, 10),
    num_tries: int = 500,
) -> Tuple[Dict, Set]:
    """
    Generates study examples for a given mode, including level one and level two examples.

    Args:
        transformations (List[Callable]): The list of possible transformations.
        transformations_kwarg_options (Dict[str, Dict]): A dictionary with possible kwargs for each transformation.
        num_primitives (int): The number of primitive function examples to generate.
        num_func_compositions (int): The number of function composition examples to generate.
        num_queries (int): The number of queries to generate.
        hashset (Set): A set of hashes to ensure uniqueness.
        mode (Literal["train", "val", "test"]): The mode of generation, either "train", "val" or "test".
        grid_size (Tuple[int, int], optional): The size of the grid. Defaults to (10, 10).
        num_tries (int, optional): The number of attempts to generate examples. Defaults to 500.

    Returns:
        Tuple[dict, Set]: A tuple containing episodes and the updated hashset.
    """
    shape_transformation_name = transformation_mapping["shape_transformation"]["name"]
    color_transformation_name = transformation_mapping["color_transformation"]["name"]
    indicator_transformation_name = transformation_mapping["indicator_transformation"][
        "name"
    ]

    shape_transformation_kwargs = transformation_mapping["shape_transformation"][
        "transformation_kwargs"
    ]
    color_transformation_kwargs = transformation_mapping["color_transformation"][
        "transformation_kwargs"
    ]
    indicator_transformation_kwargs = transformation_mapping[
        "indicator_transformation"
    ]["transformation_kwargs"]

    # ---- complete function composition (query) ----
    query_groups = gen_query_groups(
        transformation_mapping=transformation_mapping,
        color_transformation_color=color_transformation_color,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_queries=num_queries,
        grid_size=grid_size,
        num_tries=num_tries,
    )
    if query_groups is None or not has_unique_groups(query_groups):
        return gen_static_episode_data(
            transformation_mapping=transformation_mapping,
            free_colors=free_colors,
            color_transformation_color=color_transformation_color,
            shape_transformation_shape=shape_transformation_shape,
            indicator_group=indicator_group,
            sample_hash=sample_hash,
            num_primitives=num_primitives,
            num_queries=num_queries,
            num_func_compositions=num_func_compositions,
            hashset=hashset,
            forbidden_shapes=forbidden_shapes,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    query_examples = set_groups_in_input_output_grids(query_groups, grid_size)

    # ---- function 1 examples ----
    shape_color_groups = gen_function_groups(
        transformation_mapping=transformation_mapping,
        first_transformation_specifier="shape_transformation",
        second_transformation_specifier="color_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_function_examples=num_func_compositions,
        grid_size=grid_size,
        num_tries=num_tries,
        forbidden_shapes=forbidden_shapes,
    )
    if shape_color_groups is None:
        return gen_static_episode_data(
            transformation_mapping=transformation_mapping,
            free_colors=free_colors,
            color_transformation_color=color_transformation_color,
            shape_transformation_shape=shape_transformation_shape,
            indicator_group=indicator_group,
            sample_hash=sample_hash,
            num_primitives=num_primitives,
            num_queries=num_queries,
            num_func_compositions=num_func_compositions,
            hashset=hashset,
            forbidden_shapes=forbidden_shapes,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    shape_color_examples = set_groups_in_input_output_grids(
        shape_color_groups, grid_size
    )

    # ---- function 2 examples ----
    shape_indicator_groups = gen_function_groups(
        transformation_mapping=transformation_mapping,
        first_transformation_specifier="shape_transformation",
        second_transformation_specifier="indicator_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_function_examples=num_func_compositions,
        grid_size=grid_size,
        num_tries=num_tries,
        forbidden_shapes=forbidden_shapes,
    )
    if shape_indicator_groups is None:
        return gen_static_episode_data(
            transformation_mapping=transformation_mapping,
            free_colors=free_colors,
            color_transformation_color=color_transformation_color,
            shape_transformation_shape=shape_transformation_shape,
            indicator_group=indicator_group,
            sample_hash=sample_hash,
            num_primitives=num_primitives,
            num_queries=num_queries,
            num_func_compositions=num_func_compositions,
            hashset=hashset,
            forbidden_shapes=forbidden_shapes,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    shape_indicator_examples = set_groups_in_input_output_grids(
        shape_indicator_groups, grid_size
    )

    # ---- function 3 examples ----
    color_indicator_groups = gen_function_groups(
        transformation_mapping=transformation_mapping,
        first_transformation_specifier="color_transformation",
        second_transformation_specifier="indicator_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_function_examples=num_func_compositions,
        grid_size=grid_size,
        num_tries=num_tries,
        forbidden_shapes=forbidden_shapes,
    )
    if color_indicator_groups is None:
        return gen_static_episode_data(
            transformation_mapping=transformation_mapping,
            free_colors=free_colors,
            color_transformation_color=color_transformation_color,
            shape_transformation_shape=shape_transformation_shape,
            indicator_group=indicator_group,
            sample_hash=sample_hash,
            num_primitives=num_primitives,
            num_queries=num_queries,
            num_func_compositions=num_func_compositions,
            hashset=hashset,
            forbidden_shapes=forbidden_shapes,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    color_indicator_examples = set_groups_in_input_output_grids(
        color_indicator_groups, grid_size
    )

    # ---- primitive 1 examples ----
    primitive_shape_transform_groups = gen_primitive_groups(
        transformation_mapping=transformation_mapping,
        transformation_specifier="shape_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_primitives=num_primitives,
        grid_size=grid_size,
        num_tries=num_tries,
        forbidden_shapes=forbidden_shapes,
    )
    if primitive_shape_transform_groups is None:
        return gen_static_episode_data(
            transformation_mapping=transformation_mapping,
            free_colors=free_colors,
            color_transformation_color=color_transformation_color,
            shape_transformation_shape=shape_transformation_shape,
            indicator_group=indicator_group,
            sample_hash=sample_hash,
            num_primitives=num_primitives,
            num_queries=num_queries,
            num_func_compositions=num_func_compositions,
            hashset=hashset,
            forbidden_shapes=forbidden_shapes,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    primitive_shape_examples = set_groups_in_input_output_grids(
        primitive_shape_transform_groups, grid_size
    )

    # ---- primitive 2 examples ----
    primitive_color_groups = gen_primitive_groups(
        transformation_mapping=transformation_mapping,
        transformation_specifier="color_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_primitives=num_primitives,
        grid_size=grid_size,
        num_tries=num_tries,
        forbidden_shapes=forbidden_shapes,
    )
    if primitive_color_groups is None:
        return gen_static_episode_data(
            transformation_mapping=transformation_mapping,
            free_colors=free_colors,
            color_transformation_color=color_transformation_color,
            shape_transformation_shape=shape_transformation_shape,
            indicator_group=indicator_group,
            sample_hash=sample_hash,
            num_primitives=num_primitives,
            num_queries=num_queries,
            num_func_compositions=num_func_compositions,
            hashset=hashset,
            forbidden_shapes=forbidden_shapes,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    primitive_color_examples = set_groups_in_input_output_grids(
        primitive_color_groups, grid_size
    )

    # ---- primitive 3 examples ----
    primitive_indicator_groups = gen_primitive_groups(
        transformation_mapping=transformation_mapping,
        transformation_specifier="indicator_transformation",
        color_transformation_color=color_transformation_color,
        free_colors=free_colors,
        shape_transformation_shape=shape_transformation_shape,
        indicator_group=indicator_group,
        num_primitives=num_primitives,
        grid_size=grid_size,
        num_tries=num_tries,
        forbidden_shapes=forbidden_shapes,
    )
    if primitive_indicator_groups is None:
        return gen_static_episode_data(
            transformation_mapping=transformation_mapping,
            free_colors=free_colors,
            color_transformation_color=color_transformation_color,
            shape_transformation_shape=shape_transformation_shape,
            indicator_group=indicator_group,
            sample_hash=sample_hash,
            num_primitives=num_primitives,
            num_queries=num_queries,
            num_func_compositions=num_func_compositions,
            hashset=hashset,
            forbidden_shapes=forbidden_shapes,
            grid_size=grid_size,
            num_tries=num_tries,
        )
    primitive_indicator_examples = set_groups_in_input_output_grids(
        primitive_indicator_groups, grid_size
    )

    episode = {
        "primitive_functions": {
            "shape_transformation": primitive_shape_examples,
            "color_transformation": primitive_color_examples,
            "indicator_transformation": primitive_indicator_examples,
        },
        "function_compositions": {
            "shape_color_transformation": shape_color_examples,
            "shape_indicator_transformation": shape_indicator_examples,
            "color_indicator_transformation": color_indicator_examples,
        },
        "queries": query_examples,
        "meta_data": {
            "shape_transformation": {
                "type": shape_transformation_name,
                "kwargs": shape_transformation_kwargs,
            },
            "color_transformation": {
                "type": color_transformation_name,
                "kwargs": color_transformation_kwargs,
            },
            "indicator_transformation": {
                "type": indicator_transformation_name,
                "kwargs": indicator_transformation_kwargs,
            },
        },
    }
    hashset.add(sample_hash["full_hash"])

    return episode, hashset
