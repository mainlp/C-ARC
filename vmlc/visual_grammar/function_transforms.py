"""
Function transformations (function compositions, etc.).
"""

from typing import Callable, List

from vmlc.visual_grammar import Group


def composition(
    group: Group | None,
    transformation_one: Callable[..., Group | None],
    transformation_one_kwargs: dict,
    transformation_two: Callable[..., Group | None],
    transformation_two_kwargs: dict,
) -> Group | None:
    """
    Apply a composition of two transformations to a `Group` object.

    Args:
        group (Group): The input group.
        transformation_one (Callable[..., Group | None]): The first transformation function.
        transformation_one_kwargs (dict): The keyword arguments for the first transformation.
        transformation_two (Callable[..., Group | None]): The second transformation function.
        transformation_two_kwargs (dict): The keyword arguments for the second transformation.

    Returns:
        Group | None: The transformed group or None if the transformation fails.
    """
    if group is None or len(group.coordinates) == 0:
        return group

    new_group = transformation_one(group=group, **transformation_one_kwargs)
    if new_group is not None and len(new_group.coordinates) > 0:
        new_group = transformation_two(group=new_group, **transformation_two_kwargs)
    return new_group


def gen_indicator_shapes() -> List[str]:
    """
    Generates a list of unique indicator shapes based on the specified mode.

    Returns:
        List[str]: A list of unique indicator shapes as strings.
    """
    return [
        "00",
        "0001",
        "0010",
        "1001",
        "0011",
        "011011",
        "000110",
        "001011",
        "00011011",
        "000111",
    ]
