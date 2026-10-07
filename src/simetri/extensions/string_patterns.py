"""String Art
The string_star function is based on the Electronic String Art book by
Stephen Erfle with some differences. This book is an incredibly in-depth
and accessible analysis of string art. If you are interested in this topic,
you should definitely buy it.
"""

from collections.abc import Sequence
from itertools import cycle
from typing import Any, TypeVar

from ..geom.geom_utils import connected_pairs
from ..geom.segments.line_utils import subdivide_segments
from ..helpers.utilities import get_cycle_size
from ..shapes.geom_items import reg_poly_points

ItemType = TypeVar("ItemType")


def get_skipped_items(
    items: Sequence[ItemType], skip: int | Sequence[int]
) -> list[ItemType]:
    """Return the first repeated traversal cycle through ``items``.

    The walk starts at index 0 and repeatedly advances by the values in
    ``skip`` modulo the item count. When ``skip`` is a sequence, its values
    are applied cyclically.

    Args:
        items: Sequence to traverse.
        skip: Step size or cyclic sequence of step sizes.

    Returns:
        list[ItemType]: Items visited in the first repeating cycle.

    Examples:
        >>> get_skipped_items([0, 1, 2, 3], 2)
        [0, 2]
        >>> get_skipped_items([0, 1, 2, 3], [40, 80])
        [0]
    """
    if isinstance(skip, int):
        skip_cycle = cycle([skip])
        n_skips = 2
    else:
        skip_cycle = cycle(skip)
        n_skips = len(skip) + 1

    n_items = len(items)
    indices = [0]
    last_idx = 0
    for i in range(n_items * n_skips):
        next_skip = next(skip_cycle)
        next_idx = (last_idx + next_skip) % n_items
        indices.append(next_idx)
        last_idx = next_idx

    n_ind_cycle = get_cycle_size(indices)

    return [items[idx] for idx in indices[:n_ind_cycle]]


def string_star(
    n_sides: int,
    n_subdivs: int,
    skip: int | Sequence[int],
    step: int | Sequence[int],
    radius: float = 100,
) -> list[Sequence[float]]:
    """Build the vertex walk for a string-art star pattern.

    The construction starts from a regular polygon, selects a skipped vertex
    cycle, subdivides the skipped edges, and then traverses those subdivided
    vertices using ``step``.

    Args:
        n_sides: Number of sides in the base regular polygon.
        n_subdivs: Number of equal subdivisions per skipped edge.
        skip: Step size or cyclic sequence of step sizes for the initial
            polygon-vertex walk.
        step: Step size or cyclic sequence of step sizes for the subdivided
            vertex walk.
        radius: Radius of the base regular polygon. Defaults to 100.

    Returns:
        list[Sequence[float]]: Vertex walk of the resulting string-art star.

    Examples:
        >>> star = string_star(7, 4, 3, 9)
        >>> len(star)
        29
        >>> [sg.round_point(star[idx], 2) for idx in (0, 10, 20, 28)]
        [(100.0, 0.0), (-13.87, -17.4), (-22.25, 97.49), (100.0, 0.0)]
    """
    reg_poly_vertices = reg_poly_points(n_sides, r=radius)

    skipped_vertices = get_skipped_items(reg_poly_vertices, skip)
    skipped_vertices += [skipped_vertices[0]]

    all_vertices = subdivide_segments(
        connected_pairs(skipped_vertices), n_subdivs, as_vertices=True
    )[:-1]

    stepped_vertices = get_skipped_items(all_vertices, step)
    stepped_vertices += [stepped_vertices[0]]

    return stepped_vertices
