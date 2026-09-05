"""Shared constants, type aliases, and ID helpers for Simetri graphics.

Unit constants convert physical lengths to PostScript points (1 inch = 72 pt).
Type aliases such as ``PointType`` and ``LineType`` are used throughout the
graphics and geometry APIs.

Examples:
    >>> import simetri.graphics as sg
    >>> print(sg.INCH, sg.CM, sg.phi
    72 28.3464 1.618033988749895
    >>> width_pt = 2 * INCH  # 144 points
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from math import cos, pi, sin
from typing import TYPE_CHECKING, Union

from ..helpers.vector import Vector2D
from ..config.settings import VOID, defaults

if TYPE_CHECKING:
    from ..shapes.shape import Shape

# These are used for type hinting and annotations
GraphEdgeType = tuple[int, int]
LineType = Sequence[Sequence]
MatrixType = Sequence[Sequence[float]]
PointType = Sequence[float]
PolygonLike = Union["Shape", Sequence[PointType]]
PolygonType = Sequence[PointType]
PolylineType = Sequence[PointType]
TurnPair = tuple[float, float]
TurnSequence = Sequence[TurnPair]
VecType = Sequence[float]

INCH = 72  # (used for converting inches to points)
CM = 28.3464  # (used for converting centimeters to points)
MM = 2.83464  # (used for converting millimeters to points)
# 2 * inch is equal to 144 points
# 10 * cm is equal to 283.46456 points


UNDER: bool = True

# Pre-computed values
two_pi = 2 * pi  # 360 degrees
tau = 2 * pi  # 360 degrees
phi = (1 + 5**0.5) / 2  # golden ratio


def gen_unique_ids() -> Iterator[int]:
    """Yield an infinite sequence of unique integer IDs.

    Every drawable object in Simetri receives an ID from this generator
    (via ``get_unique_id``).

    Yields:
        int: The next unique identifier, starting at 0.

    Examples:
        >>> gen = gen_unique_ids()
        >>> next(gen), next(gen)
        (0, 1)
    """
    id_ = 0
    while True:
        yield id_
        id_ += 1


unique_id = gen_unique_ids()

d_id_obj = {}  # for Shape objects


def get_unique_id(item) -> int:
    """Allocate a unique ID and register ``item`` in ``d_id_obj``.

    Args:
        item: Object to register (typically a Shape, Group, or sketch).

    Returns:
        int: Newly assigned unique identifier.

    Examples:
        >>> class _T: pass
        >>> get_unique_id(_T())  # doctest: +SKIP
        0
    """
    id_ = next(unique_id)
    d_id_obj[id_] = item
    return id_


origin = (0.0, 0.0)  # used for a point at the origin
axis_x = (origin, (1.0, 0.0))  # used for a line along x axis
axis_y = (origin, (0.0, 1.0))  # used for a line along y axis
axis_diag1 = (origin, (1.0, 1.0))  # used for a line along y = x
axis_diag2 = (origin, (1.0, -1.0))

axis_hex = (
    (0.0, 0.0),
    (cos(pi / 3), sin(pi / 3)),
)  # used for 3 and 6 rotation symmetries

i_vec = Vector2D(1.0, 0.0)  # x direction unit vector
j_vec = Vector2D(0.0, 1.0)  # y direction unit vector


def _set_Nones(obj, args, values):
    """
    Internally used in instance construction to set default values for None values.

    Args:
        obj (Any): The object to set values for.
        args (list): The arguments to set.
        values (list): The values to set.
    """
    for i, arg in enumerate(args):
        if values[i] is None:
            setattr(obj, arg, defaults[arg])
        else:
            setattr(obj, arg, values[i])


def get_defaults(args, values):
    """
    Internally used in instance construction to set default values for None values.

    Args:
        args (list): The arguments to set.
        values (list): The values to set.

    Returns:
        list: The default values.
    """
    res = len(args) * [None]
    for i, arg in enumerate(args):
        if values[i] is None:
            res[i] = defaults[arg]
        else:
            res[i] = values[i]

    return res
