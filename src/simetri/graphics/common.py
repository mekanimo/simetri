"""Simetri library's constants and common data."""

from math import pi, cos, sin
from typing import Sequence, Tuple, Any, Iterator
from ..settings.settings import defaults
from ..helpers.vector import Vector2D

Point = Sequence[float]  # used for type hinting
Vec2 = Tuple[float, float]  # used for type hinting size, scale, offset etc.
Line = Sequence[Sequence]  # used for type hinting
VecType = Sequence[float]  # used for type hinting
Polyline = Sequence[Point]  # used for type hinting
Polygon = Sequence[Point]  # used for type hinting
GraphEdge = Tuple[int, int]  # used for type hinting
Matrix = Sequence[Sequence[float]]  # used for type hinting

INCH = 72  # (used for converting inches to points)
CM = 28.3464  # (used for converting centimeters to points)
MM = 2.83464  # (used for converting millimeters to points)
# 2 * inch is equal to 144 points
# 10 * cm is equal to 283.46456 points

VOID = "VOID"  # used for non-existent values
UNDER: bool = True

# Pre-computed values
two_pi = 2 * pi  # 360 degrees
tau = 2 * pi  # 360 degrees
phi = (1 + 5**0.5) / 2  # golden ratio

_d_id_obj = {}  # dictionary of obj.id: obj, use get_item_by_id(id)

def common_properties(obj, graphics_object=True, id_only=False):
    """
    Set common properties for an object. All objects in Simetri have these properties.

    Args:
        obj (Any): The object to set properties for.
        graphics_object (bool, optional): Whether the object is a graphics object. Defaults to True.
        id_only (bool, optional): Whether to set only the id. Defaults to False.
    """
    obj.id = get_unique_id(obj)
    _d_id_obj[obj.id] = obj
    if id_only:
        return
    obj.active = True
    if graphics_object:
        obj.visible = True

def gen_unique_ids() -> Iterator[int]:
    """
    Generate unique Ids.
    Every object in Simetri has a unique id.

    Yields:
        Iterator[int]: A unique id.
    """
    id_ = 0
    while True:
        yield id_
        id_ += 1

def get_item_by_id(id_: int) -> Any:
    """
    Return an object by its id.

    Args:
        id_ (int): The id of the object.

    Returns:
        Any: The object with the given id.
    """
    return _d_id_obj[id_]

unique_id = gen_unique_ids()

def get_unique_id(item) -> int:
    """
    Return a unique id.
    Every object in Simetri has a unique id.
    Register the object in _d_id_obj.

    Args:
        item (Any): The object to get a unique id for.

    Returns:
        int: The unique id.
    """
    id_ = next(unique_id)
    _d_id_obj[id_] = item
    return id_

origin = (0.0, 0.0)  # used for a point at the origin
axis_x = (origin, (1.0, 0.0))  # used for a line along x axis
axis_y = (origin, (0.0, 1.0))  # used for a line along y axis

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
