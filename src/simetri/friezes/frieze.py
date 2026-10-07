"""Simetri graphics library's frieze patterns.

Implements the seven frieze symmetries (hop, jump, sidle, step, and
spinning variants) by transforming a motif along a strip.

**Examples**

```python
import simetri.graphics as sg
from simetri.friezes import frieze
motif = sg.Circle(10)
strip = frieze.hop(motif, vector=(40, 0), reps=4)
```
"""

from collections.abc import Sequence
from math import pi

from ..base.common import LineType, PointType, VecType
from ..geom.vectors import point_to_line_vec, vec_along_line
from ..group.batch import Group
from ..shapes.shape import Shape


def hop(
    design: Group | Shape, vector: VecType = (1, 0), reps: int = 3
) -> Group:
    """
    p1 symmetry group.

    Args:
        design (Group | Shape): The design to be repeated.
        vector (VecType, optional): The direction and distance of the hop. Defaults to (1, 0).
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of Shapes with the p1 symmetry.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> row = hop(mark, vector=(20, 0), reps=2)
        >>> row.__class__.__name__
        'Group'
        >>> len(row)
        3
    """
    dx, dy = vector[:2]
    return design.translate(dx, dy, reps=reps)


def p1(design: Group | Shape, vector: VecType = (1, 0), reps: int = 3) -> Group:
    """
    p1 symmetry group.

    Args:
        design (Group | Shape): The design to be repeated.
        vector (VecType, optional): The direction and distance of the hop. Defaults to (1, 0).
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of Shapes with the p1 symmetry.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> row = p1(mark, vector=(15, 0), reps=1)
        >>> len(row)
        2
    """
    return hop(design, vector, reps)


def jump(
    design: Group | Shape,
    mirror_line: LineType,
    dist: float,
    reps: int = 3,
) -> Group:
    """
    p11m symmetry group.

    Args:
        design (Group | Shape): The design to be repeated.
        mirror_line (Line): The line to mirror the design.
        dist (float): The distance between the shapes.
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of shapes with the p11m symmetry.

    Examples:
        >>> import simetri.graphics as sg
        >>> band = sg.Group([sg.Shape([(0, 0), (40, 0)])])
        >>> axis = ((0, -20), (100, -20))
        >>> row = jump(band, axis, 20, reps=1)
        >>> len(row)
        4
        >>> row is band
        True
    """
    dx, dy = vec_along_line(mirror_line, dist)[:2]
    design.mirror(mirror_line, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps=reps)
    return design


def jump_along(
    design: Group,
    mirror_line: LineType,
    path: Sequence[PointType],
    reps: int = 3,
) -> Group:
    """
    Jump along the given path.

    Args:
        design (Group): The design to be repeated.
        mirror_line (Line): The line to mirror the design.
        path (Sequence[PointType]): The path along which to translate the design.
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of shapes with the jump along symmetry.

    Examples:
        >>> import simetri.graphics as sg
        >>> band = sg.Group([sg.Shape([(0, 0), (40, 0)])])
        >>> axis = ((0, -20), (100, -20))
        >>> path = [(0, 0), (30, 0)]
        >>> row = jump_along(band, axis, path, reps=1)
        >>> len(row)
        3
        >>> row is band
        True
    """
    design.mirror(mirror_line, reps=1)
    if reps > 0:
        design.translate_along(path, reps)
    return design


def sidle(
    design: Group, mirror_line: LineType, dist: float, reps: int = 3
) -> Group:
    """
    p1m1 symmetry group.

    Args:
        design (Group): The design to be repeated.
        mirror_line (Line): The line to mirror the design.
        dist (float): The distance between the shapes.
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of Shapes with the sidle symmetry.

    Examples:
        >>> import simetri.graphics as sg
        >>> band = sg.Group([sg.Shape([(0, 0), (40, 0)])])
        >>> axis = ((0, -20), (100, -20))
        >>> row = sidle(band, axis, 20, reps=1)
        >>> len(row)
        4
        >>> row is band
        True
    """

    return design.mirror(mirror_line, reps=1).translate(dist, 0, reps=reps)


def sidle_along(
    design: Group,
    mirror_line: LineType,
    path: Sequence[PointType],
    reps: int = 3,
) -> Group:
    """
    Sidle along the given path.

    Args:
        design (Group): The design to be repeated.
        mirror_line (Line): The line to mirror the design.
        path (Sequence[PointType]): The path along which to translate the design.
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of shapes with the sidle along symmetry.

    Examples:
        >>> import simetri.graphics as sg
        >>> band = sg.Group([sg.Shape([(0, 0), (40, 0)])])
        >>> axis = ((0, -20), (100, -20))
        >>> path = [(0, 0), (30, 0)]
        >>> row = sidle_along(band, axis, path, reps=1)
        >>> len(row)
        3
        >>> row is band
        True
    """

    design.mirror(mirror_line, reps=1)
    return design.translate_along(path, reps)


def spinning_hop(
    design: Group, rotocenter: PointType, dx: float, dy: float, reps: int = 3
) -> Group:
    """
    p2 symmetry group.

    Args:
        design (Group): The design to be repeated.
        rotocenter (PointType): The center of rotation.
        dx (float): The distance to translate in the x direction.
        dy (float): The distance to translate in the y direction.
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of Shapes with spinning hop symmetry.

    Examples:
        >>> import simetri.graphics as sg
        >>> band = sg.Group([sg.Shape([(0, 0), (40, 0)])])
        >>> row = spinning_hop(band, (20, 0), 20, 0, reps=1)
        >>> len(row)
        4
        >>> row is band
        True
    """
    design.rotate(pi, rotocenter, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps=reps)
    return design


def spinning_jump(
    design: Group,
    mirror1: LineType,
    mirror2: LineType,
    dist: float,
    reps: int = 3,
) -> Group:
    """
    p2mm symmetry group.

    Args:
        design (Group): The design to be repeated.
        mirror1 (Line): The first mirror line.
        mirror2 (Line): The second mirror line.
        dist (float): The distance between the shapes along mirror1.
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of Shapes with spinning jump symmetry.

    Examples:
        >>> import simetri.graphics as sg
        >>> band = sg.Group([sg.Shape([(0, 0), (40, 0)])])
        >>> row = spinning_jump(
        ...     band,
        ...     ((0, -20), (100, -20)),
        ...     ((0, 0), (0, 100)),
        ...     20,
        ...     reps=1,
        ... )
        >>> len(row)
        8
        >>> row is band
        True
    """
    dx, dy = vec_along_line(mirror1, dist)[:2]
    design.mirror(mirror1, reps=1).mirror(mirror2, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps=reps)
    return design


def spinning_sidle(
    design: Group,
    mirror_line: LineType = None,
    glide_line: LineType = None,
    glide_dist: float | None = None,
    trans_dist: float | None = None,
    reps: int = 3,
) -> Group:
    """
    p2mg symmetry group.

    Args:
        design (Group): The design to be repeated.
        mirror_line (Line, optional): The mirror line. Defaults to None.
        glide_line (Line, optional): The glide line. Defaults to None.
        glide_dist (float, optional): The distance of the glide. Defaults to None.
        trans_dist (float, optional): The distance of the translation. Defaults to None.
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of Shapes with spinning sidle symmetry.

    Examples:
        >>> import simetri.graphics as sg
        >>> band = sg.Group([sg.Shape([(0, 0), (40, 0)])])
        >>> row = spinning_sidle(
        ...     band,
        ...     ((0, -20), (100, -20)),
        ...     ((0, -20), (100, -20)),
        ...     10,
        ...     20,
        ...     reps=1,
        ... )
        >>> len(row)
        8
        >>> row is band
        True
    """
    dx, dy = vec_along_line(glide_line, trans_dist)[:2]
    design.mirror(mirror_line, reps=1).glide(glide_line, glide_dist, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps=reps)
    return design


def step(
    design: Group,
    glide_line: LineType = None,
    glide_dist: float | None = None,
    reps: int = 3,
) -> Group:
    """
    p11g symmetry group.

    Args:
        design (Group): The design to be repeated.
        glide_line (Line, optional): The glide line. Defaults to None.
        glide_dist (float, optional): The distance of the glide. Defaults to None.
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of Shapes with step symmetry.

    Examples:
        >>> import simetri.graphics as sg
        >>> band = sg.Group([sg.Shape([(0, 0), (40, 0)])])
        >>> axis = ((0, -20), (100, -20))
        >>> row = step(band, axis, 10, reps=1)
        >>> len(row)
        4
        >>> row is band
        True
    """
    design.glide(glide_line, glide_dist, reps=1)
    if reps > 0:
        dx, dy = vec_along_line(glide_line, 2 * glide_dist)[:2]
        design.translate(dx, dy, reps=reps)
    return design


def step_along(
    design: Group,
    glide_line: LineType = None,
    glide_dist: float | None = None,
    path: Sequence[PointType] | None = None,
    reps: int = 3,
) -> Group:
    """
    Step along a path.

    Args:
        design (Group): The design to be repeated.
        glide_line (Line, optional): The glide line. Defaults to None.
        glide_dist (float, optional): The distance of the glide. Defaults to None.
        path (Sequence[PointType], optional): The path along which to translate the design. Defaults to None.
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of shapes with the step along symmetry.

    Examples:
        >>> import simetri.graphics as sg
        >>> band = sg.Group([sg.Shape([(0, 0), (40, 0)])])
        >>> axis = ((0, -20), (100, -20))
        >>> path = [(0, 0), (30, 0)]
        >>> row = step_along(band, axis, 10, path, reps=1)
        >>> len(row)
        3
        >>> row is band
        True
    """
    design.glide(glide_line, glide_dist, reps=1)
    if reps > 0:
        design.translate_along(path, reps)
    return design
