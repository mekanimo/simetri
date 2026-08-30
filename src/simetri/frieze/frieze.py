"""Simetri graphics library's frieze patterns."""

from collections.abc import Sequence
from math import pi

from ..geometry.vectors import point_to_line_vec, vec_along_line
from ..graphics.batch import Group
from ..graphics.common import LineType, PointType, VecType
from ..graphics.shape import Shape


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
    """
    dx, dy = vector[:2]
    return design.translate(dx, dy, reps)


def p1(design: Group | Shape, vector: VecType = (1, 0), reps: int = 3) -> Group:
    """
    p1 symmetry group.

    Args:
        design (Group | Shape): The design to be repeated.
        vector (VecType, optional): The direction and distance of the hop. Defaults to (1, 0).
        reps (int, optional): The number of repetitions. Defaults to 3.

    Returns:
        Group: A Group of Shapes with the p1 symmetry.
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
    """
    dx, dy = vec_along_line(mirror_line, dist)[:2]
    design.mirror(mirror_line, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps)
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
    """
    x, y = point_to_line_vec(design.midpoint, mirror_line, unit=True)[:2]
    dx = x * dist
    dy = y * dist
    return design.mirror(mirror_line, reps=1).translate(dist, 0, reps)


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
    """
    x, y = point_to_line_vec(design.midpoint, mirror_line, unit=True)[:2]
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
    """
    design.rotate(pi, rotocenter, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps)
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
    """
    dx, dy = vec_along_line(mirror1, dist)[:2]
    design.mirror(mirror1, reps=1).mirror(mirror2, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps)
    return design


def spinning_sidle(
    design: Group,
    mirror_line: LineType = None,
    glide_line: LineType = None,
    glide_dist: float = None,
    trans_dist: float = None,
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
    """
    dx, dy = vec_along_line(glide_line, trans_dist)[:2]
    design.mirror(mirror_line, reps=1).glide(glide_line, glide_dist, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps)
    return design


def step(
    design: Group,
    glide_line: LineType = None,
    glide_dist: float = None,
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
    """
    design.glide(glide_line, glide_dist, reps=1)
    if reps > 0:
        dx, dy = vec_along_line(glide_line, 2 * glide_dist)[:2]
        design.translate(dx, dy, reps=reps)
    return design


def step_along(
    design: Group,
    glide_line: LineType = None,
    glide_dist: float = None,
    path: Sequence[PointType] = None,
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
    """
    design.glide(glide_dist, glide_line, reps=1)
    if reps > 0:
        design.translate_along(path, reps)
    return design
