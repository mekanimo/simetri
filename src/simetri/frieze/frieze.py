"""Simetri graphics library's frieze patterns."""

from typing import Sequence, Union

from ..helpers.geometry import vec_along_line, point_to_line_vec, pi
from ..graphics.common import VecType, Line, Point
from ..graphics.batch import Batch

from ..graphics.shape import Shape



def hop(design: Union[Batch, Shape], vector: VecType = (1, 0), reps: int = 3) -> Batch:
    """p1 symmetry group.
    vector argument's orientation is the direction of the hop.
    vector argument's magnitude is the distance between the shapes.
    Return a Batch of Shapes with the p1 symmetry."""
    dx, dy = vector
    return design.translate(dx, dy, reps)


# this is the same as hop
def p1(design: Union[Batch, Shape], vector: VecType = (1, 0), reps: int = 3) -> Batch:
    """p1 symmetry group.
    vector argument's orientation is the direction of the hop.
    vector argument's magnitude is the distance between the shapes.
    Return a Batch of Shapes with the p1 symmetry."""
    return hop(design, vector, reps)


def jump(
    design: Union[Batch, Shape],
    mirror_line: Line,
    dist: float,
    reps: int = 3,
) -> Batch:
    """p11m symmetry group.
    mirror_line can be axis_x, axis_y, or (p1, p2)
    Return a Batch of shapes with the p11m symmetry."""
    dx, dy = vec_along_line(mirror_line, dist)
    design.mirror(mirror_line, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps)
    return design


def jump_along(
    design: Batch,
    mirror_line: Line,
    path: Sequence[Point],
    reps: int = 3,
) -> Batch:
    """Jump along the given path."""
    design.mirror(mirror_line, reps=1)
    if reps > 0:
        design.translate_along(path, reps)
    return design


def sidle(design: Batch, mirror_line: Line, dist: float, reps: int = 3) -> Batch:
    """p1m1 symmetry group.
    mirror_line can be axis_x/axis_y, or (p1, p2)
    Return a Batch of Shapes with the sidle symmetry."""
    # vector(dx, dy) is a unit vector perpendicular to mirror_line
    x, y = point_to_line_vec(design.center, mirror_line, unit=True)
    dx = x * dist
    dy = y * dist
    return design.mirror(mirror_line, reps=1).translate(dist, 0, reps)


def sidle_along(
    design: Batch, mirror_line: Line, path: Sequence[Point], reps: int = 3
) -> Batch:
    """sidle along the given path."""
    # vector(dx, dy) is a unit vector perpendicular to mirror_line
    x, y = point_to_line_vec(design.center, mirror_line, unit=True)
    design.mirror(mirror_line, reps=1)
    return design.translate_along(path, reps)


def spinning_hop(
    design: Batch, rotocenter: Point, dx: float, dy: float, reps: int = 3
) -> Batch:
    """p2 symmetry group.
    Return a Batch of Shapes with spinning hop symmetry."""
    design.rotate(pi, rotocenter, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps)
    return design


def spinning_jump(
    design: Batch, mirror1: Line, mirror2: Line, dist: float, reps: int = 3
) -> Batch:
    """p2mm symmetry group.
    mirrors can be axis_x, axis_y, or (p1, p2)
    distance is along mirror1
    Return a Batch of Shapes with spinning_hop symmetry."""
    dx, dy = vec_along_line(mirror1, dist)
    design.mirror(mirror1, reps=1).mirror(mirror2, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps)
    return design


def spinning_sidle(
    design: Batch,
    mirror_line: Line = None,
    glide_line: Line = None,
    glide_dist: float = None,
    trans_dist: float = None,
    reps: int = 3,
) -> Batch:
    """p2mg symmetry group.
    lines can be axis_x, axis_y, or (p1, p2)
    Return a Batch of Shapes with spinning_sidle symmetry."""
    dx, dy = vec_along_line(glide_line, trans_dist)
    design.mirror(mirror_line, reps=1).glide(glide_line, glide_dist, reps=1)
    if reps > 0:
        design.translate(dx, dy, reps)
    return design


def step(
    design: Batch,
    glide_line: Line = None,
    glide_dist: float = None,
    reps: int = 3,
) -> Batch:
    """p11g symmetry group.
    glide_line can be axis_x, axis_y, or (p1, p2)
    Return a Batch of Shapes with step symmetry."""
    design.glide(glide_line, glide_dist, reps=1)
    if reps > 0:
        dx, dy = vec_along_line(glide_line, 2 * glide_dist)
        design.translate(dx, dy, reps=reps)
    return design


def step_along(
    design: Batch,
    glide_line: Line = None,
    glide_dist: float = None,
    path: Sequence[Point] = None,
    reps: int = 3,
) -> Batch:
    """Step along a path."""
    design.glide(glide_dist, glide_line, reps=1)
    if reps > 0:
        design.translate_along(path, reps)
    return design
