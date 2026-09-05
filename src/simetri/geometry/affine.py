"""3×3 affine transformation matrices and point-transform helpers.

Matrices are stored in row-major form suitable for ``points @ matrix`` with
homogeneous coordinates. Public aliases on ``simetri.graphics`` include
``TM``, ``RM``, ``MM``, ``GM``, ``SM``, and ``SHM``.

Examples:
    >>> import simetri.graphics as sg
    >>> from math import pi
    >>> M = sg.translation_matrix(10, 20)
    >>> R = sg.rotation_matrix(pi / 2, about=(0, 0))
"""

from __future__ import annotations

from collections.abc import Sequence
from math import atan2, cos, sin, tan
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..helpers.validation import is_line, is_point
from .homogenize import homogenize
from .vectors import vec_along_line
from ..core.common import LineType, PointType

if TYPE_CHECKING:
    from ..group.batch import Group
    from ..graphics.shape import Shape


def identity_matrix() -> NDArray:
    """Return the 3×3 identity matrix.

    Returns:
        np.ndarray: ``[[1, 0, 0], [0, 1, 0], [0, 0, 1]]``.

    Examples:
        >>> identity_matrix()[0, 0]
        1.0
    """
    return np.identity(3)


def xform_matrix(
    a: float, b: float, c: float, d: float, e: float, f: float
) -> NDArray:
    """
    Return a transformation matrix in row-major form
    [[a, b, 0], [c, d, 0], [e, f, 1.0]].

    Args:
        a (float): The a component of the transformation matrix.
        b (float): The b component of the transformation matrix.
        c (float): The c component of the transformation matrix.
        d (float): The d component of the transformation matrix.
        e (float): The e component of the transformation matrix.
        f (float): The f component of the transformation matrix.

    Returns:
        np.ndarray: The transformation matrix.
    """
    return np.array([[a, b, 0], [c, d, 0], [e, f, 1.0]])


def translation_matrix(dx: float, dy: float) -> NDArray:
    """Return a translation matrix in row-major form.

    The matrix is ``[[1, 0, 0], [0, 1, 0], [dx, dy, 1]]``.

    Args:
        dx: Translation along the x-axis.
        dy: Translation along the y-axis.

    Returns:
        np.ndarray: The translation matrix.

    Examples:
        >>> import simetri.graphics as sg
        >>> M = sg.translation_matrix(5, -2)
        >>> M[2, 0], M[2, 1]
        (5.0, -2.0)
    """
    return np.array([[1.0, 0, 0], [0, 1.0, 0], [dx, dy, 1.0]])


def inv_translation_matrix(dx: float, dy: float) -> NDArray:
    """
    Return the inverse of a translation matrix in row-major form
    [[1.0, 0, 0], [0, 1.0, 0], [-dx, -dy, 1.0]].

    Args:
        dx (float): The translation distance along the x-axis.
        dy (float): The translation distance along the y-axis.

    Returns:
        np.ndarray: The inverse translation matrix.
    """
    return np.array([[1.0, 0, 0], [0, 1.0, 0], [-dx, -dy, 1.0]])


def rot_about_origin_matrix(angle: float) -> NDArray:
    """
    Return a rotation matrix in row-major form
    [[cos(angle), sin(angle), 0], [-sin(angle), cos(angle), 0], [0, 0, 1.0]].

    Args:
        angle (float): The rotation angle in radians.

    Returns:
        np.ndarray: The rotation matrix.
    """
    c = cos(angle)
    s = sin(angle)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])


def rotation_matrix(angle: float, about=(0, 0)) -> NDArray:
    """Return a rotation matrix about a point.

    Composes translate-to-origin, rotate by ``angle`` (radians), then
    translate back. row-major form for ``points @ matrix``.

    Args:
        angle: Rotation angle in radians (counterclockwise).
        about: Point to rotate about. Defaults to ``(0, 0)``.

    Returns:
        np.ndarray: The rotation matrix.

    Examples:
        >>> import simetri.graphics as sg
        >>> from math import pi
        >>> M = sg.rotation_matrix(pi / 2)
        >>> round(M[0, 1], 6)
        1.0
    """
    dx, dy = about[:2]
    # translate 'about' to the origin
    trans_mat = translation_matrix(-dx, -dy)
    # rotate around the origin
    rot_mat = rot_about_origin_matrix(angle)
    # translate it back to initial pos
    inv_trans_mat = translation_matrix(dx, dy)
    # compose the transformation matrix
    return trans_mat @ rot_mat @ inv_trans_mat


def inv_rotation_matrix(angle: float, about=(0, 0)) -> NDArray:
    """
    Construct the inverse of a rotation matrix that can be used to rotate a point
    about another point by angle float.
    Return a rotation matrix in row-major form
    dx, dy = about
    [[cos(angle), -sin(angle), 0],
    [sin(angle), cos(angle), 0],
    -cos(angle)dx-sin(angle)dy+x, -sin(angle)dx+cos(angle)dy+y, 1]].

    Args:
        angle (float): The rotation angle in radians.
        about (tuple, optional): The point to rotate about, defaults to (0, 0).

    Returns:
        np.ndarray: The inverse rotation matrix.
    """
    dx, dy = about[:2]
    # translate 'about' to the origin
    trans_mat = translation_matrix(-dx, -dy)
    # rotate around the origin
    rot_mat = rot_about_origin_matrix(angle)
    # translate it back to initial pos
    inv_trans_mat = translation_matrix(dx, dy)
    # compose the transformation matrix
    return inv_trans_mat @ rot_mat.T @ trans_mat


def glide_matrix(mirror_line: LineType, distance: float) -> NDArray:
    """
    Return a glide-reflection matrix in row-major form.
    Reflect about the given vector then translate by dx
    along the same vector.

    Args:
        mirror_line (LineType): The line to mirror about.
        distance (float): The distance to translate along the line.

    Returns:
        np.ndarray: The glide-reflection matrix.
    """
    mirror_mat = mirror_about_line_matrix(mirror_line)
    x, y = vec_along_line(mirror_line, distance)[:2]
    trans_mat = translation_matrix(x, y)

    return mirror_mat @ trans_mat


def inv_glide_matrix(mirror_line: LineType, distance: float) -> NDArray:
    """
    Return the inverse of a glide-reflection matrix in row-major form.
    Reflect about the given vector then translate by dx
    along the same vector.

    Args:
        mirror_line (LineType): The line to mirror about.
        distance (float): The distance to translate along the line.

    Returns:
        np.ndarray: The inverse glide-reflection matrix.
    """
    mirror_mat = mirror_about_line_matrix(mirror_line)
    x, y = vec_along_line(mirror_line, distance)[:2]
    trans_matrix = translation_matrix(x, y)

    return trans_matrix @ mirror_mat


def scale_matrix(scale_x: float, scale_y: float | None = None) -> NDArray:
    """
    Return a scale matrix in row-major form.

    Args:
        scale_x (float): Scale factor in x direction.
        scale_y (float, optional): Scale factor in y direction, defaults to None.

    Returns:
        np.ndarray: A scale matrix in row-major form.
    """
    if scale_y is None:
        scale_y = scale_x
    return np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1.0]])


def inv_scale_matrix(scale_x: float, scale_y: float | None = None) -> NDArray:
    """
    Return the inverse of a scale matrix in row-major form.

    Args:
        scale_x (float): Scale factor in x direction.
        scale_y (float, optional): Scale factor in y direction, defaults to None.

    Returns:
        np.ndarray: The inverse of a scale matrix in row-major form.
    """
    if scale_y is None:
        scale_y = scale_x
    return np.array([[1 / scale_x, 0, 0], [0, 1 / scale_y, 0], [0, 0, 1.0]])


def scale_in_place_matrix(
    scale_x: float, scale_y: float, about: PointType = (0, 0)
) -> NDArray:
    """
    Return a scale matrix in row-major form that scales about a point.

    Args:
        scale_x (float): Scale factor in x direction.
        scale_y (float): Scale factor in y direction.
        about (PointType): PointType about which the scaling is performed.

    Returns:
        np.ndarray: A scale matrix in row-major form that scales about a point.
    """
    dx, dy = about[:2]
    trans_mat = translation_matrix(-dx, -dy)
    scale_mat = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1.0]])
    inv_trans_mat = translation_matrix(dx, dy)
    return trans_mat @ scale_mat @ inv_trans_mat


def shear_matrix(angle_x: float, angle_y: float = 0) -> NDArray:
    """
    Return a shear matrix in row-major form.

    Args:
        angle_x (float): Angle of shear in x direction.
        angle_y (float, optional): Angle of shear in y direction, defaults to 0.

    Returns:
        np.ndarray: A shear matrix in row-major form.
    """
    return np.array([[1, tan(angle_y), 0], [tan(angle_x), 1, 0], [0, 0, 1.0]])


def inv_shear_matrix(angle_x: float, angle_y: float = 0) -> NDArray:
    """
    Return the inverse of a shear matrix in row-major form.

    Args:
        angle_x (float): Angle of shear in x direction.
        angle_y (float, optional): Angle of shear in y direction, defaults to 0.

    Returns:
        np.ndarray: The inverse of a shear matrix in row-major form.
    """
    return np.array([[1, -tan(angle_x), 0], [-tan(angle_y), 1, 0], [0, 0, 1.0]])


def mirror_matrix(about: LineType | PointType) -> NDArray:
    """
    Return a matrix to perform reflection about a line or a point.

    Args:
        about (LineType | PointType): A line or point about which the reflection is performed.

    Returns:
        np.ndarray: A matrix to perform reflection about a line or a point.

    Raises:
        RuntimeError: If about is not a line or a point.
    """
    if is_line(about):
        res = mirror_about_line_matrix(about)
    elif is_point(about):
        res = mirror_about_point_matrix(about)
    else:
        raise RuntimeError(f"{about} is invalid!")
    return res


def mirror_about_x_matrix() -> NDArray:
    """
    Return a matrix to perform reflection about the x-axis.

    Returns:
        np.ndarray: A matrix to perform reflection about the x-axis.
    """
    return np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])


def mirror_about_y_matrix() -> NDArray:
    """
    Return a matrix to perform reflection about the y-axis.

    Returns:
        np.ndarray: A matrix to perform reflection about the y-axis.
    """
    return np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])


def mirror_about_line_matrix(line: LineType) -> NDArray:
    """
    Return a matrix to perform reflection about a line.

    Args:
        line (LineType): The line about which the reflection is performed.

    Returns:
        np.ndarray: A matrix to perform reflection about a line.
    """
    p1, p2 = line
    x1, y1 = p1[:2]
    theta = atan2(p2[1] - p1[1], p2[0] - p1[0])
    two_theta = 2 * theta

    # translate the line to the origin
    # T = translation_matrix(-x1, -y1)
    # rotate about the origin by 2*theta
    # R = rot_about_origin_matrix(2*theta)
    # translate back
    # inv_t = translation_matrix(x1, y1)
    # return T @ R @ inv_t

    # We precompute the matrix
    c2 = cos(two_theta)
    s2 = sin(two_theta)
    return np.array(
        [
            [c2, s2, 0],
            [s2, -c2, 0],
            [-x1 * c2 + x1 - y1 * s2, -x1 * s2 + y1 * c2 + y1, 1.0],
        ]
    )


def mirror_about_origin_matrix() -> NDArray:
    """
    Return a matrix to perform reflection about the origin.

    Returns:
        np.ndarray: A matrix to perform reflection about the origin.
    """
    return np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])


def mirror_about_point_matrix(point: PointType) -> NDArray:
    """
    Return a matrix to perform reflection about a point.

    Args:
        point (PointType): The point about which the reflection is performed.

    Returns:
        np.ndarray: A matrix to perform reflection about a point.
    """
    x, y = point[:2]
    # T = translation_matrix(-x, -y)
    # M = mirror_about_origin_matrix()
    # inv_t = translation_matrix(x, y)
    # return T @ M @ inv_t
    # We precompute the matrix

    return np.array([[-1.0, 0, 0], [0, -1.0, 0], [2 * x, 2 * y, 1.0]])


def rotate(
    points: Sequence[PointType], angle: float, about: PointType = (0, 0)
) -> NDArray:
    """Rotate points by ``angle`` about a point.

    Args:
        points: Points to rotate (homogeneous or 2D; homogenized if needed).
        angle: Rotation angle in radians.
        about: Center of rotation. Defaults to ``(0, 0)``.

    Returns:
        np.ndarray: Homogeneous rotated points.

    Examples:
        >>> import simetri.graphics as sg
        >>> from math import pi
        >>> pts = sg.rotate([(1, 0)], pi / 2)
        >>> round(float(pts[0, 1]), 6)
        1.0
    """
    points = homogenize(points)
    return points @ rotation_matrix(angle, about)


def translate(points: Sequence[PointType], dx: float, dy: float) -> NDArray:
    """
    Translate points by dx, dy.

    Args:
        points (Sequence[PointType]): The points to translate.
        dx (float): The translation distance along the x-axis.
        dy (float): The translation distance along the y-axis.

    Returns:
        np.ndarray: The translated points.
    """
    return points @ translation_matrix(dx, dy)


def mirror(points: Sequence[PointType], about: LineType) -> NDArray:
    """
    Mirror points about a line.

    Args:
        points (Sequence[PointType]): The points to mirror.
        about (LineType): The line to mirror about.

    Returns:
        np.ndarray: The mirrored points.
    """
    return points @ mirror_matrix(about)


def glide(
    points: Sequence[PointType], mirror_line: LineType, distance: float
) -> NDArray:
    """
    Glide (mirror about a line then translate along the same line) points about a line.

    Args:
        points (Sequence[PointType]): The points to glide.
        mirror_line (LineType): The line to mirror about.
        distance (float): The distance to translate along the line.

    Returns:
        np.ndarray: The glided points.
    """
    return points @ glide_matrix(mirror_line, distance)


def shear(
    points: Sequence[PointType], angle_x: float, angle_y: float = 0
) -> NDArray:
    """
    Shear points by angle_x in x direction and angle_y in y direction.

    Args:
        points (Sequence[PointType]): The points to shear.
        angle_x (float): The angle of shear in x direction.
        angle_y (float, optional): The angle of shear in y direction, defaults to 0.

    Returns:
        np.ndarray: The sheared points.
    """
    return points @ shear_matrix(angle_x, angle_y)


def scale(
    points: Sequence[PointType], scale_x: float, scale_y: float
) -> NDArray:
    """
    Scale points by scale_x in x direction and scale_y in y direction.

    Args:
        points (Sequence[PointType]): The points to scale.
        scale_x (float): The scale factor in x direction.
        scale_y (float): The scale factor in y direction.

    Returns:
        np.ndarray: The scaled points.
    """
    return points @ scale_matrix(scale_x, scale_y)


def scale_in_place(
    points: Sequence[PointType],
    scale_x: float,
    scale_y: float,
    about: PointType,
) -> NDArray:
    """
    Scale points about a point by scale_x in x direction and scale_y in y direction.

    Args:
        points (Sequence[PointType]): The points to scale.
        scale_x (float): The scale factor in x direction.
        scale_y (float): The scale factor in y direction.
        about (PointType): The point about which the scaling is performed.

    Returns:
        np.ndarray: The scaled points.
    """
    return points @ scale_in_place_matrix(scale_x, scale_y, about)


def rotate_point_3D(
    point: PointType, line: LineType, angle: float
) -> PointType:
    """Rotate a 2D point out of the plane about a 2D line by ``angle``.

    Used for animating mirror reflections (folding a point around an axis).

    Args:
        point: Point to rotate.
        line: Axis line (two points) to rotate about.
        angle: Rotation angle in radians.

    Returns:
        PointType: Rotated point ``(x, y)`` in the plane projection.
    """

    p1, p2 = line
    line_angle_ = atan2(p2[1] - p1[1], p2[0] - p1[0])
    translation = translation_matrix(-p1[0], -p1[1])
    rotation = rotation_matrix(-line_angle_, (0, 0))
    xform = translation @ rotation
    x, y = point[:2]
    x, y, _ = [x, y, 1] @ xform

    y *= cos(angle)

    inv_translation = translation_matrix(p1[0], p1[1])
    inv_rotation = rotation_matrix(line_angle_, (0, 0))
    inv_xform = inv_rotation @ inv_translation
    x, y, _ = [x, y, 1] @ inv_xform

    return (x, y)


def rotate_line_3D(line: LineType, about: LineType, angle: float) -> LineType:
    """Rotate a 3d line about a 3d line by the given angle

    Args:
        line (LineType): Line to rotate.
        about (LineType): Line to rotate about.
        angle (float): Angle of rotation in radians.

    Returns:
        LineType: Rotated line.
    """
    p1 = rotate_point_3D(line[0], about, angle)
    p2 = rotate_point_3D(line[1], about, angle)

    return [p1, p2]


def rotate_point(
    point: PointType, angle: float, center: PointType = (0, 0)
) -> PointType:
    """Rotate a single 2D point about ``center`` by ``angle`` radians.

    Args:
        point: Point ``(x, y)`` to rotate.
        angle: Rotation angle in radians (counterclockwise).
        center: Center of rotation. Defaults to ``(0, 0)``.

    Returns:
        PointType: Rotated point as ``(x, y)``.

    Examples:
        >>> from math import pi
        >>> rotate_point((1, 0), pi / 2)
        (0.0, 1.0)
    """
    x, y = point[:2]
    cx, cy = center[:2]
    x -= cx
    y -= cy
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    x, y = x * cos_angle - y * sin_angle, x * sin_angle + y * cos_angle
    x += cx
    y += cy

    return (x, y)
