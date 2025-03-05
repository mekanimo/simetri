"""Transformation matrices."""

from math import cos, sin, tan
from typing import Sequence, Union

import numpy as np

from .common import Line, Point
from ..geometry.geometry import line_angle, vec_along_line, is_line, is_point


def identity_matrix() -> np.ndarray:
    """
    Return the identity matrix
    [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]].

    :return: The identity matrix.
    :rtype: np.ndarray
    """
    return np.identity(3)


def xform_matrix(
    a: float, b: float, c: float, d: float, e: float, f: float
) -> np.ndarray:
    """
    Return a transformation matrix in row form
    [[a, b, 0], [c, d, 0], [e, f, 1.0]].

    :param a: The a component of the transformation matrix.
    :type a: float
    :param b: The b component of the transformation matrix.
    :type b: float
    :param c: The c component of the transformation matrix.
    :type c: float
    :param d: The d component of the transformation matrix.
    :type d: float
    :param e: The e component of the transformation matrix.
    :type e: float
    :param f: The f component of the transformation matrix.
    :type f: float
    :return: The transformation matrix.
    :rtype: np.ndarray
    """
    return np.array([[a, b, 0], [c, d, 0], [e, f, 1.0]])


def translation_matrix(dx: float, dy: float) -> np.ndarray:
    """
    Return a translation matrix in row form
    [[1.0, 0, 0], [0, 1.0, 0], [dx, dy, 1.0]].

    :param dx: The translation distance along the x-axis.
    :type dx: float
    :param dy: The translation distance along the y-axis.
    :type dy: float
    :return: The translation matrix.
    :rtype: np.ndarray
    """
    return np.array([[1.0, 0, 0], [0, 1.0, 0], [dx, dy, 1.0]])


def inv_translation_matrix(dx: float, dy: float) -> np.ndarray:
    """
    Return the inverse of a translation matrix in row form
    [[1.0, 0, 0], [0, 1.0, 0], [-dx, -dy, 1.0]].

    :param dx: The translation distance along the x-axis.
    :type dx: float
    :param dy: The translation distance along the y-axis.
    :type dy: float
    :return: The inverse translation matrix.
    :rtype: np.ndarray
    """
    return np.array([[1.0, 0, 0], [0, 1.0, 0], [-dx, -dy, 1.0]])


def rot_about_origin_matrix(theta: float) -> np.ndarray:
    """
    Return a rotation matrix in row form
    [[cos(theta), sin(theta), 0], [-sin(theta), cos(theta), 0], [0, 0, 1.0]].

    :param theta: The rotation angle in radians.
    :type theta: float
    :return: The rotation matrix.
    :rtype: np.ndarray
    """
    c = cos(theta)
    s = sin(theta)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])


def rotation_matrix(theta: float, about=(0, 0)) -> np.ndarray:
    """
    Construct a rotation matrix that can be used to rotate a point
    about another point by theta float.
    Return a rotation matrix in row form
    dx, dy = about
    [[cos(theta), sin(theta), 0],
    [-sin(theta), cos(theta), 0],
    cos(theta)dx-sin(theta)dy+x, cos(theta)dy+sin(theta)dx+y, 1]].

    :param theta: The rotation angle in radians.
    :type theta: float
    :param about: The point to rotate about, defaults to (0, 0).
    :type about: tuple, optional
    :return: The rotation matrix.
    :rtype: np.ndarray
    """
    dx, dy = about[:2]
    # translate 'about' to the origin
    trans_mat = translation_matrix(-dx, -dy)
    # rotate around the origin
    rot_mat = rot_about_origin_matrix(theta)
    # translate it back to initial pos
    inv_trans_mat = translation_matrix(dx, dy)
    # compose the transformation matrix
    return trans_mat @ rot_mat @ inv_trans_mat


def inv_rotation_matrix(theta: float, about=(0, 0)) -> np.ndarray:
    """
    Construct the inverse of a rotation matrix that can be used to rotate a point
    about another point by theta float.
    Return a rotation matrix in row form
    dx, dy = about
    [[cos(theta), -sin(theta), 0],
    [sin(theta), cos(theta), 0],
    -cos(theta)dx-sin(theta)dy+x, -sin(theta)dx+cos(theta)dy+y, 1]].

    :param theta: The rotation angle in radians.
    :type theta: float
    :param about: The point to rotate about, defaults to (0, 0).
    :type about: tuple, optional
    :return: The inverse rotation matrix.
    :rtype: np.ndarray
    """
    dx, dy = about[:2]
    # translate 'about' to the origin
    trans_mat = translation_matrix(-dx, -dy)
    # rotate around the origin
    rot_mat = rot_about_origin_matrix(theta)
    # translate it back to initial pos
    inv_trans_mat = translation_matrix(dx, dy)
    # compose the transformation matrix
    return inv_trans_mat @ rot_mat.T @ trans_mat


def glide_matrix(mirror_line: Line, distance: float) -> np.ndarray:
    """
    Return a glide-reflection matrix in row form.
    Reflect about the given vector then translate by dx
    along the same vector.

    :param mirror_line: The line to mirror about.
    :type mirror_line: Line
    :param distance: The distance to translate along the line.
    :type distance: float
    :return: The glide-reflection matrix.
    :rtype: np.ndarray
    """
    mirror_mat = mirror_about_line_matrix(mirror_line)
    x, y = vec_along_line(mirror_line, distance)[:2]
    trans_mat = translation_matrix(x, y)

    return mirror_mat @ trans_mat


def inv_glide_matrix(mirror_line: Line, distance: float) -> np.ndarray:
    """
    Return the inverse of a glide-reflection matrix in row form.
    Reflect about the given vector then translate by dx
    along the same vector.

    :param mirror_line: The line to mirror about.
    :type mirror_line: Line
    :param distance: The distance to translate along the line.
    :type distance: float
    :return: The inverse glide-reflection matrix.
    :rtype: np.ndarray
    """
    mirror_mat = mirror_about_line_matrix(mirror_line)
    x, y = vec_along_line(mirror_line, distance)[:2]
    trans_matrix = translation_matrix(x, y)

    return trans_matrix @ mirror_mat


def scale_matrix(scale_x: float, scale_y: float = None) -> np.ndarray:
    """
    Return a scale matrix in row form.

    :param scale_x: Scale factor in x direction.
    :type scale_x: float
    :param scale_y: Scale factor in y direction, defaults to None.
    :type scale_y: float, optional
    :return: A scale matrix in row form.
    :rtype: np.ndarray
    """
    if scale_y is None:
        scale_y = scale_x
    return np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1.0]])


def inv_scale_matrix(scale_x: float, scale_y: float = None) -> np.ndarray:
    """
    Return the inverse of a scale matrix in row form.

    :param scale_x: Scale factor in x direction.
    :type scale_x: float
    :param scale_y: Scale factor in y direction, defaults to None.
    :type scale_y: float, optional
    :return: The inverse of a scale matrix in row form.
    :rtype: np.ndarray
    """
    if scale_y is None:
        scale_y = scale_x
    return np.array([[1 / scale_x, 0, 0], [0, 1 / scale_y, 0], [0, 0, 1.0]])


def scale_in_place_matrix(scale_x: float, scale_y: float, point: Point) -> np.ndarray:
    """
    Return a scale matrix in row form that scales about a point.

    :param scale_x: Scale factor in x direction.
    :type scale_x: float
    :param scale_y: Scale factor in y direction.
    :type scale_y: float
    :param point: Point about which the scaling is performed.
    :type point: Point
    :return: A scale matrix in row form that scales about a point.
    :rtype: np.ndarray
    """
    dx, dy = point[:2]
    trans_mat = translation_matrix(-dx, -dy)
    scale_mat = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1.0]])
    inv_trans_mat = translation_matrix(dx, dy)
    return trans_mat @ scale_mat @ inv_trans_mat


def shear_matrix(theta_x: float, theta_y: float = 0) -> np.ndarray:
    """
    Return a shear matrix in row form.

    :param theta_x: Angle of shear in x direction.
    :type theta_x: float
    :param theta_y: Angle of shear in y direction, defaults to 0.
    :type theta_y: float, optional
    :return: A shear matrix in row form.
    :rtype: np.ndarray
    """
    return np.array([[1, tan(theta_y), 0], [tan(theta_x), 1, 0], [0, 0, 1.0]])


def inv_shear_matrix(theta_x: float, theta_y: float = 0) -> np.ndarray:
    """
    Return the inverse of a shear matrix in row form.

    :param theta_x: Angle of shear in x direction.
    :type theta_x: float
    :param theta_y: Angle of shear in y direction, defaults to 0.
    :type theta_y: float, optional
    :return: The inverse of a shear matrix in row form.
    :rtype: np.ndarray
    """
    return np.array([[1, -tan(theta_x), 0], [-tan(theta_y), 1, 0], [0, 0, 1.0]])


def mirror_matrix(about: Union[Line, Point]) -> np.ndarray:
    """
    Return a matrix to perform reflection about a line or a point.

    :param about: A line or point about which the reflection is performed.
    :type about: Union[Line, Point]
    :raises RuntimeError: If about is not a line or a point.
    :return: A matrix to perform reflection about a line or a point.
    :rtype: np.ndarray
    """
    if is_line(about):
        res = mirror_about_line_matrix(about)
    elif is_point(about):
        res = mirror_about_point_matrix(about)
    else:
        raise RuntimeError(f"{about} is invalid!")
    return res


def mirror_about_x_matrix() -> np.ndarray:
    """
    Return a matrix to perform reflection about the x-axis.

    :return: A matrix to perform reflection about the x-axis.
    :rtype: np.ndarray
    """
    return np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])


def mirror_about_y_matrix() -> np.ndarray:
    """
    Return a matrix to perform reflection about the y-axis.

    :return: A matrix to perform reflection about the y-axis.
    :rtype: np.ndarray
    """
    return np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])


def mirror_about_line_matrix(line: Line) -> np.ndarray:
    """
    Return a matrix to perform reflection about a line.

    :param line: The line about which the reflection is performed.
    :type line: Line
    :return: A matrix to perform reflection about a line.
    :rtype: np.ndarray
    """
    p1, p2 = line
    x1, y1 = p1[:2]
    theta = line_angle(p1, p2)
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


def mirror_about_origin_matrix() -> np.ndarray:
    """
    Return a matrix to perform reflection about the origin.

    :return: A matrix to perform reflection about the origin.
    :rtype: np.ndarray
    """
    return np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])


def mirror_about_point_matrix(point: Point) -> np.ndarray:
    """
    Return a matrix to perform reflection about a point.

    :param point: The point about which the reflection is performed.
    :type point: Point
    :return: A matrix to perform reflection about a point.
    :rtype: np.ndarray
    """
    x, y = point[:2]
    # T = translation_matrix(-x, -y)
    # M = mirror_about_origin_matrix()
    # inv_t = translation_matrix(x, y)
    # return T @ M @ inv_t
    # We precompute the matrix

    return np.array([[-1.0, 0, 0], [0, -1.0, 0], [2 * x, 2 * y, 1.0]])


def rotate(points: Sequence[Point], theta: float, about: Point = (0, 0)) -> np.ndarray:
    """
    Rotate points by theta about a point.

    :param points: The points to rotate.
    :type points: Sequence[Point]
    :param theta: The angle to rotate by.
    :type theta: float
    :param about: The point to rotate about, defaults to (0, 0).
    :type about: Point, optional
    :return: The rotated points.
    :rtype: np.ndarray
    """
    return points @ rotation_matrix(theta, about)


def translate(points: Sequence[Point], dx: float, dy: float) -> np.ndarray:
    """
    Translate points by dx, dy.

    :param points: The points to translate.
    :type points: Sequence[Point]
    :param dx: The translation distance along the x-axis.
    :type dx: float
    :param dy: The translation distance along the y-axis.
    :type dy: float
    :return: The translated points.
    :rtype: np.ndarray
    """
    return points @ translation_matrix(dx, dy)


def mirror(points: Sequence[Point], about: Line) -> np.ndarray:
    """
    Mirror points about a line.

    :param points: The points to mirror.
    :type points: Sequence[Point]
    :param about: The line to mirror about.
    :type about: Line
    :return: The mirrored points.
    :rtype: np.ndarray
    """
    return points @ mirror_matrix(about)


def glide(points: Sequence[Point], mirror_line: Line, distance: float) -> np.ndarray:
    """
    Glide (mirror about a line then translate along the same line) points about a line.

    :param points: The points to glide.
    :type points: Sequence[Point]
    :param mirror_line: The line to mirror about.
    :type mirror_line: Line
    :param distance: The distance to translate along the line.
    :type distance: float
    :return: The glided points.
    :rtype: np.ndarray
    """
    return points @ glide_matrix(mirror_line, distance)


def shear(points: Sequence[Point], theta_x: float, theta_y: float = 0) -> np.ndarray:
    """
    Shear points by theta_x in x direction and theta_y in y direction.

    :param points: The points to shear.
    :type points: Sequence[Point]
    :param theta_x: The angle of shear in x direction.
    :type theta_x: float
    :param theta_y: The angle of shear in y direction, defaults to 0.
    :type theta_y: float, optional
    :return: The sheared points.
    :rtype: np.ndarray
    """
    return points @ shear_matrix(theta_x, theta_y)


def scale(points: Sequence[Point], scale_x: float, scale_y: float) -> np.ndarray:
    """
    Scale points by scale_x in x direction and scale_y in y direction.

    :param points: The points to scale.
    :type points: Sequence[Point]
    :param scale_x: The scale factor in x direction.
    :type scale_x: float
    :param scale_y: The scale factor in y direction.
    :type scale_y: float
    :return: The scaled points.
    :rtype: np.ndarray
    """
    return points @ scale_matrix(scale_x, scale_y)


def scale_in_place(
    points: Sequence[Point], scale_x: float, scale_y: float, point: Point
) -> np.ndarray:
    """
    Scale points about a point by scale_x in x direction and scale_y in y direction.

    :param points: The points to scale.
    :type points: Sequence[Point]
    :param scale_x: The scale factor in x direction.
    :type scale_x: float
    :param scale_y: The scale factor in y direction.
    :type scale_y: float
    :param point: The point about which the scaling is performed.
    :type point: Point
    :return: The scaled points.
    :rtype: np.ndarray
    """
    return points @ scale_in_place_matrix(scale_x, scale_y, point)
