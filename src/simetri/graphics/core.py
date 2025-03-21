"""Base class. This is the parent for Shape and Batch classes."""

__all__ = [
    "Base",
]

from typing import Sequence, Any, Union
from typing_extensions import Self

import numpy as np
from numpy import ndarray

from .all_enums import Anchor, Side, get_enum_value, anchors
from .common import (
    Point,
    Line,
)
from .affine import (
    translation_matrix,
    rotation_matrix,
    mirror_matrix,
    glide_matrix,
    scale_in_place_matrix,
    shear_matrix,
)
from ..geometry.geometry import line_angle


class Base:
    """Base class for Shape and Batch objects."""

    def __getattr__(self, name: str) -> Any:
        if name in anchors:
            res = getattr(self.b_box, name)
        else:
            try:
                res = self.__dict__[name]
            except KeyError as exc:
                try:
                    res = super().__getattr__(name)
                except AttributeError as exc:
                    msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
                    raise AttributeError(msg) from exc

        return res


    def translate(self, dx: float = 0, dy: float = 0, reps: int = 0) -> Self:
        """
        Translates the object by dx and dy.

        Args:
            dx (float): The translation distance along the x-axis.
            dy (float): The translation distance along the y-axis.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The transformed object.
        """
        transform = translation_matrix(dx, dy)
        return self._update(transform, reps=reps)

    def translate_along(
        self,
        path: Sequence[Point],
        step: int = 1,
        align_tangent: bool = False,
        scale: float = 1,  # scale factor
        rotate: float = 0,  # angle in radians
    ) -> Self:
        """
        Translates the object along the given curve.
        Every n-th point is used to calculate the translation vector.
        If align_tangent is True, the object is rotated to align with the tangent at each point.
        scale is the scale factor applied at each point.
        rotate is the angle in radians applied at each point.

        Args:
            path (Sequence[Point]): The path to translate along.
            step (int, optional): The step size. Defaults to 1.
            align_tangent (bool, optional): Whether to align the object with the tangent. Defaults to False.
            scale (float, optional): The scale factor. Defaults to 1.
            rotate (float, optional): The rotation angle in radians. Defaults to 0.

        Returns:
            Self: The transformed object.
        """
        x, y = path[0][:2]
        self.move_to((x, y))
        dup = self.copy()
        if align_tangent:
            tangent = line_angle(path[-1], path[0])
            self.rotate(tangent, about=path[0], reps=0)
        dup2 = dup.copy()
        for i, point in enumerate(path[1::step]):
            dup2 = dup2.copy()
            px, py = point[:2]
            dup2.move_to((px, py))
            if scale != 1:
                dup2.scale(scale, about=point)
            if rotate != 0:
                dup2.rotate(rotate, about=point)
            self.append(dup2)
            if align_tangent:
                tangent = line_angle(path[i - 1], path[i])
                dup2.rotate(tangent, about=point, reps=0)
            # scale *= scale
            # rotate += rotate
        return self

    def rotate(self, angle: float, about: Point = (0, 0), reps: int = 0) -> Self:
        """
        Rotates the object by the given angle (in radians) about the given point.

        Args:
            angle (float): The rotation angle in radians.
            about (Point, optional): The point to rotate about. Defaults to (0, 0).
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The rotated object.
        """
        transform = rotation_matrix(angle, about)
        return self._update(transform, reps=reps)

    def mirror(self, about: Union[Line, Point], reps: int = 0) -> Self:
        """
        Mirrors the object about the given line or point.

        Args:
            about (Union[Line, Point]): The line or point to mirror about.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The mirrored object.
        """
        transform = mirror_matrix(about)
        return self._update(transform, reps=reps)

    def glide(self, glide_line: Line, glide_dist: float, reps: int = 0) -> Self:
        """
        Glides (first mirror then translate) the object along the given line
        by the given glide_dist.

        Args:
            glide_line (Line): The line to glide along.
            glide_dist (float): The distance to glide.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The glided object.
        """
        transform = glide_matrix(glide_line, glide_dist)
        return self._update(transform, reps=reps)

    def scale(
        self,
        scale_x: float,
        scale_y: Union[float, None] = None,
        about: Point = (0, 0),
        reps: int = 0,
    ) -> Self:
        """
        Scales the object by the given scale factors about the given point.

        Args:
            scale_x (float): The scale factor in the x direction.
            scale_y (float, optional): The scale factor in the y direction. Defaults to None.
            about (Point, optional): The point to scale about. Defaults to (0, 0).
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The scaled object.
        """
        if scale_y is None:
            scale_y = scale_x
        transform = scale_in_place_matrix(scale_x, scale_y, about)
        return self._update(transform, reps=reps)

    def shear(self, theta_x: float, theta_y: float, reps: int = 0) -> Self:
        """
        Shears the object by the given angles.

        Args:
            theta_x (float): The shear angle in the x direction.
            theta_y (float): The shear angle in the y direction.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The sheared object.
        """
        transform = shear_matrix(theta_x, theta_y)
        return self._update(transform, reps=reps)

    def reset_xform_matrix(self) -> Self:
        """
        Resets the transformation matrix to the identity matrix.

        Returns:
            Self: The object with the reset transformation matrix.
        """
        self.__dict__["xform_matrix"] = np.identity(3)
        return self

    def transform(self, xform_matrix: ndarray, reps: int = 0) -> Self:
        """
        Transforms the object by the given transformation matrix.

        Args:
            xform_matrix (ndarray): The transformation matrix.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The transformed object.
        """
        return self._update(xform_matrix, reps=reps)

    def move_to(self, pos: Point, anchor: Anchor = Anchor.CENTER) -> Self:
        """
        Moves the object to the given position by using its center point.

        Args:
            pos (Point): The position to move to.
            anchor (Anchor, optional): The anchor point. Defaults to Anchor.CENTER.

        Returns:
            Self: The moved object.
        """
        x, y = pos[:2]
        anchor = get_enum_value(Anchor, anchor)
        x1, y1 = getattr(self.b_box, anchor)
        transform = translation_matrix(x - x1, y - y1)
        return self._update(transform, reps=0)

    def offset_line(self, side: Side, offset: float) -> Line:
        """
        Offset the line by the given side and offset distance.
        side can be Side.LEFT, Side.RIGHT, Side.TOP, or Side.BOTTOM.
        offset is applied outwards.

        Args:
            side (Side): The side to offset.
            offset (float): The offset distance.

        Returns:
            Line: The offset line.
        """
        side = get_enum_value(Side, side)
        return self.b_box.offset_line(side, offset)

    def offset_point(self, anchor: Anchor, dx: float, dy: float = 0) -> Point:
        """
        Offset the point by the given anchor and offset distances.
        anchor can be Anchor.CENTER, Anchor.SOUTHWEST, Anchor.SOUTHEAST,
        Anchor.NORTHWEST, Anchor.NORTHEAST, Anchor.SOUTH, Anchor.WEST,
        Anchor.EAST, or Anchor.NORTH.

        Args:
            anchor (Anchor): The anchor point.
            dx (float): The x offset.
            dy (float, optional): The y offset. Defaults to 0.

        Returns:
            Point: The offset point.
        """
        anchor = get_enum_value(Anchor, anchor)
        return self.b_box.offset_point(anchor, dx, dy)


class StyleMixin:
    """Mixin class for style attributes.
    Shape class inherits from this.
    Some Batch classes with different subtypes also inherit from this.
    """
    def __setattr__(self, name, value):
        """Set an attribute of the shape.

        Args:
            name (str): The name of the attribute.
            value (Any): The value to set.
        """
        obj, attrib = self.__dict__["_aliasses"].get(name, (None, None))
        if obj:
            setattr(obj, attrib, value)
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        """Retrieve an attribute of the shape.

        Args:
            name (str): The attribute name to return.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute cannot be found.
        """
        obj, attrib = self.__dict__["_aliasses"].get(name, (None, None))
        if obj:
            res = getattr(obj, attrib)
        else:
            res = self.__dict__[name]
        return res

    def _set_aliases(self):
        """Set aliases for style attributes based on the style map."""
        _aliasses = {}
        for alias, path_attrib in self._style_map.items():
            style_path, attrib = path_attrib
            obj = self
            for attrib_name in style_path.split("."):
                obj = obj.__dict__[attrib_name]
            _aliasses[alias] = (obj, attrib)
        self.__dict__["_aliasses"] = _aliasses
