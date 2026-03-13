"""Base class. This is the parent for Shape and Batch classes."""

__all__ = ["Base", "StyleMixin"]

from typing import Sequence, Any, Union, List
from typing_extensions import Self
import operator
from math import hypot

import numpy as np
from numpy import ndarray

from .all_enums import (
    Anchor,
    Side,
    InPlace,
    get_enum_value,
    anchors,
    Reference,
    TransformationType,
    Types,
    point_refs,
    line_refs,
    length_refs,
)
from .common import (
    PointType,
    LineType,
)
from .affine import (
    translation_matrix,
    rotation_matrix,
    mirror_matrix,
    glide_matrix,
    scale_in_place_matrix,
    shear_matrix,
)
from ..geometry.geometry import line_angle, angled_line, offset_line

from ..canvas.style_map import shape_args

STYLE_ATTRIBUTES = set(shape_args)

def _update_inplace(
    xform_matrix: "ndarray",
    xform_type: TransformationType,
    incr: Union[
        float | tuple[float, float] | tuple[callable, Any] | tuple[InPlace, Any]
    ]=None,

):
    """Update a transformation matrix in-place for repeated transformations.

    Supported ``incr`` forms:
    - ``float``: additive increment for rotation/glide
    - ``(x, y)``: additive increment for translate/scale/shear
    - ``(callable, arg)``: callable returns one of the above increment values
    - ``(InPlace.OP, value)``: applies operation to current transform parameter(s)
    """

    def _is_number(value: Any) -> bool:
        return isinstance(value, (int, float, np.integer, np.floating))

    def _coerce_scalar(value: Any) -> float:
        if not _is_number(value):
            raise TypeError("Expected a numeric increment value")
        return float(value)

    def _coerce_pair(value: Any) -> tuple[float, float]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise ValueError("Expected a 2-item increment sequence")
            x, y = value
            if not (_is_number(x) and _is_number(y)):
                raise TypeError("Expected numeric increment values")
            return float(x), float(y)

        scalar = _coerce_scalar(value)
        return scalar, scalar

    def _set_rotation(angle: float) -> None:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xform_matrix[0, 0] = cos_a
        xform_matrix[0, 1] = sin_a
        xform_matrix[1, 0] = -sin_a
        xform_matrix[1, 1] = cos_a

    def _add_increment(value: Any) -> None:
        if xform_type == TransformationType.TRANSLATE:
            incr_x, incr_y = _coerce_pair(value)
            xform_matrix[2, 0] += incr_x
            xform_matrix[2, 1] += incr_y
        elif xform_type == TransformationType.ROTATE:
            angle = np.arctan2(xform_matrix[0, 1], xform_matrix[0, 0])
            angle += _coerce_scalar(value)
            _set_rotation(angle)
        elif xform_type == TransformationType.SCALE:
            incr_x, incr_y = _coerce_pair(value)
            xform_matrix[0, 0] += incr_x
            xform_matrix[1, 1] += incr_y
        elif xform_type == TransformationType.SHEAR:
            incr_x, incr_y = _coerce_pair(value)
            theta_x = np.arctan(xform_matrix[1, 0])
            theta_y = np.arctan(xform_matrix[0, 1])
            xform_matrix[1, 0] = np.tan(theta_x + incr_x)
            xform_matrix[0, 1] = np.tan(theta_y + incr_y)
        elif xform_type == TransformationType.GLIDE:
            dx, dy = xform_matrix[2, :2]
            dist = hypot(dx, dy)
            if dist == 0:
                return
            new_dist = dist + _coerce_scalar(value)
            scale = new_dist / dist
            xform_matrix[2, 0] = dx * scale
            xform_matrix[2, 1] = dy * scale

    def _apply_operator(ip_op: InPlace, value: Any) -> None:
        operators = {
            InPlace.ADD: operator.iadd,
            InPlace.SUB: operator.isub,
            InPlace.MUL: operator.imul,
            InPlace.TRUE_DIV: operator.itruediv,
            InPlace.FLOOR_DIV: operator.ifloordiv,
            InPlace.MOD: operator.imod,
            InPlace.POW: operator.ipow,
        }
        oper = operators[ip_op]

        if xform_type == TransformationType.TRANSLATE:
            val_x, val_y = _coerce_pair(value)
            xform_matrix[2, 0] = oper(xform_matrix[2, 0], val_x)
            xform_matrix[2, 1] = oper(xform_matrix[2, 1], val_y)
        elif xform_type == TransformationType.ROTATE:
            angle = np.arctan2(xform_matrix[0, 1], xform_matrix[0, 0])
            new_angle = oper(angle, _coerce_scalar(value))
            _set_rotation(new_angle)
        elif xform_type == TransformationType.SCALE:
            val_x, val_y = _coerce_pair(value)
            xform_matrix[0, 0] = oper(xform_matrix[0, 0], val_x)
            xform_matrix[1, 1] = oper(xform_matrix[1, 1], val_y)
        elif xform_type == TransformationType.SHEAR:
            val_x, val_y = _coerce_pair(value)
            theta_x = np.arctan(xform_matrix[1, 0])
            theta_y = np.arctan(xform_matrix[0, 1])
            xform_matrix[1, 0] = np.tan(oper(theta_x, val_x))
            xform_matrix[0, 1] = np.tan(oper(theta_y, val_y))
        elif xform_type == TransformationType.GLIDE:
            dx, dy = xform_matrix[2, :2]
            dist = hypot(dx, dy)
            if dist == 0:
                return
            new_dist = oper(dist, _coerce_scalar(value))
            scale = new_dist / dist
            xform_matrix[2, 0] = dx * scale
            xform_matrix[2, 1] = dy * scale


    if _is_number(incr):
        _add_increment(incr)
    elif isinstance(incr, Sequence) and not isinstance(incr, (str, bytes)):
        if len(incr) == 2 and _is_number(incr[0]) and _is_number(incr[1]):
            _add_increment(incr)
        elif len(incr) == 2 and callable(incr[0]):
            _add_increment(incr[0](incr[1]))
        elif len(incr) == 2:
            ip_op, value = incr
            ip_op = get_enum_value(InPlace, ip_op)
            _apply_operator(ip_op, value)

    return xform_matrix


def _resolve_reference(target, reference):
    if isinstance(reference, [tuple, List]):
        ref, value = reference
        if isinstance(value, Reference):
            value = getattr(target, f"{reference}")
        if ref in point_refs:
            if isinstance(value, [float, int]):
                # Angled line
                res = angled_line(getattr(target, f"{ref}"), value)
            elif isinstance(value, [list, tuple]):
                # Offset point
                x, y = getattr(target, f"{ref}")
                dx, dy = value[:2]
                res = [x + dx, y + dy]
        elif ref in line_refs:
            # Offset line
            offset_line(
                getattr(target, f"{ref}"),
            )
    else:
        # A reference point, line, or length
        res = getattr(target, f"{reference}")

    return res


class Base:
    """Base class for Shape and Batch objects."""

    def __getattr__(self, name: str) -> Any:
        if name in anchors:
            if name.startswith("bbox_"):
                name = name[4:]
            res = getattr(self.b_box, name)
        else:
            try:
                res = self.__dict__[name]
            except KeyError as exc:
                try:
                    res = getattr(super(), name)
                except AttributeError as attr_exc:
                    # For style attributes, return None instead of raising an error
                    # This allows the canvas property resolution to work properly
                    if name in STYLE_ATTRIBUTES:
                        return None
                    msg = (
                        f"'{self.__class__.__name__}' object has no attribute '{name}'"
                    )
                    raise AttributeError(msg) from attr_exc

        return res

    def translate(
        self,
        dx: float = 0,
        dy: float = 0,
        take: slice = None,
        reps: int = 0,
        incr: Union[
            float,
            tuple[float, float],
            tuple[callable, Any],
            tuple[InPlace, Any],
            None,
        ] = None,
        merge: bool = False,
    ) -> Self:
        """
        Translates the object by dx and dy.

        Args:
            dx (float): The translation distance along the x-axis.
            dy (float): The translation distance along the y-axis.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The transformed object.
        """
        if self.active:
            transform = translation_matrix(dx, dy)
            if self.type == Types.SHAPE:
                res = self._update(
                    transform,
                    reps=reps,
                    incr=incr,
                    merge=merge,
                    xform_type=TransformationType.TRANSLATE,
                )
            else:
                res = self._update(
                    transform,
                    reps=reps,
                    take=take,
                    incr=incr,
                    merge=merge,
                    xform_type=TransformationType.TRANSLATE,
                )
        else:
            res = self.copy()

        return res

    def translate_along(
        self,
        path: Sequence[PointType],
        step: int = 1,
        align_tangent: bool = False,
        scale: float = 1,  # scale factor
        rotate: float = 0,  # angle in radians
        incr: Union[
            float,
            tuple[float, float],
            tuple[callable, Any],
            tuple[InPlace, Any],
            None,
        ] = None,
        merge: bool = False,
    ) -> Self:
        # This is not up to date anymore!!!
        """
        Translates the object along the given curve.
        Every n-th point is used to calculate the translation vector.
        If align_tangent is True, the object is rotated to align with the tangent at each point.
        scale is the scale factor applied at each point.
        rotate is the angle in radians applied at each point.

        Args:
            path (Sequence[PointType]): The path to translate along.
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

    def rotate(
        self,
        angle: float,
        about: PointType = (0, 0),
        reps: int = 0,
        take: slice = None,
        incr: Union[
            float,
            tuple[float, float],
            tuple[callable, Any],
            tuple[InPlace, Any],
            None,
        ] = None,
        merge: bool = False,
    ) -> Self:
        """
        Rotates the object by the given angle (in radians) about the given point.

        Args:
            angle (float): The rotation angle in radians.
            about (PointType, optional): The point to rotate about. Defaults to (0, 0).
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The rotated object.
        """
        if self.active:
            transform = rotation_matrix(angle, about)
            if self.__class__.__name__ == "Shape":
                res = self._update(
                    transform,
                    reps=reps,
                    incr=incr,
                    merge=merge,
                    xform_type=TransformationType.ROTATE,
                )

            else:
                res = self._update(
                    transform,
                    reps=reps,
                    take=take,
                    incr=incr,
                    merge=merge,
                    xform_type=TransformationType.ROTATE,
                )
        else:
            res = self.copy()
        return res

    def mirror(
        self,
        about: Union[LineType, PointType],
        reps: int = 0,
        take: slice = None,
        incr: Union[
            float,
            tuple[float, float],
            tuple[callable, Any],
            tuple[InPlace, Any],
            None,
        ] = None,
        merge: bool = False,
    ) -> Self:
        """
        Mirrors the object about the given line or point.

        Args:
            about (Union[LineType, PointType]): The line or point to mirror about.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The mirrored object.
        """
        if self.active:
            transform = mirror_matrix(about)
            if self.__class__.__name__ == "Shape":
                res = self._update(
                    transform,
                    reps=reps,
                    incr=incr,
                    merge=merge,
                    xform_type=TransformationType.MIRROR,
                )
            else:
                res = self._update(
                    transform,
                    reps=reps,
                    take=take,
                    incr=incr,
                    merge=merge,
                    xform_type=TransformationType.MIRROR,
                )
        else:
            res = self.copy()

        return res

    def glide(
        self,
        glide_line: LineType,
        glide_dist: float,
        reps: int = 0,
        take: slice = None,
        incr: Union[
            float,
            tuple[float, float],
            tuple[callable, Any],
            tuple[InPlace, Any],
            None,
        ] = None,
        merge: bool = False,
    ) -> Self:
        """
        Glides (first mirror then translate) the object along the given line
        by the given glide_dist.

        Args:
            glide_line (LineType): The line to glide along.
            glide_dist (float): The distance to glide.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The glided object.
        """
        if self.active:
            transform = glide_matrix(glide_line, glide_dist)
            if self.__class__.__name__ == "Shape":
                res = self._update(
                    transform,
                    reps=reps,
                    incr=incr,
                    merge=merge,
                    xform_type=TransformationType.GLIDE,
                )
            else:
                res = self._update(
                    transform,
                    reps=reps,
                    take=take,
                    incr=incr,
                    merge=merge,
                    xform_type=TransformationType.GLIDE,
                )
        else:
            res = self.copy()

        return res

    def scale(
        self,
        scale_x: float,
        scale_y: Union[float, None] = None,
        about: PointType = (0, 0),
        reps: int = 0,
        take: slice = None,
        incr: Union[
            float,
            tuple[float, float],
            tuple[callable, Any],
            tuple[InPlace, Any],
            None,
        ] = None,
        merge: bool = False,
    ) -> Self:
        """
        Scales the object by the given scale factors about the given point.

        Args:
            scale_x (float): The scale factor in the x direction.
            scale_y (float, optional): The scale factor in the y direction. Defaults to None.
            about (PointType, optional): The point to scale about. Defaults to (0, 0).
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The scaled object.
        """
        if scale_y is None:
            scale_y = scale_x
        if self.active:
            transform = scale_in_place_matrix(scale_x, scale_y, about)
            res = self._update(
                    transform,
                    reps=reps,
                    take=take,
                    incr=incr,
                    merge=merge,
                    xform_type=TransformationType.SCALE,
                )
        else:
            res = self.copy()

        return res

    def shear(
        self,
        theta_x: float,
        theta_y: float,
        reps: int = 0,
        take: slice = None,
        incr: Union[
            float,
            tuple[float, float],
            tuple[callable, Any],
            tuple[InPlace, Any],
            None,
        ] = None,
        merge: bool = False,
    ) -> Self:
        """
        Shears the object by the given angles.

        Args:
            theta_x (float): The shear angle in the x direction.
            theta_y (float): The shear angle in the y direction.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The sheared object.
        """
        if self.active:
            transform = shear_matrix(theta_x, theta_y)
            if self.__class__.__name__ == "Shape":
                res = self._update(
                    transform,
                    reps=reps,
                    incr=incr,
                    merge=merge,
                    xform_type=TransformationType.SHEAR,
                )
            else:
                res = self._update(
                    transform,
                    reps=reps,
                    take=take,
                    incr=incr,
                    merge=merge,
                    xform_type=TransformationType.SHEAR,
                )
        else:
            res = self.copy()

        return res

    def reset_xform_matrix(self) -> Self:
        """
        Resets the transformation matrix to the identity matrix.

        Returns:
            Self: The object with the reset transformation matrix.
        """
        self.__dict__["xform_matrix"] = np.identity(3)
        return self

    def transform(
        self,
        transform_matrix: ndarray,
        reps: int = 0,
        take: slice = None,
        merge: bool = False,
    ) -> Self:
        """
        Transforms the object by the given transformation matrix.

        Args:
            transform_matrix (ndarray): The transformation matrix.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The transformed object.
        """
        if self.active:
            if self.__class__.__name__ == "Shape":
                res = self._update(
                    transform_matrix,
                    reps=reps,
                    merge=merge,
                    xform_type=TransformationType.TRANSFORM,
                )
            else:
                res = self._update(
                    transform_matrix,
                    reps=reps,
                    take=take,
                    merge=merge,
                    xform_type=TransformationType.TRANSFORM,
                )
        else:
            res = self.copy()

        return res

    def move_to(self, pos: PointType, anchor: Anchor = Anchor.CENTER) -> Self:
        """
        Moves the object to the given position by using its center point.

        Args:
            pos (PointType): The position to move to.
            anchor (Anchor, optional): The anchor point. Defaults to Anchor.CENTER.

        Returns:
            Self: The moved object.
        """
        if self.active:
            x, y = pos[:2]
            anchor = get_enum_value(Anchor, anchor)
            x1, y1 = getattr(self.b_box, anchor)
            transform = translation_matrix(x - x1, y - y1)
            res = self._update(transform, reps=0)
        else:
            res = self.copy()

        return res

    def offset_line(self, side: Side, offset: float) -> LineType:
        """
        Offset the line by the given side and offset distance.
        side can be Side.LEFT, Side.RIGHT, Side.TOP, or Side.BOTTOM.
        offset is applied outwards.

        Args:
            side (Side): The side to offset.
            offset (float): The offset distance.

        Returns:
            LineType: The offset line.
        """
        side = get_enum_value(Side, side)
        return self.b_box.offset_line(side, offset)

    def offset_point(self, anchor: Anchor, dx: float, dy: float = 0) -> PointType:
        """
        Offset the point by the given anchor and offset distances.
        anchor can be Anchor.MIDPOINT, Anchor.SOUTHWEST, Anchor.SOUTHEAST,
        Anchor.NORTHWEST, Anchor.NORTHEAST, Anchor.SOUTH, Anchor.WEST,
        Anchor.EAST, or Anchor.NORTH.

        Args:
            anchor (Anchor): The anchor point.
            dx (float): The x offset.
            dy (float, optional): The y offset. Defaults to 0.

        Returns:
            PointType: The offset point.
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
        # Handle case where _aliases might not be set up yet
        aliases = self.__dict__.get("_aliases", {})
        obj, attrib = aliases.get(name, (None, None))
        if obj:
            setattr(obj, attrib, value)
            if name == "gradient" or attrib == "gradient_style":
                self._set_aliases()
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
        obj, attrib = self.__dict__["_aliases"].get(name, (None, None))
        if obj:
            res = getattr(obj, attrib)
        else:
            res = self.__dict__[name]
        return res

    def _set_aliases(self):
        """Set aliases for style attributes based on the style map."""
        _aliases = {}
        for alias, path_attrib in self._style_map.items():
            style_path, attrib = path_attrib
            obj = self
            for attrib_name in style_path.split("."):
                obj = obj.__dict__[attrib_name]
            _aliases[alias] = (obj, attrib)

        self.__dict__["_aliases"] = _aliases
