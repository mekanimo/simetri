"""Base transforms for Shape and Group.

``Base`` provides ``translate``, ``rotate``, ``mirror``, ``glide``,
``scale``, ``shear``, ``move``, and ``move_to``.

Examples:
    >>> import simetri.graphics as sg
    >>> s = sg.Shape([(0, 0), (10, 0), (10, 10)], closed=True)
    >>> s.translate(5, 0).rotate(sg.pi / 4, about=s.midpoint)
"""

__all__ = ["Base"]

import operator
from collections.abc import Sequence
from math import hypot
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from ..geom.affine import (
    glide_matrix,
    mirror_matrix,
    rotation_matrix,
    scale_in_place_matrix,
    shear_matrix,
    translation_matrix,
)
from ..geom.segments.line_utils import angled_line, line_angle, offset_line
from ..render.style_map import shape_args
from .all_enums import (
    Anchor,
    InPlace,
    Reference,
    Side,
    TransformationType,
    Types,
    anchors,
    get_enum_value,
    line_refs,
    point_refs,
)
from .common import (
    LineType,
    PointType,
)

STYLE_ATTRIBUTES = set(shape_args)


def _update_inplace(
    xform_matrix: NDArray,
    xform_type: TransformationType,
    incr: float
    | tuple[float, float]
    | tuple[callable, Any]
    | tuple[InPlace, Any]
    | None = None,
):
    """Update a transformation matrix for one more repetition.

    Supported ``incr`` forms:
    - ``float``: additive increment for rotation or glide
    - ``(x, y)``: additive increment for translate, scale, or shear
    - ``(callable, arg)``: callable returns one of the above increment values
    - ``(InPlace.OP, value)``: applies that operation to the current parameter

    Args:
        xform_matrix (NDArray): Affine matrix to update (mutated).
        xform_type (TransformationType): Kind of transform stored in the matrix.
        incr: Increment or operator pair. Defaults to None.

    Returns:
        NDArray: The same matrix after the update.
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
    """Return a point or line resolved from a named reference on ``target``.

    Args:
        target: Object whose attributes supply the reference.
        reference: A reference name, or ``(name, value)``.

    Returns:
        The resolved point, line, or length.
    """
    if isinstance(reference, [tuple, list]):
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
    """Shared transform and anchor API for Shape and Group.

    Anchor names (``midpoint``, ``southwest``, ``left``, …) resolve through
    ``b_box``. Transform methods support ``reps`` (repeated copies) and
    optional ``incr`` increments between repetitions.

    Note:
        Concrete subclasses must implement ``_update``, ``copy``, ``append``
        (for groups), and expose ``b_box`` / ``type``.
    """

    def __getattr__(self, name: str) -> Any:
        """Resolve bounding-box anchors or fall back to instance attributes.

        Args:
            name: Attribute or anchor name (also accepts ``bbox_`` prefix).

        Returns:
            Any: Anchor geometry, a stored attribute, or ``None`` for an
            unset style name.

        Raises:
            AttributeError: If the name is unknown and not a style attribute.

        Examples:
            >>> import simetri.graphics as sg
            >>> square = sg.Shape([(0, 0), (2, 0), (2, 2)], closed=True)
            >>> square.southwest
            (0.0, 0.0)
            >>> square.midpoint
            (1.0, 1.0)
        """
        if name in anchors:
            if name.startswith("bbox_"):
                name = name[4:]
            res = getattr(self.b_box, name)
        else:
            try:
                res = self.__dict__[name]
            except KeyError:
                try:
                    res = getattr(super(), name)
                except AttributeError as attr_exc:
                    # For style attributes, return None instead of raising an error
                    # This allows the canvas property resolution to work properly
                    if name in STYLE_ATTRIBUTES:
                        return None
                    msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
                    raise AttributeError(msg) from attr_exc

        return res

    def translate(
        self,
        dx: float = 0,
        dy: float = 0,
        take: slice | None = None,
        reps: int = 0,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | None = None,
        merge: bool = False,
    ) -> Self:
        """Translate the object by ``dx`` and ``dy``.

        This object is updated.

        Args:
            dx: Translation along the x-axis.
            dy: Translation along the y-axis.
            take: Optional slice selecting which group elements to transform.
            reps: Extra repetitions of the transform. Defaults to 0.
            incr: Optional increment applied between repetitions.
            merge: If True, merge results where supported.

        Returns:
            Self: This object after the translation is applied.

        Examples:
            >>> import simetri.graphics as sg
            >>> square = sg.Shape([(0, 0), (1, 0)])
            >>> square.translate(10, 5) is square
            True
            >>> square.vertices
            ((10.0, 5.0), (11.0, 5.0))
            >>> square.translate(0, -5).vertices[0]
            (10.0, 0.0)
        """
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

        return res

    def translate_along(
        self,
        path: Sequence[PointType],
        step: int = 1,
        align_tangent: bool = False,
        scale: float = 1,  # scale factor
        rotate: float = 0,  # angle in radians
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | None = None,
        merge: bool = False,
    ) -> Self:
        """Place copies of this object at points along ``path``.

        The first path point is used as the new position of this object.
        Later points, taken every ``step``, are appended as copies.

        Args:
            path (Sequence[PointType]): Points to place the object on.
            step (int, optional): Use every ``step``-th point after the first.
                Defaults to 1.
            align_tangent (bool, optional): Rotate to the path direction.
                Defaults to False.
            scale (float, optional): Scale applied at each placed copy.
                Defaults to 1.
            rotate (float, optional): Extra rotation in radians at each copy.
                Defaults to 0.
            incr: Extra translation accumulated between copies, in the
                same forms as ``translate``. The first copy is not shifted.
                Later copies add this increment to their path position.
                Defaults to None.
            merge (bool, optional): If True and copies were appended, replace
                this object's elements with ``merge_shapes()``. Defaults to False.

        Returns:
            Self: This object, with the extra placements appended.

        Examples:
            >>> import simetri.graphics as sg
            >>> path = [(0, 0), (5, 0), (10, 0)]
            >>> mark = sg.Shape([(0, 0), (1, 0)])
            >>> result = mark.translate_along(path, step=1, incr=(1, 0))
            >>> result is mark
            True
            >>> path
            [(0, 0), (5, 0), (10, 0)]
        """
        x, y = path[0][:2]
        self.move_to((x, y))
        dup = self.copy()
        if align_tangent:
            tangent = line_angle(path[-1], path[0])
            self.rotate(tangent, about=path[0], reps=0)
        dup2 = dup.copy()
        offset = translation_matrix(0, 0)
        for i, point in enumerate(path[1::step]):
            dup2 = dup2.copy()
            px, py = point[:2]
            if incr is not None and i > 0:
                offset = _update_inplace(
                    offset, TransformationType.TRANSLATE, incr
                )
                ox, oy = offset[2, :2]
                px += float(ox)
                py += float(oy)
            dup2.move_to((px, py))
            if scale != 1:
                dup2.scale(scale, about=(px, py))
            if rotate != 0:
                dup2.rotate(rotate, about=(px, py))
            self.append(dup2)
            if align_tangent:
                tangent = line_angle(path[i - 1], path[i])
                dup2.rotate(tangent, about=(px, py), reps=0)
        if merge and len(path[1::step]) > 0:
            merged = self.merge_shapes()
            self[:] = merged.elements[:]
        return self

    def rotate(
        self,
        angle: float,
        about: PointType = (0, 0),
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | None = None,
        merge: bool = False,
    ) -> Self:
        """Rotate by ``angle`` radians about a point.

        This object is updated.

        Args:
            angle: Rotation angle in radians, counterclockwise.
            about: Center of rotation. Defaults to ``(0, 0)``.
            reps: Extra repetitions of the transform. Defaults to 0.
            take: Optional slice of group elements to transform.
            incr: Optional angle or operator increment between repetitions.
            merge: If True, merge results where supported.

        Returns:
            Self: This object after the rotation is applied.

        Examples:
            >>> import simetri.graphics as sg
            >>> arm = sg.Shape([(1, 0)])
            >>> arm.rotate(sg.pi / 2) is arm
            True
            >>> abs(arm.vertices[0][0]) < 1e-9 and abs(arm.vertices[0][1] - 1) < 1e-9
            True
            >>> arm.rotate(-sg.pi / 2).vertices[0][0]
            1.0
        """
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
        return res

    def mirror(
        self,
        about: LineType | PointType,
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | None = None,
        merge: bool = False,
    ) -> Self:
        """Mirror this object about a line or a point.

        This object is updated.

        Args:
            about (LineType | PointType): Mirror line, or a point treated as
                the mirror origin.
            reps (int, optional): Extra repetitions. Defaults to 0.
            take: Optional slice of group elements to transform.
            incr: Optional increment between repetitions. Defaults to None.
            merge (bool, optional): Merge results where supported.
                Defaults to False.

        Returns:
            Self: This object after the mirror is applied.

        Examples:
            >>> import simetri.graphics as sg
            >>> mark = sg.Shape([(0, 1)])
            >>> axis = [(0, 0), (1, 0)]
            >>> mark.mirror(axis) is mark
            True
            >>> abs(mark.vertices[0][1] + 1) < 1e-9
            True
            >>> axis
            [(0, 0), (1, 0)]
        """
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

        return res

    def glide(
        self,
        glide_line: LineType,
        glide_dist: float,
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | None = None,
        merge: bool = False,
    ) -> Self:
        """Mirror this object across ``glide_line``, then slide it.

        This object is updated.

        Args:
            glide_line (LineType): Line to mirror across and travel along.
            glide_dist (float): Distance to travel along that line after
                the mirror.
            reps (int, optional): Extra repetitions. Defaults to 0.
            take: Optional slice of group elements to transform.
            incr: Optional increment between repetitions. Defaults to None.
            merge (bool, optional): Merge results where supported.
                Defaults to False.

        Returns:
            Self: This object after the glide is applied.

        Examples:
            >>> import simetri.graphics as sg
            >>> mark = sg.Shape([(0, 1)])
            >>> line = [(0, 0), (1, 0)]
            >>> mark.glide(line, 2) is mark
            True
            >>> abs(mark.vertices[0][0] - 2) < 1e-9
            True
            >>> abs(mark.vertices[0][1] + 1) < 1e-9
            True
            >>> line
            [(0, 0), (1, 0)]
        """
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

        return res

    def scale(
        self,
        scale_x: float,
        scale_y: float | None = None,
        about: PointType = (0, 0),
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | None = None,
        merge: bool = False,
    ) -> Self:
        """Scale this object about a point.

        This object is updated. If ``scale_y`` is
        omitted, both axes use ``scale_x``.

        Args:
            scale_x (float): Scale factor on x.
            scale_y (float, optional): Scale factor on y. Defaults to
                ``scale_x``.
            about (PointType, optional): Fixed point. Defaults to ``(0, 0)``.
            reps (int, optional): Extra repetitions. Defaults to 0.
            take: Optional slice of group elements to transform.
            incr: Optional increment between repetitions. Defaults to None.
            merge (bool, optional): Merge results where supported.
                Defaults to False.

        Returns:
            Self: This object after the scale is applied.

        Examples:
            >>> import simetri.graphics as sg
            >>> bar = sg.Shape([(1, 0)])
            >>> bar.scale(2) is bar
            True
            >>> bar.vertices
            ((2.0, 0.0),)
            >>> bar.scale(1, 3).vertices[0][1]
            0.0
        """
        if scale_y is None:
            scale_y = scale_x
        transform = scale_in_place_matrix(scale_x, scale_y, about)
        res = self._update(
            transform,
            reps=reps,
            take=take,
            incr=incr,
            merge=merge,
            xform_type=TransformationType.SCALE,
        )

        return res

    def shear(
        self,
        theta_x: float,
        theta_y: float,
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | None = None,
        merge: bool = False,
    ) -> Self:
        """Shear this object by the given angles.

        This object is updated. A zero pair leaves the coordinates unchanged.

        Args:
            theta_x (float): Shear angle for the x direction, in radians.
            theta_y (float): Shear angle for the y direction, in radians.
            reps (int, optional): Extra repetitions. Defaults to 0.
            take: Optional slice of group elements to transform.
            incr: Optional increment between repetitions. Defaults to None.
            merge (bool, optional): Merge results where supported.
                Defaults to False.

        Returns:
            Self: This object after the shear is applied.

        Examples:
            >>> import simetri.graphics as sg
            >>> mark = sg.Shape([(1, 1)])
            >>> mark.shear(0, 0) is mark
            True
            >>> mark.vertices
            ((1.0, 1.0),)
        """
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

        return res

    def reset_xform_matrix(self) -> Self:
        """Set this object's transform matrix back to the identity.

        This object is updated. Later reads of the vertices use the
        original points.

        Returns:
            Self: This object.

        Examples:
            >>> import simetri.graphics as sg
            >>> mark = sg.Shape([(0, 0), (1, 0)])
            >>> mark.translate(3, 0)
            >>> mark.reset_xform_matrix() is mark
            True
            >>> mark.vertices
            ((0.0, 0.0), (1.0, 0.0))
        """
        self.__dict__["xform_matrix"] = np.identity(3)
        return self

    def transform(
        self,
        transform_matrix: NDArray,
        reps: int = 0,
        take: slice | None = None,
        merge: bool = False,
    ) -> Self:
        """Apply an affine matrix to this object.

        This object is updated.

        Args:
            transform_matrix (NDArray): Affine matrix to apply.
            reps (int, optional): Extra repetitions. Defaults to 0.
            take: Optional slice of group elements to transform.
            merge (bool, optional): Merge results where supported.
                Defaults to False.

        Returns:
            Self: This object after the matrix is applied.

        Examples:
            >>> import simetri.graphics as sg
            >>> mark = sg.Shape([(0, 0), (1, 0)])
            >>> mark.transform(mark.xform_matrix) is mark
            True
            >>> mark.vertices
            ((0.0, 0.0), (1.0, 0.0))
        """
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

        return res

    def move(
        self, pos: PointType, anchor: Anchor = Anchor.CENTER, **kwargs
    ) -> Self:
        """Move this object so the chosen anchor lands on ``pos``.

        This is the same as :meth:`move_to`. This object is updated.

        Args:
            pos (PointType): Target position of the anchor.
            anchor (Anchor, optional): Anchor to place on ``pos``.
                Defaults to ``Anchor.CENTER``.
            **kwargs: Extra attributes assigned on this object.

        Returns:
            Self: This object after the move.

        Examples:
            >>> import simetri.graphics as sg
            >>> mark = sg.Shape([(0, 0), (2, 0)])
            >>> mark.move((10, 0)) is mark
            True
            >>> mark.midpoint
            (10.0, 0.0)
        """
        return self.move_to(pos, anchor, **kwargs)

    def move_to(
        self, pos: PointType, anchor: Anchor = Anchor.CENTER, **kwargs
    ) -> Self:
        """Move this object so the chosen anchor lands on ``pos``.

        This object is updated. Keyword arguments are assigned onto this
        object before the translation.

        Args:
            pos (PointType): Target position of the anchor.
            anchor (Anchor, optional): Anchor to place on ``pos``.
                Defaults to ``Anchor.CENTER``.
            **kwargs: Extra attributes assigned on this object.

        Returns:
            Self: This object after the move.

        Examples:
            >>> import simetri.graphics as sg
            >>> mark = sg.Shape([(0, 0), (2, 0)])
            >>> target = (10, 4)
            >>> mark.move_to(target, anchor=sg.Anchor.SOUTHWEST) is mark
            True
            >>> mark.southwest
            (10.0, 4.0)
            >>> target
            (10, 4)
        """
        x, y = pos[:2]
        anchor = get_enum_value(Anchor, anchor)
        x1, y1 = getattr(self.b_box, anchor)
        transform = translation_matrix(x - x1, y - y1)
        for k, v in kwargs.items():
            setattr(self, k, v)
        res = self._update(transform, reps=0)

        return res

    def offset_line(self, side: Side, offset: float) -> LineType:
        """Return a bounding-box side shifted outward by ``offset``.

        Args:
            side (Side): ``LEFT``, ``RIGHT``, ``TOP``, or ``BOTTOM``.
            offset (float): Outward distance.

        Returns:
            LineType: The shifted side.

        Examples:
            >>> import simetri.graphics as sg
            >>> box = sg.Shape([(0, 0), (4, 0), (4, 2)], closed=True)
            >>> box.offset_line(sg.Side.BOTTOM, 1)[0][1]
            -1.0
        """
        side = get_enum_value(Side, side)
        return self.b_box.offset_line(side, offset)

    def offset_point(
        self, anchor: Anchor, dx: float, dy: float = 0
    ) -> PointType:
        """Return an anchor point shifted by ``dx`` and ``dy``.

        Args:
            anchor (Anchor): Anchor on the bounding box.
            dx (float): Shift along x.
            dy (float, optional): Shift along y. Defaults to 0.

        Returns:
            PointType: The shifted anchor.

        Examples:
            >>> import simetri.graphics as sg
            >>> box = sg.Shape([(0, 0), (4, 0), (4, 2)], closed=True)
            >>> box.offset_point(sg.Anchor.SOUTHWEST, 1, 2)
            (1.0, 2.0)
            >>> box.offset_point(sg.Anchor.NORTHEAST, -1)
            (3.0, 2.0)
        """
        anchor = get_enum_value(Anchor, anchor)
        return self.b_box.offset_point(anchor, dx, dy)
