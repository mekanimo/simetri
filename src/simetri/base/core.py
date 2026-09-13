"""Base transforms for Shape and Group.

``Base`` provides ``translate``, ``rotate``, ``mirror``, ``glide``,
``scale``, ``shear``, ``move``, and ``move_to``.

Examples:
    >>> import simetri.graphics as sg
    >>> s = sg.Shape([(0, 0), (10, 0), (10, 10)], closed=True)
    >>> s.translate(5, 0).rotate(sg.pi / 4, about=s.midpoint)
"""

__all__ = ["Base", "DynRef"]

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
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
from ..geom.geom_utils import offset_point
from ..geom.segments.line_utils import line_angle, offset_line, update_line
from ..helpers.utilities import decompose_transformations
from ..helpers.validation import is_line
from ..render.style_map import shape_args
from .all_enums import (
    Anchor,
    InPlace,
    Reference,
    ReferenceTarget,
    Side,
    TransformationType,
    Types,
    anchors,
    get_enum_value,
)
from .common import (
    LineType,
    PointType,
)

STYLE_ATTRIBUTES = set(shape_args)

_mirror_lines: dict[int, LineType] = {}


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
    - ``float``: additive increment for rotation or glide; perpendicular
      offset for a mirror line
    - ``(x, y)``: additive increment for translate, scale, or shear;
      translation of both endpoints of a mirror line
    - ``(angle, about)``: rotation of a mirror line about a point
    - a sequence of the mirror forms above, applied in that order
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

    def _replace_matrix(updated: NDArray) -> None:
        xform_matrix[:, :] = updated

    def _scale_factors(parts) -> tuple[float, float]:
        if xform_matrix[0, 1] == 0 and xform_matrix[1, 0] == 0:
            return float(xform_matrix[0, 0]), float(xform_matrix[1, 1])
        return float(parts.scale[0]), float(parts.scale[1])

    def _add_increment(value: Any) -> None:
        if xform_type == TransformationType.TRANSLATE:
            incr_x, incr_y = _coerce_pair(value)
            xform_matrix[2, 0] += incr_x
            xform_matrix[2, 1] += incr_y
        elif xform_type == TransformationType.ROTATE:
            parts = decompose_transformations(xform_matrix)
            angle = parts.rotation + _coerce_scalar(value)
            _replace_matrix(rotation_matrix(angle, parts.about))
        elif xform_type == TransformationType.SCALE:
            parts = decompose_transformations(xform_matrix)
            incr_x, incr_y = _coerce_pair(value)
            scale_x, scale_y = _scale_factors(parts)
            _replace_matrix(
                scale_in_place_matrix(
                    scale_x + incr_x, scale_y + incr_y, parts.about
                )
            )
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
            parts = decompose_transformations(xform_matrix)
            new_angle = oper(parts.rotation, _coerce_scalar(value))
            _replace_matrix(rotation_matrix(new_angle, parts.about))
        elif xform_type == TransformationType.SCALE:
            parts = decompose_transformations(xform_matrix)
            val_x, val_y = _coerce_pair(value)
            scale_x, scale_y = _scale_factors(parts)
            _replace_matrix(
                scale_in_place_matrix(
                    oper(scale_x, val_x), oper(scale_y, val_y), parts.about
                )
            )
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

    if xform_type == TransformationType.MIRROR:
        line = _mirror_lines[id(xform_matrix)]
        updated_line = update_line(line, incr)
        _mirror_lines[id(xform_matrix)] = updated_line
        _replace_matrix(mirror_matrix(updated_line))
        return xform_matrix

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


@dataclass
class DynRef:
    """A transform argument that is resolved again on every repetition.

    Pass a ``DynRef`` in place of a numeric transform argument (``dx``,
    ``angle``, ``about``, ...) together with ``dyn_ref=True``. The reference
    is resolved before each repetition, so the transformation matrix follows
    the geometry as it changes instead of staying fixed.

    Attributes:
        reference: A ``Reference`` member, or a callable invoked as
            ``reference(target, index, **kwargs)``. ``VERTEX`` and
            ``EDGE`` read ``target.vertices[index]`` and
            ``target.edges[index]`` on a Shape, or
            ``target.all_vertices[index]`` and
            ``target.all_edges[index]`` on a Group. Other members are
            attributes of the target.
        target: Which object the reference resolves against. ``KERNEL`` is
            the object as it was when the transform was called, ``PATTERN``
            is the result accumulated so far, and ``ACTIVE`` is the copy
            being produced for the current repetition. Defaults to
            ``ReferenceTarget.ACTIVE``.
        offset: Added to the resolved value. A number for lengths and lines,
            ``(dx, dy)`` for points. Defaults to None.
        multiplier: Factor applied to a resolved length. Defaults to None.
        modifier: Callable applied to the resolved value. Defaults to None.
        kwargs: Extra keyword arguments for a callable ``reference``.
            Defaults to None.
        index: Vertex or edge index for ``Reference.VERTEX`` and
            ``Reference.EDGE``. Defaults to None.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.base.all_enums import Reference, ReferenceTarget
        >>> from simetri.base.core import DynRef
        >>> box = sg.Shape([(0, 0), (100, 0), (100, 40)], closed=True)
        >>> gap = DynRef(Reference.WIDTH, ReferenceTarget.KERNEL)
        >>> row = box.translate(gap, 0, reps=2, dyn_ref=True)
        >>> [shape.midpoint[0] for shape in row]
        [50.0, 150.0, 250.0]

        Aimed at the pattern instead, each step clears everything placed
        so far.

        >>> box = sg.Shape([(0, 0), (100, 0), (100, 40)], closed=True)
        >>> gap = DynRef(Reference.WIDTH, ReferenceTarget.PATTERN)
        >>> row = box.translate(gap, 0, reps=3, dyn_ref=True)
        >>> [shape.midpoint[0] for shape in row]
        [50.0, 150.0, 350.0, 750.0]

        Vertex and edge references take ``index``.

        >>> box = sg.Shape([(0, 0), (100, 0), (100, 40), (0, 40)], closed=True)
        >>> fan = box.rotate(
        ...     sg.pi / 2,
        ...     about=DynRef(Reference.VERTEX, ReferenceTarget.ACTIVE, index=0),
        ...     reps=1,
        ...     dyn_ref=True,
        ... )
        >>> [tuple(shape.midpoint[:2]) for shape in fan]
        [(50.0, 20.0), (-20.0, 50.0)]

        >>> walk = box.mirror(
        ...     DynRef(Reference.EDGE, ReferenceTarget.ACTIVE, index=1),
        ...     reps=1,
        ...     dyn_ref=True,
        ... )
        >>> [shape.midpoint[0] for shape in walk]
        [50.0, 150.0]
    """

    reference: Reference | Callable
    target: ReferenceTarget = ReferenceTarget.ACTIVE
    offset: PointType | float | None = None
    multiplier: float | None = None
    modifier: Callable | None = None
    kwargs: dict | None = None
    index: int | None = None


class _Targets:
    """The three objects a ``DynRef`` can resolve against.

    Indexed by ``ReferenceTarget`` values, so a transform whose references
    aim at different targets resolves each one against its own object.

    ``active`` starts out as the kernel; a reps loop reassigns it to the copy
    it is about to transform. ``pattern`` is expected to track the growing
    result by itself, so it needs no per-repetition update.

    Args:
        kernel: The object as it was when the transform was called.
        pattern: The result accumulated so far.
    """

    def __init__(self, kernel: Any, pattern: Any):
        self.kernel = kernel
        self.pattern = pattern
        self.active = kernel

    def __getitem__(self, target: str) -> Any:
        """Return the object belonging to a ``ReferenceTarget`` value.

        Args:
            target: A ``ReferenceTarget`` value.

        Returns:
            The kernel, the pattern, or the active copy.

        Raises:
            KeyError: If ``target`` is not a ``ReferenceTarget`` value.
        """
        if target == ReferenceTarget.KERNEL.value:
            res = self.kernel
        elif target == ReferenceTarget.PATTERN.value:
            res = self.pattern
        elif target == ReferenceTarget.ACTIVE.value:
            res = self.active
        else:
            raise KeyError(f"{target!r} is not a reference target.")

        return res


def _is_scalar(value: Any) -> bool:
    """Return True if ``value`` is a single number."""
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_point(value: Any) -> bool:
    """Return True if ``value`` is one ``(x, y)`` or ``(x, y, 1)`` point."""
    if _is_scalar(value) or isinstance(value, (str, bytes)):
        return False
    try:
        length = len(value)
    except TypeError:
        return False

    return length in (2, 3) and _is_scalar(value[0])


def _is_line(value: Any) -> bool:
    """Return True if ``value`` is a two-point line."""
    if _is_scalar(value) or isinstance(value, (str, bytes)):
        return False
    try:
        length = len(value)
    except TypeError:
        return False

    return length == 2 and _is_point(value[0])


def _multiply_reference(value: Any, multiplier: float) -> float:
    """Scale a resolved length reference.

    Args:
        value: Resolved reference value.
        multiplier: Factor to apply.

    Returns:
        float: The scaled length.

    Raises:
        TypeError: If ``value`` is not a length.
    """
    if not _is_scalar(value):
        raise TypeError(
            "DynRef.multiplier applies to length references only "
            f"(Reference.WIDTH, Reference.HEIGHT); got {value!r}."
        )

    return value * multiplier


def _offset_reference(value: Any, offset: PointType | float) -> Any:
    """Shift a resolved reference by ``offset``.

    Args:
        value: Resolved length, point, or line.
        offset: A number for lengths and lines, ``(dx, dy)`` for points.

    Returns:
        The shifted length, point, or line.

    Raises:
        TypeError: If ``offset`` does not match the kind of ``value``.
    """
    if _is_scalar(value):
        if not _is_scalar(offset):
            raise TypeError(
                f"A length reference needs a numeric offset; got {offset!r}."
            )
        res = value + offset
    elif _is_line(value):
        if not _is_scalar(offset):
            raise TypeError(
                f"A line reference needs a numeric offset; got {offset!r}."
            )
        res = offset_line(value, offset)
    elif _is_point(value):
        if not _is_point(offset):
            raise TypeError(
                f"A point reference needs a (dx, dy) offset; got {offset!r}."
            )
        dx, dy = offset[:2]
        res = offset_point(value, dx, dy)
    else:
        raise TypeError(f"Cannot offset the reference value {value!r}.")

    return res


def _resolve_reference(dyn_ref: DynRef, target: Any, index: int) -> Any:
    """Resolve ``dyn_ref`` against ``target`` for repetition ``index``.

    Args:
        dyn_ref: The reference to resolve.
        target: Object supplying the reference geometry.
        index: Index of the current repetition.

    Returns:
        The resolved length, point, or line.
    """
    if dyn_ref.kwargs is None:
        kwargs = {}
    else:
        kwargs = dyn_ref.kwargs

    if callable(dyn_ref.reference):
        if dyn_ref.index is not None:
            raise ValueError(
                "DynRef.index is only for Reference.VERTEX and "
                "Reference.EDGE."
            )
        res = dyn_ref.reference(target, index, **kwargs)
    else:
        reference = get_enum_value(Reference, dyn_ref.reference)
        if reference == Reference.VERTEX:
            if dyn_ref.index is None:
                raise ValueError("Reference.VERTEX needs DynRef.index.")
            if target.type == Types.GROUP:
                res = target.all_vertices[dyn_ref.index]
            else:
                res = target.vertices[dyn_ref.index]
        elif reference == Reference.EDGE:
            if dyn_ref.index is None:
                raise ValueError("Reference.EDGE needs DynRef.index.")
            if target.type == Types.GROUP:
                res = target.all_edges[dyn_ref.index]
            else:
                res = target.edges[dyn_ref.index]
        else:
            if dyn_ref.index is not None:
                raise ValueError(
                    "DynRef.index is only for Reference.VERTEX and "
                    "Reference.EDGE."
                )
            res = getattr(target, reference)

    if dyn_ref.multiplier is not None:
        res = _multiply_reference(res, dyn_ref.multiplier)
    if dyn_ref.modifier is not None:
        res = dyn_ref.modifier(res)
    if dyn_ref.offset is not None:
        res = _offset_reference(res, dyn_ref.offset)

    return res


def _resolve_arg(value: Any, targets: _Targets, index: int) -> Any:
    """Resolve one transform argument that may be a ``DynRef``.

    Args:
        value: Transform argument, dynamic or plain.
        targets: The objects references resolve against.
        index: Index of the current repetition.

    Returns:
        The resolved value, or ``value`` unchanged when it is not a
        ``DynRef``.
    """
    if isinstance(value, DynRef):
        target = targets[get_enum_value(ReferenceTarget, value.target)]
        res = _resolve_reference(value, target, index)
    else:
        res = value

    return res


def _reject_dyn_refs(**arguments: Any) -> None:
    """Raise if a ``DynRef`` was passed without ``dyn_ref=True``.

    Args:
        **arguments: Transform arguments by name.

    Raises:
        ValueError: If any argument is a ``DynRef``.
    """
    dynamic = [
        name for name, value in arguments.items() if isinstance(value, DynRef)
    ]
    if dynamic:
        names = ", ".join(sorted(dynamic))
        raise ValueError(
            f"{names} is a DynRef, so the transform needs dyn_ref=True."
        )


def _dyn_matrix(
    builder: Callable, arguments: tuple, targets: _Targets, index: int
) -> NDArray:
    """Build a transformation matrix, resolving its dynamic arguments.

    Args:
        builder: Matrix constructor from ``geom.affine``.
        arguments: The builder's positional arguments, dynamic or plain.
        targets: The objects references resolve against.
        index: Index of the current repetition.

    Returns:
        NDArray: The transformation matrix for this repetition.
    """
    return builder(
        *[_resolve_arg(argument, targets, index) for argument in arguments]
    )


def _make_xform(
    builder: Callable, dyn_ref: bool, kernel: Any, **arguments: Any
) -> tuple[NDArray, Callable | None]:
    """Return a transform's first matrix and its per-repetition factory.

    Args:
        builder: Matrix constructor from ``geom.affine``.
        dyn_ref: True if dynamic references are enabled.
        kernel: Object the first repetition resolves against.
        **arguments: The builder's arguments, in positional order.

    Returns:
        tuple: The matrix for the first repetition, and a factory callable
        that builds the matrix for later repetitions, or None when
        ``dyn_ref`` is False.

    Raises:
        ValueError: If an argument is a ``DynRef`` while ``dyn_ref`` is False.
    """
    if dyn_ref:
        factory = partial(_dyn_matrix, builder, tuple(arguments.values()))
        matrix = factory(_Targets(kernel, kernel), 0)
    else:
        _reject_dyn_refs(**arguments)
        factory = None
        matrix = builder(*arguments.values())

    return matrix, factory


def _next_xform_matrix(
    xform_matrix: NDArray,
    xform_type: TransformationType,
    incr: Any,
    dyn_ref: Callable | None,
    targets: _Targets | None,
    index: int,
) -> NDArray:
    """Return the transformation matrix to use for repetition ``index``.

    Args:
        xform_matrix: Matrix used by the previous repetition.
        xform_type: Kind of transform stored in the matrix.
        incr: Increment applied between repetitions, or None.
        dyn_ref: Matrix factory built by the transform method, or None.
        targets: The objects references resolve against, or None when
            ``dyn_ref`` is None.
        index: Index of the current repetition.

    Returns:
        NDArray: The matrix for this repetition.

    Raises:
        ValueError: If both ``incr`` and ``dyn_ref`` are given, or if
            ``dyn_ref`` is not a matrix factory.
    """
    if incr is not None and dyn_ref:
        raise ValueError(
            "incr and dyn_ref both change the matrix between repetitions. "
            "Use one or the other."
        )

    if dyn_ref:
        if not callable(dyn_ref):
            raise ValueError(
                f"dyn_ref reached _update as {dyn_ref!r}. Dynamic references "
                "need a transform method (translate, rotate, mirror, glide, "
                "scale, or shear) to build the matrix factory."
            )
        res = dyn_ref(targets, index)
    elif incr is not None and index > 0:
        res = _update_inplace(xform_matrix, xform_type, incr)
    else:
        res = xform_matrix

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
        dyn_ref: bool = False,
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
        transform, dyn_ref = _make_xform(
            translation_matrix, dyn_ref, self, dx=dx, dy=dy
        )
        if self.type == Types.SHAPE:
            res = self._update(
                transform,
                reps=reps,
                incr=incr,
                dyn_ref=dyn_ref,
                merge=merge,
                xform_type=TransformationType.TRANSLATE,
            )
        else:
            res = self._update(
                transform,
                reps=reps,
                take=take,
                incr=incr,
                dyn_ref=dyn_ref,
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
        dyn_ref: bool = False,
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

        Raises:
            ValueError: If ``dyn_ref`` is True.

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
        if dyn_ref:
            raise ValueError(
                "translate_along places copies on given path points, so "
                "dyn_ref has no transform arguments to resolve. Use "
                "translate, rotate, mirror, glide, scale, or shear instead."
            )
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
        dyn_ref: bool = False,
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
        transform, dyn_ref = _make_xform(
            rotation_matrix, dyn_ref, self, angle=angle, about=about
        )
        if self.__class__.__name__ == "Shape":
            res = self._update(
                transform,
                reps=reps,
                incr=incr,
                dyn_ref=dyn_ref,
                merge=merge,
                xform_type=TransformationType.ROTATE,
            )

        else:
            res = self._update(
                transform,
                reps=reps,
                take=take,
                incr=incr,
                dyn_ref=dyn_ref,
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
        dyn_ref: bool = False,
        merge: bool = False,
    ) -> Self:
        """Mirror this object about a line or a point.

        This object is updated.

        Args:
            about (LineType | PointType): Mirror line, or a point treated as
                the mirror origin.
            reps (int, optional): Extra repetitions. Defaults to 0.
            take: Optional slice of group elements to transform.
            incr: Optional increment applied to the mirror line between
                repetitions. A number offsets the line. ``(dx, dy)``
                translates it. ``(angle, about)`` rotates it about a
                point. A sequence of those is applied in order.
                Defaults to None.
            merge (bool, optional): Merge results where supported.
                Defaults to False.

        Returns:
            Self: This object after the mirror is applied.

        Raises:
            ValueError: If ``incr`` is given and ``about`` is not a line.

        Examples:
            >>> import simetri.graphics as sg
            >>> mark = sg.Shape([(0, 40)])
            >>> axis = [(0, 0), (100, 0)]
            >>> mark.mirror(axis) is mark
            True
            >>> abs(mark.vertices[0][1] + 40) < 1e-9
            True
            >>> axis
            [(0, 0), (100, 0)]
            >>> mark = sg.Shape([(0, 40)])
            >>> row = mark.mirror([(0, 0), (100, 0)], reps=2, incr=40)
            >>> [round(shape.vertices[0][1], 10) for shape in row]
            [40.0, -40.0, 120.0]
        """
        transform, dyn_ref = _make_xform(
            mirror_matrix, dyn_ref, self, about=about
        )
        seeded = False
        if incr is not None:
            if not is_line(about):
                raise ValueError(
                    "mirror incr updates a line; about must be a line "
                    "defined by two points."
                )
            start, end = about
            x1, y1 = start[:2]
            x2, y2 = end[:2]
            _mirror_lines[id(transform)] = [[x1, y1], [x2, y2]]
            seeded = True
        try:
            if self.__class__.__name__ == "Shape":
                res = self._update(
                    transform,
                    reps=reps,
                    incr=incr,
                    dyn_ref=dyn_ref,
                    merge=merge,
                    xform_type=TransformationType.MIRROR,
                )
            else:
                res = self._update(
                    transform,
                    reps=reps,
                    take=take,
                    incr=incr,
                    dyn_ref=dyn_ref,
                    merge=merge,
                    xform_type=TransformationType.MIRROR,
                )
        finally:
            if seeded:
                del _mirror_lines[id(transform)]

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
        dyn_ref: bool = False,
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
        transform, dyn_ref = _make_xform(
            glide_matrix,
            dyn_ref,
            self,
            glide_line=glide_line,
            glide_dist=glide_dist,
        )
        if self.__class__.__name__ == "Shape":
            res = self._update(
                transform,
                reps=reps,
                incr=incr,
                dyn_ref=dyn_ref,
                merge=merge,
                xform_type=TransformationType.GLIDE,
            )
        else:
            res = self._update(
                transform,
                reps=reps,
                take=take,
                incr=incr,
                dyn_ref=dyn_ref,
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
        dyn_ref: bool = False,
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
        transform, dyn_ref = _make_xform(
            scale_in_place_matrix,
            dyn_ref,
            self,
            scale_x=scale_x,
            scale_y=scale_y,
            about=about,
        )
        if self.__class__.__name__ == "Shape":
            res = self._update(
                transform,
                reps=reps,
                incr=incr,
                dyn_ref=dyn_ref,
                merge=merge,
                xform_type=TransformationType.SCALE,
            )
        else:
            res = self._update(
                transform,
                reps=reps,
                take=take,
                incr=incr,
                dyn_ref=dyn_ref,
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
        dyn_ref: bool = False,
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
        transform, dyn_ref = _make_xform(
            shear_matrix, dyn_ref, self, theta_x=theta_x, theta_y=theta_y
        )
        if self.__class__.__name__ == "Shape":
            res = self._update(
                transform,
                reps=reps,
                incr=incr,
                dyn_ref=dyn_ref,
                merge=merge,
                xform_type=TransformationType.SHEAR,
            )
        else:
            res = self._update(
                transform,
                reps=reps,
                take=take,
                incr=incr,
                dyn_ref=dyn_ref,
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
        dyn_ref: bool = False,
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

        Raises:
            ValueError: If ``dyn_ref`` is True.

        Examples:
            >>> import simetri.graphics as sg
            >>> mark = sg.Shape([(0, 0), (1, 0)])
            >>> mark.transform(mark.xform_matrix) is mark
            True
            >>> mark.vertices
            ((0.0, 0.0), (1.0, 0.0))
        """
        if dyn_ref:
            raise ValueError(
                "transform takes a ready-made matrix, so dyn_ref has no "
                "arguments to resolve. Use translate, rotate, mirror, "
                "glide, scale, or shear instead."
            )
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
