"""Convert ``Vector`` draw requests into drawable arrow groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from simetri.base.all_enums import MarkerType
from simetri.base.common import PointType, VecType
from simetri.config.settings import runtime_defaults
from simetri.geom.vectors import Vector
from simetri.group.batch import Group
from simetri.helpers.illustration import Arrow, vec_arrow
from simetri.shapes.shape import Shape

_VECTOR_PLACEMENT_KEYS = frozenset({"vec_end", "vec_start"})
_STYLE_PREFIXES = ("head_", "shaft_")
_PART_STYLE_KEYS = (
    "alpha",
    "color",
    "fill_alpha",
    "fill_color",
    "line_alpha",
    "line_color",
    "line_dash_array",
    "line_width",
)


def validate_vector_draw_batch(
    items: Sequence[Any],
    draw_kwargs: Mapping[str, Any],
    pos: PointType | None,
) -> None:
    """Raise when vector placement kwargs conflict with the draw call.

    Args:
        items: Drawables passed to ``Canvas.draw``.
        draw_kwargs: Style and placement keyword arguments.
        pos: Optional midpoint destination for the draw call.

    Raises:
        ValueError: If placement rules for vectors are violated.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.vector_draw import validate_vector_draw_batch
        >>> validate_vector_draw_batch([sg.Vector(40, 0)], {}, None)
    """
    if "vec_start" in draw_kwargs and "vec_end" in draw_kwargs:
        vector_count = sum(1 for item in items if isinstance(item, Vector))
        if vector_count > 1:
            raise ValueError(
                "Cannot use both vec_start and vec_end when drawing "
                "more than one Vector."
            )
    for item in items:
        if not isinstance(item, Vector):
            continue
        vector_x, vector_y = item[:2]
        if vector_x == 0 and vector_y == 0:
            raise ValueError("Cannot draw a zero-length Vector.")
        if pos is not None and (
            "vec_start" in draw_kwargs or "vec_end" in draw_kwargs
        ):
            raise ValueError(
                "Cannot combine pos with vec_start or vec_end "
                "when drawing a Vector."
            )


def _resolve_part_style(
    draw_kwargs: Mapping[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Return style overrides for one arrow part from canvas draw kwargs."""
    other_prefix = "head_" if prefix == "shaft_" else "shaft_"
    part_style: dict[str, Any] = {}
    for key, value in draw_kwargs.items():
        if key.startswith(other_prefix):
            continue
        if key.startswith(prefix):
            mapped = key[len(prefix) :]
            part_style[mapped] = value
            if mapped == "line_color" and "color" in part_style:
                del part_style["color"]
    for key in _PART_STYLE_KEYS:
        if key in draw_kwargs and not key.startswith(("shaft_", "head_")):
            if key not in part_style:
                part_style[key] = draw_kwargs[key]
    if prefix == "head_":
        if "fill_color" in part_style and "line_color" not in part_style:
            if "color" in part_style:
                part_style["line_color"] = part_style["color"]
        if "line_color" in part_style and "fill_color" not in part_style:
            if "color" in part_style:
                part_style["fill_color"] = part_style["color"]
        if "color" in part_style and (
            "line_color" in part_style or "fill_color" in part_style
        ):
            del part_style["color"]
    elif prefix == "shaft_" and "line_color" in part_style and "color" in part_style:
        del part_style["color"]
    return part_style


def _apply_part_style(shape: Shape, part_style: Mapping[str, Any]) -> None:
    """Apply resolved style overrides onto a shaft or head shape."""
    for name, value in part_style.items():
        setattr(shape, name, value)


def _default_shaft_style() -> dict[str, Any]:
    return {
        "line_width": runtime_defaults["shaft_line_width"],
        "line_color": runtime_defaults["shaft_line_color"],
        "fill": False,
    }


def _default_head_style() -> dict[str, Any]:
    return {
        "line_width": runtime_defaults["head_line_width"],
        "line_color": runtime_defaults["head_line_color"],
        "fill_color": runtime_defaults["head_fill_color"],
        "fill": True,
    }


def _merge_style(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    merged.update(overrides)
    return merged


def _placement_for_vector(
    draw_kwargs: dict[str, Any],
) -> tuple[PointType | None, PointType | None]:
    """Remove vector placement kwargs and return tail/tip positions."""
    start = draw_kwargs.pop("vec_start", None)
    end = draw_kwargs.pop("vec_end", None)
    if start is None and end is None:
        start = (0.0, 0.0)
    return start, end


def _strip_vector_only_kwargs(draw_kwargs: dict[str, Any]) -> None:
    """Remove vector-only kwargs that must not reach ``draw.draw``."""
    for key in list(draw_kwargs):
        if key in _VECTOR_PLACEMENT_KEYS or key.startswith(_STYLE_PREFIXES):
            del draw_kwargs[key]


def _strip_applied_style_kwargs(draw_kwargs: dict[str, Any]) -> None:
    """Remove generic style kwargs already applied to arrow parts."""
    for key in _PART_STYLE_KEYS:
        draw_kwargs.pop(key, None)


def vector_to_arrow(vector: Vector, draw_kwargs: dict[str, Any]) -> Group:
    """Build a positioned arrow group from a vector and canvas draw kwargs.

    ``draw_kwargs`` is copied by the caller; this function removes vector
    placement and arrow-part style keys that it consumes.

    Args:
        vector: Displacement drawn as an arrow.
        draw_kwargs: Canvas draw keyword arguments (mutated).

    Returns:
        Group: An ``Arrow`` whose line and head shapes carry resolved styles.

    Examples:
        >>> import simetri.graphics as sg
        >>> kwargs = {"vec_start": (40, 20)}
        >>> arrow = vector_to_arrow(sg.Vector(60, 80), kwargs)
        >>> arrow.p1
        (40, 20)
        >>> "vec_start" in kwargs
        False
    """
    start, end = _placement_for_vector(draw_kwargs)
    arrow = vec_arrow(vector, start=start, end=end)
    shaft_style = _merge_style(
        _default_shaft_style(),
        _resolve_part_style(draw_kwargs, "shaft_"),
    )
    head_style = _merge_style(
        _default_head_style(),
        _resolve_part_style(draw_kwargs, "head_"),
    )
    _apply_part_style(arrow.line, shaft_style)
    for head_shape in arrow.heads:
        if head_shape is not None:
            _apply_part_style(head_shape, head_style)

    _strip_vector_only_kwargs(draw_kwargs)
    _strip_applied_style_kwargs(draw_kwargs)
    return arrow


def prepare_vector_batch(
    items: Sequence[Any],
    draw_kwargs: Mapping[str, Any],
    pos: PointType | None,
) -> None:
    """Validate placement when the draw call includes a ``Vector``.

    Items that are not vectors are ignored. A call with no ``Vector``
    does not read ``vec_start`` or ``vec_end``.

    Args:
        items: Drawables passed to ``Canvas.draw``.
        draw_kwargs: Style and placement keyword arguments.
        pos: Optional midpoint destination for the draw call.

    Raises:
        ValueError: If placement rules for vectors are violated.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.vector_draw import prepare_vector_batch
        >>> prepare_vector_batch([sg.Circle(20)], {"vec_start": (0, 0)}, None)
        >>> prepare_vector_batch([sg.Vector(0, 0)], {}, None)
        Traceback (most recent call last):
            ...
        ValueError: Cannot draw a zero-length Vector.
    """
    for item in items:
        if isinstance(item, Vector):
            validate_vector_draw_batch(items, draw_kwargs, pos)
            return


def drawable_for_item(
    item: Any,
    draw_kwargs: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any]]:
    """Return the drawable and kwargs for one draw item.

    A ``Vector`` becomes an ``Arrow``. Other items are returned unchanged.

    Args:
        item: One drawable from ``Canvas.draw``.
        draw_kwargs: Style and placement keyword arguments. Copied when
            ``item`` is a ``Vector``; that copy is mutated.

    Returns:
        tuple: The drawable to sketch, and the kwargs to pass to ``draw``.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.vector_draw import drawable_for_item
        >>> arrow, kwargs = drawable_for_item(sg.Vector(40, 0), {})
        >>> arrow.p1
        (0.0, 0.0)
        >>> arrow.p2
        (40.0, 0.0)
        >>> kwargs
        {}
    """
    if isinstance(item, Vector):
        part_kwargs = dict(draw_kwargs)
        return vector_to_arrow(item, part_kwargs), part_kwargs
    return item, draw_kwargs


def is_vector_marker_shape(
    item: Any,
    draw_kwargs: Mapping[str, Any],
) -> bool:
    """Return whether ``item`` is a shape drawn with vector markers.

    Args:
        item: One drawable from ``Canvas.draw``.
        draw_kwargs: Style and placement keyword arguments.

    Returns:
        bool: True when ``item`` is a ``Shape`` and its marker type is
        ``MarkerType.VECTOR``.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.vector_draw import is_vector_marker_shape
        >>> shape = sg.Shape([(0, 0), (40, 0), (20, 30)])
        >>> is_vector_marker_shape(shape, {"marker_type": sg.MarkerType.VECTOR})
        True
        >>> is_vector_marker_shape(shape, {})
        False
        >>> is_vector_marker_shape(sg.Vector(40, 0), {})
        False
    """
    if not isinstance(item, Shape):
        return False
    if "marker_type" in draw_kwargs:
        marker_type = draw_kwargs["marker_type"]
    elif (
        "marker_type" in item.__dict__
        and item.__dict__["marker_type"] is not None
    ):
        marker_type = item.__dict__["marker_type"]
    else:
        return False
    return (
        marker_type == MarkerType.VECTOR
        or marker_type == MarkerType.VECTOR.value
    )


def draw_vector_marker_shape(
    canvas: Any,
    shape: Shape,
    *,
    pos: PointType | None,
    angle: float,
    rotocenter: PointType,
    scale: VecType | float,
    about: PointType,
    draw_kwargs: Mapping[str, Any],
) -> None:
    """Draw a shape's edges as vectors, and its body when that is requested.

    Args:
        canvas: Canvas that receives the sketches (mutated).
        shape: Shape whose edges are drawn as vectors.
        pos: Midpoint destination. Applied to the body and to each edge.
        angle: Rotation applied to each draw.
        rotocenter: Center of that rotation.
        scale: Scale applied to each draw.
        about: Center of that scale.
        draw_kwargs: Style and placement keyword arguments.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.vector_draw import draw_vector_marker_shape
        >>> canvas = sg.Canvas()
        >>> shape = sg.Shape([(0, 0), (40, 0)], fill=False)
        >>> draw_vector_marker_shape(
        ...     canvas,
        ...     shape,
        ...     pos=None,
        ...     angle=0,
        ...     rotocenter=(0, 0),
        ...     scale=(1, 1),
        ...     about=(0, 0),
        ...     draw_kwargs={"marker_type": sg.MarkerType.VECTOR},
        ... )
        >>> canvas.active_page.sketches[-1].vertices
        [(0.0, 0.0), (40.0, 0.0)]
    """
    body_kwargs = dict(draw_kwargs)
    del body_kwargs["marker_type"]
    if "draw_markers" in body_kwargs:
        del body_kwargs["draw_markers"]
    body_kwargs["stroke"] = False
    if "fill" in body_kwargs:
        wants_fill = body_kwargs["fill"]
    elif "fill" in shape.__dict__ and shape.__dict__["fill"] is not None:
        wants_fill = shape.__dict__["fill"]
    else:
        wants_fill = runtime_defaults["fill"]
    if "indices" in body_kwargs:
        wants_indices = bool(body_kwargs["indices"])
    elif "indices" in shape.__dict__:
        wants_indices = bool(shape.indices)
    else:
        wants_indices = False
    if "show_vertex_coords" in body_kwargs:
        wants_coords = bool(body_kwargs["show_vertex_coords"])
    elif "show_vertex_coords" in shape.__dict__:
        wants_coords = bool(shape.show_vertex_coords)
    else:
        wants_coords = False
    if wants_fill or wants_indices or wants_coords:
        canvas.draw(
            shape,
            pos=pos,
            angle=angle,
            rotocenter=rotocenter,
            scale=scale,
            about=about,
            show=False,
            **body_kwargs,
        )
    vector_kwargs = dict(draw_kwargs)
    for key in (
        "draw_markers",
        "fill",
        "indices",
        "marker_type",
        "show_vertex_coords",
        "stroke",
    ):
        if key in vector_kwargs:
            del vector_kwargs[key]
    pos_dx = 0.0
    pos_dy = 0.0
    if pos is not None:
        mid_x, mid_y = shape.midpoint[:2]
        dest_x, dest_y = pos[:2]
        pos_dx = dest_x - mid_x
        pos_dy = dest_y - mid_y
    for edge in shape.edges:
        start_x, start_y = edge[0][:2]
        end_x, end_y = edge[1][:2]
        start = (start_x + pos_dx, start_y + pos_dy)
        end = (end_x + pos_dx, end_y + pos_dy)
        canvas.draw(
            Vector(start, end),
            vec_start=start,
            angle=angle,
            rotocenter=rotocenter,
            scale=scale,
            about=about,
            show=False,
            **vector_kwargs,
        )
