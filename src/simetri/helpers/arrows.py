"""Arrows."""

from __future__ import annotations

from math import atan2, hypot, pi
from typing import Any

from numpy.typing import NDArray

from ..base.all_enums import Align, FrameShape, HeadPos, Types, WarningType
from ..base.common import PointType, VecType, alias_argument, get_defaults
from ..coloring import colors
from ..config.settings import issue_warning, runtime_defaults
from ..geom.geom_utils import midpoint
from ..geom.nonlinear.ellipse import Arc, resolve_arc_sweep
from ..geom.points.point_utils import distance
from ..geom.segments.line_utils import line_angle
from ..group.batch import Group
from ..render.style_map import shape_style_map
from ..shapes.shape import Shape
from .illustration import Tag
from .utilities import get_transform

Color = colors.Color


_ANNOTATION_HEAD_KEYS = {
    "head_fill_alpha": "fill_alpha",
    "head_fill_color": "fill_color",
    "head_line_alpha": "line_alpha",
    "head_line_color": "line_color",
    "head_line_width": "line_width",
}
_ANNOTATION_SHAFT_KEYS = {
    "shaft_line_alpha": "line_alpha",
    "shaft_line_color": "line_color",
    "shaft_line_dash_array": "line_dash_array",
    "shaft_line_width": "line_width",
}
_ANNOTATION_TAG_KEYS = {
    "tag_bold": "bold",
    "tag_fill": "fill",
    "tag_fill_color": "fill_color",
    "tag_font_alpha": "font_alpha",
    "tag_font_color": "font_color",
    "tag_font_family": "font_family",
    "tag_font_size": "font_size",
    "tag_line_color": "line_color",
    "tag_line_width": "line_width",
    "tag_stroke": "stroke",
}


_LEADER_LINE_KEYS = (
    "line_alpha",
    "line_color",
    "line_dash_array",
    "line_width",
)


def _split_annotation_kwargs(
    draw_kwargs: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Separate annotation part styles from leader shape kwargs."""
    style_names = set(_ANNOTATION_HEAD_KEYS)
    style_names.update(_ANNOTATION_SHAFT_KEYS)
    style_names.update(_ANNOTATION_TAG_KEYS)
    style_names.update(_LEADER_LINE_KEYS)
    style_kwargs: dict[str, object] = {}
    leader_kwargs: dict[str, object] = {}
    for key, value in draw_kwargs.items():
        if key in style_names or key in ("alpha", "color"):
            style_kwargs[key] = value
        else:
            leader_kwargs[key] = value
    return style_kwargs, leader_kwargs


def _annotation_part(
    draw_kwargs: dict[str, object],
    names: dict[str, str],
) -> dict[str, object]:
    """Remove known part names and return them without the prefix."""
    part: dict[str, object] = {}
    for key, suffix in names.items():
        if key not in draw_kwargs:
            continue
        part[suffix] = draw_kwargs[key]
        del draw_kwargs[key]
    return part


def _stated_style(part: dict[str, object], name: str) -> tuple[bool, object]:
    """Return whether ``name`` was passed and is not the default marker."""
    if name not in part or part[name] is None:
        return False, None
    return True, part[name]


def _annotation_line_style(
    part: dict[str, object],
    color: object,
    alpha: object,
) -> dict[str, object]:
    """Line style for an annotation shaft and its landing."""
    style: dict[str, object] = {}
    stated, value = _stated_style(part, "line_color")
    if stated:
        style["line_color"] = value
    elif color is not None:
        style["line_color"] = color
    stated, value = _stated_style(part, "line_width")
    if stated:
        style["line_width"] = value
    stated, value = _stated_style(part, "line_dash_array")
    if stated:
        style["line_dash_array"] = value
    stated, value = _stated_style(part, "line_alpha")
    if stated:
        style["line_alpha"] = value
    elif alpha is not None:
        style["line_alpha"] = alpha
    return style


def _annotation_head_style(
    part: dict[str, object],
    color: object,
    alpha: object,
) -> dict[str, object]:
    """Arrowhead style for an annotation."""
    style: dict[str, object] = {}
    stated, value = _stated_style(part, "fill_color")
    if stated:
        style["fill_color"] = value
    elif color is not None:
        style["fill_color"] = color
    stated, value = _stated_style(part, "line_color")
    if stated:
        style["line_color"] = value
    elif color is not None:
        style["line_color"] = color
    stated, value = _stated_style(part, "line_width")
    if stated:
        style["line_width"] = value
    stated, value = _stated_style(part, "fill_alpha")
    if stated:
        style["fill_alpha"] = value
    elif alpha is not None:
        style["fill_alpha"] = alpha
    stated, value = _stated_style(part, "line_alpha")
    if stated:
        style["line_alpha"] = value
    elif alpha is not None:
        style["line_alpha"] = alpha
    return style


def _apply_annotation_style(shape: object, style: dict[str, object]) -> None:
    """Set one resolved style on an annotation part."""
    if shape is None:
        return
    for name, value in style.items():
        setattr(shape, name, value)


def _style_annotation(
    item: AnnotationArrow, draw_kwargs: dict[str, object]
) -> dict[str, object]:
    """Apply shaft, head, and tag styles. Return the remaining kwargs.

    ``draw_kwargs`` is copied. ``item`` is mutated.
    """
    kwargs = dict(draw_kwargs)
    color = kwargs["color"] if "color" in kwargs else None
    alpha = kwargs["alpha"] if "alpha" in kwargs else None
    if "color" in kwargs:
        del kwargs["color"]
    if "alpha" in kwargs:
        del kwargs["alpha"]
    leader_line: dict[str, object] = {}
    for name in _LEADER_LINE_KEYS:
        if name not in kwargs:
            continue
        leader_line[name] = kwargs[name]
        del kwargs[name]
    shaft_part = _annotation_part(kwargs, _ANNOTATION_SHAFT_KEYS)
    for name, value in leader_line.items():
        if name not in shaft_part:
            shaft_part[name] = value
    shaft_style = _annotation_line_style(shaft_part, color, alpha)
    head_style = _annotation_head_style(
        _annotation_part(kwargs, _ANNOTATION_HEAD_KEYS), color, alpha
    )
    tag_part = _annotation_part(kwargs, _ANNOTATION_TAG_KEYS)
    _apply_annotation_style(item.arrow.line, shaft_style)
    _apply_annotation_style(item.landing_line, shaft_style)
    for head in item.arrow.heads:
        _apply_annotation_style(head, head_style)
    stated, value = _stated_style(tag_part, "font_color")
    if stated:
        item.tag.font_color = value
    elif color is not None:
        item.tag.font_color = color
    stated, value = _stated_style(tag_part, "font_size")
    if stated:
        item.tag.font_size = value
    stated, value = _stated_style(tag_part, "font_family")
    if stated:
        item.tag.font_family = value
    stated, value = _stated_style(tag_part, "font_alpha")
    if stated:
        item.tag.font_alpha = value
    elif alpha is not None:
        item.tag.font_alpha = alpha
    stated, value = _stated_style(tag_part, "bold")
    if stated:
        item.tag.bold = value
    stated, value = _stated_style(tag_part, "fill")
    if stated:
        item.tag.fill = value
    stated, value = _stated_style(tag_part, "fill_color")
    if stated:
        item.tag.fill_color = value
    stated, value = _stated_style(tag_part, "stroke")
    if stated:
        item.tag.stroke = value
    stated, value = _stated_style(tag_part, "line_color")
    if stated:
        item.tag.line_color = value
    stated, value = _stated_style(tag_part, "line_width")
    if stated:
        item.tag.line_width = value
    return kwargs


# annotation is a label with a broken leader and an arrow
class AnnotationArrow(Group):
    """A leader from a feature point to text or a circled number.

    The leader is two segments: an angled shaft from ``tip`` to ``elbow``,
    then a landing from ``elbow`` to ``landing``. The arrowhead sits at
    ``tip``. If only ``landing`` or only ``elbow`` is given, the missing
    point is inferred so the landing is horizontal.

    Args:
        tip (PointType): Feature point the arrowhead points at.
        text (str | int): Label text, or a number for a balloon. Defaults
            to "".
        landing (PointType, optional): End of the landing, at the label.
            Defaults to None.
        elbow (PointType, optional): Break between the angled shaft and
            the landing. Defaults to None.
        landing_length (float, optional): Horizontal landing length used
            when inferring ``elbow`` or ``landing``. Defaults to None
            (``runtime_defaults["landing_length"]``).
        circled (bool, optional): If True, the label is a balloon
            (circled number or text). Defaults to False.
        font_size (float, optional): Label font size. Defaults to None
            (``runtime_defaults["font_size"]``).
        **kwargs: ``shaft_line_alpha``, ``shaft_line_color``,
            ``shaft_line_dash_array``, and ``shaft_line_width`` style
            the angled shaft and the landing. ``head_fill_alpha``,
            ``head_fill_color``, ``head_line_alpha``,
            ``head_line_color``, and ``head_line_width`` style the
            arrowhead. ``tag_bold``, ``tag_fill``, ``tag_fill_color``,
            ``tag_font_alpha``, ``tag_font_color``, ``tag_font_family``,
            ``tag_font_size``, ``tag_line_color``, ``tag_line_width``,
            and ``tag_stroke`` style the label. ``color`` and ``alpha``
            set every part; a prefixed name wins. ``tag_stroke`` is
            left unchanged unless given. Other shape kwargs, such as
            ``line_color``, go to the leader ``Arrow`` and the landing
            ``Shape``.

    Examples:
        >>> import simetri.graphics as sg
        >>> note = sg.AnnotationArrow((0, 0), "A", landing=(40, 20))
        >>> note.tip
        (0, 0)
        >>> note.elbow
        (20, 20)
        >>> note.landing
        (40, 20)
        >>> balloon = sg.AnnotationArrow(
        ...     (0, 0), 1, landing=(40, 20), circled=True
        ... )
        >>> balloon.text
        '1'
        >>> balloon.tag.frame_shape == sg.FrameShape.CIRCLE
        True
        >>> balloon.tag.stroke
        True
        >>> note.tag.stroke
        False
        >>> styled = sg.AnnotationArrow(
        ...     (0, 0),
        ...     "A",
        ...     landing=(40, 20),
        ...     head_fill_color=sg.red,
        ...     shaft_line_color=sg.blue,
        ...     tag_font_color=sg.green,
        ... )
        >>> styled.arrow.line.line_color == sg.blue
        True
        >>> styled.landing_line.line_color == sg.blue
        True
        >>> styled.arrow.heads[0].fill_color == sg.red
        True
        >>> styled.tag.font_color == sg.green
        True
        >>> sg.AnnotationArrow((0, 0), "A")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: AnnotationArrow requires landing or elbow.
    """

    def __init__(
        self,
        tip: PointType,
        text: str | int = "",
        landing: PointType | None = None,
        elbow: PointType | None = None,
        landing_length: float | None = None,
        circled: bool = False,
        font_size: float | None = None,
        **kwargs: object,
    ) -> None:
        """Create a broken-leader annotation arrow.

        See the class docstring for argument details.

        Examples:
            >>> import simetri.graphics as sg
            >>> note = sg.AnnotationArrow((0, 0), "A", landing=(40, 20))
            >>> note.tip
            (0, 0)
            >>> note.elbow
            (20, 20)
            >>> note.landing
            (40, 20)
        """
        if elbow is None and landing is None:
            raise ValueError("AnnotationArrow requires landing or elbow.")
        if elbow is not None and landing is not None:
            if landing_length is not None:
                raise ValueError(
                    "Do not pass landing_length when both elbow "
                    "and landing are given."
                )
        else:
            (landing_length,) = get_defaults(
                ["landing_length"], [landing_length]
            )

        (font_size,) = get_defaults(["font_size"], [font_size])
        tip_x, tip_y = tip[:2]
        self.tip = (tip_x, tip_y)
        self.circled = circled
        self.font_size = font_size
        self.text = str(text)

        if elbow is None:
            landing_x, landing_y = landing[:2]
            if landing_x >= tip_x:
                elbow_x = landing_x - landing_length
            else:
                elbow_x = landing_x + landing_length
            elbow_y = landing_y
            self.elbow = (elbow_x, elbow_y)
            self.landing = (landing_x, landing_y)
        elif landing is None:
            elbow_x, elbow_y = elbow[:2]
            if elbow_x >= tip_x:
                landing_x = elbow_x + landing_length
            else:
                landing_x = elbow_x - landing_length
            landing_y = elbow_y
            self.elbow = (elbow_x, elbow_y)
            self.landing = (landing_x, landing_y)
        else:
            elbow_x, elbow_y = elbow[:2]
            landing_x, landing_y = landing[:2]
            self.elbow = (elbow_x, elbow_y)
            self.landing = (landing_x, landing_y)

        style_kwargs, leader_kwargs = _split_annotation_kwargs(kwargs)
        self.arrow = Arrow(self.elbow, self.tip, **leader_kwargs)
        items = [self.arrow]
        abs_tol = runtime_defaults["abs_tol"]
        landing_span = distance(self.elbow, self.landing)
        if landing_span > abs_tol:
            self.landing_line = Shape(
                [self.elbow, self.landing], fill=False, **leader_kwargs
            )
            items.append(self.landing_line)
        else:
            self.landing_line = None

        land_dx = landing_x - elbow_x
        land_dy = landing_y - elbow_y
        land_len = hypot(land_dx, land_dy)
        if land_len > abs_tol:
            unit_x = land_dx / land_len
            unit_y = land_dy / land_len
        elif landing_x >= tip_x:
            unit_x, unit_y = 1.0, 0.0
        else:
            unit_x, unit_y = -1.0, 0.0

        text_gap = runtime_defaults["text_offset"]
        if circled:
            tag_x, tag_y = landing_x, landing_y
            tag_align = Align.CENTER
        else:
            tag_x = landing_x + unit_x * text_gap
            tag_y = landing_y + unit_y * text_gap
            if landing_x >= tip_x:
                tag_align = Align.LEFT
            else:
                tag_align = Align.RIGHT
        self.tag = Tag(
            self.text,
            (tag_x, tag_y),
            font_size=font_size,
            align=tag_align,
        )
        if circled:
            self.tag.frame_shape = FrameShape.CIRCLE
            self.tag.stroke = True
            self.tag.fill = True
            self.tag.fill_color = colors.white
        _style_annotation(self, style_kwargs)
        items.append(self.tag)
        super().__init__(items, subtype=Types.ANNOTATION)


class ArrowHead(Shape):
    """An ArrowHead object is a shape that represents the head of an arrow.

    Args:
        length (float, optional): The length of the arrow head. Defaults to None.
        width_ (float, optional): The width of the arrow head. Defaults to None.
        points (list, optional): The points defining the arrow head. Defaults to None.
        **kwargs: Additional keyword arguments for arrow head styling.

    Examples:
        >>> import simetri.graphics as sg
        >>> head = sg.ArrowHead(length=80, width_=3)
        >>> head.vertices
        ((0.0, 0.0), (0.0, -1.5), (80.0, 0.0), (0.0, 1.5))
        >>> head.head_length
        80
        >>> head.head_width
        3
    """

    def __init__(
        self,
        length: float | None = None,
        width_: float | None = None,
        points: list | None = None,
        **kwargs: object,
    ) -> None:
        """Create an arrow head shape.

        See the class docstring for argument details.
        """
        length, width_ = get_defaults(
            ["arrow_head_length", "arrow_head_width"], [length, width_]
        )
        if points is None:
            w2 = width_ / 2
            points = [(0, 0), (0, -w2), (length, 0), (0, w2)]
        super().__init__(
            points, closed=True, subtype=Types.ARROW_HEAD, **kwargs
        )
        self.head_length = length
        self.head_width = width_

        self.kwargs = kwargs

    def __repr__(self) -> str:
        """Return an ArrowHead string from this head's vertices.

        Examples:
            >>> import simetri.graphics as sg
            >>> repr(sg.ArrowHead(length=80, width_=3))
            'ArrowHead([(0.0, 0.0), ..., (0.0, 1.5)])'
            >>> str(sg.ArrowHead(length=80, width_=3)).startswith("Shape")
            True
        """
        if len(self.primary_points) == 0:
            return "ArrowHead()"
        if len(self.primary_points) < 4:
            return f"ArrowHead({self.vertices})"
        return f"ArrowHead([{self.vertices[0]}, ..., {self.vertices[-1]}])"


def arrow(
    p1: PointType,
    p2: PointType,
    head_length: float = 10,
    head_width: float = 4,
    line_width: float = 1,
    line_color: Color = colors.black,
    fill_color: Color = colors.black,
    centered: bool = False,
) -> Group:
    """Return an arrow from p1 to p2.

    Args:
        p1 (tuple): The starting point of the arrow.
        p2 (tuple): The ending point of the arrow.
        head_length (int, optional): The length of the arrow head. Defaults to 10.
        head_width (int, optional): The width of the arrow head. Defaults to 4.
        line_width (int, optional): The width of the arrow line. Defaults to 1.
        line_color (Color, optional): The color of the arrow line. Defaults to colors.black.
        fill_color (Color, optional): The fill color of the arrow head. Defaults to colors.black.
        centered (bool, optional): Whether the arrow is centered. Defaults to False.

    Returns:
        Group: A Group object containing the arrow shapes.

    Examples:
        >>> import simetri.graphics as sg
        >>> shaft = sg.arrow((0, 0), (40, 0))
        >>> len(shaft)
        2
        >>> shaft[0].vertices
        ((0.0, 0.0), (40.0, 0.0))
        >>> shaft[1].vertices
        ((30.0, 2.0), (40.0, 0.0), (30.0, -2.0))
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    dx = x2 - x1
    dy = y2 - y1
    angle = atan2(dy, dx)
    body = Shape(
        [(x1, y1), (x2, y2)],
        closed=False,
        line_color=line_color,
        fill_color=fill_color,
        line_width=line_width,
    )
    w2 = head_width / 2
    head = Shape(
        [(-head_length, w2), (0, 0), (-head_length, -w2)],
        closed=True,
        line_color=line_color,
        fill_color=fill_color,
        line_width=line_width,
    )
    head.rotate(angle)
    if centered:
        head.translate(*midpoint((x1, y1), (x2, y2)))
    else:
        head.translate(x2, y2)
    return Group([body, head])



def draw_cs_small(
    canvas: Any,
    pos: PointType = (0, 0),
    width: float = 80,
    height: float = 100,
    neg_width: float = 5,
    neg_height: float = 5,
) -> None:
    """Draws a small coordinate system.

    Args:
        canvas: The canvas to draw on.
        pos (tuple, optional): The position of the coordinate system. Defaults to (0, 0).
        width (int, optional): The length of the x-axis. Defaults to 80.
        height (int, optional): The length of the y-axis. Defaults to 100.
        neg_width (int, optional): The negative length of the x-axis. Defaults to 5.
        neg_height (int, optional): The negative length of the y-axis. Defaults to 5.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> sg.draw_cs_small(canvas, (0, 0))
        >>> [sketch.subtype.name for sketch in canvas.active_page.sketches]
        ['SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH']
    """
    x, y = pos[:2]
    x_axis = arrow(
        (-neg_width + x, y), (width + 10 + x, y), head_length=8, head_width=2
    )
    y_axis = arrow(
        (x, -neg_height + y), (x, height + 10 + y), head_length=8, head_width=2
    )
    canvas.draw(x_axis, line_width=1)
    canvas.draw(y_axis, line_width=1)


class ArcArrow(Group):
    """An ArcArrow object is an arrow with an arc.

    Args:
        center: The center of the arc.
        radius_x: Semi-axis along x.
        radius_y: Semi-axis along y; defaults to ``radius_x``.
        start_angle: The starting angle of the arc in radians. Defaults to 0.
        span_angle: Sweep in radians. A negative value draws clockwise.
            Mutually exclusive with ``end_angle``. At least one of
            ``span_angle`` or ``end_angle`` is required.
        end_angle: Ending angle in radians. Mutually exclusive with
            ``span_angle``.
        clockwise: If True, the arc is drawn clockwise. Defaults to False.
        xform_matrix: The transformation matrix. Defaults to None.
        **kwargs: Additional keyword arguments for arc arrow styling.

    Examples:
        >>> import simetri.graphics as sg
        >>> arrow = sg.ArcArrow((0, 0), 20, start_angle=0, end_angle=sg.pi / 2)
        >>> arrow.radius_x
        20
        >>> round(arrow.end_angle, 6)
        1.570796
    """

    @alias_argument({"radius_x": ["radius", "r", "rx"], "radius_y": "ry"})
    def __init__(
        self,
        center: PointType,
        radius_x: float,
        radius_y: float | None = None,
        start_angle: float = 0,
        span_angle: float | None = None,
        end_angle: float | None = None,
        clockwise: bool = False,
        xform_matrix: NDArray | None = None,
        **kwargs: object,
    ) -> None:
        """Create an arc with arrow heads at both ends.

        See the class docstring for argument details.

        Raises:
            AttributeError: If an invalid style keyword is provided.
        """
        if radius_y is None:
            radius_y = radius_x
        signed_span = resolve_arc_sweep(
            start_angle, span_angle, end_angle, clockwise
        )
        clockwise = signed_span < 0
        if end_angle is None:
            end_angle = start_angle + signed_span
        self.center = center
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.start_angle = start_angle
        self.span_angle = abs(signed_span)
        self.end_angle = end_angle
        self.clockwise = clockwise
        # create the arc
        self.arc = Arc(
            center,
            radius_x,
            radius_y,
            start_angle=start_angle,
            span_angle=self.span_angle,
            clockwise=clockwise,
        )
        self.arc.fill = False
        # create arrow_head1
        self.arrow_head1 = ArrowHead()
        # create arrow_head2
        self.arrow_head2 = ArrowHead()
        start = self.arc[0]
        end = self.arc[-1]
        self.points = [center, start, end]

        self.arrow_head1.translate(-1 * self.arrow_head1.head_length, 0)
        self.arrow_head1.rotate(start_angle - pi / 2)
        self.arrow_head1.translate(*start)
        self.arrow_head2.translate(-1 * self.arrow_head2.head_length, 0)
        self.arrow_head2.rotate(end_angle + pi / 2)
        self.arrow_head2.translate(*end)
        items = [self.arc, self.arrow_head1, self.arrow_head2]
        super().__init__(items, subtype=Types.ARC_ARROW, **kwargs)
        for k, v in kwargs.items():
            if k in shape_style_map:
                setattr(self, k, v)  # we should check for valid values here
            else:
                raise AttributeError(f"{k}. Invalid attribute!")
        self.xform_matrix = get_transform(xform_matrix)

    def __repr__(self) -> str:
        """Return an ArcArrow string from this arrow's elements.

        Examples:
            >>> import simetri.graphics as sg
            >>> arrow = sg.ArcArrow((0, 0), 20, start_angle=0, end_angle=sg.pi / 2)
            >>> repr(arrow).startswith("ArcArrow(")
            True
            >>> str(arrow).startswith("Group")
            True
        """
        if self.elements is None or len(self.elements) == 0:
            return "ArcArrow()"
        if len(self.elements) in [1, 2]:
            return f"ArcArrow({self.elements})"
        return f"ArcArrow({self.elements[0]}...{self.elements[-1]})"


class Arrow(Group):
    """An Arrow object is a line with an arrow head.

    Args:
        p1 (PointType): The starting point of the arrow.
        p2 (PointType): The ending point of the arrow.
        head_pos (HeadPos, optional): The position of the arrow head. Defaults to HeadPos.END.
        head (Shape, optional): The shape of the arrow head. Defaults to None.
        **kwargs: Additional keyword arguments for arrow styling.

    Examples:
        >>> import simetri.graphics as sg
        >>> shaft = sg.Arrow((0, 0), (40, 0))
        >>> shaft.p1
        (0, 0)
        >>> shaft.p2
        (40, 0)
        >>> shaft.line.vertices
        ((0.0, 0.0), (40.0, 0.0))
    """

    def __init__(
        self,
        p1: PointType,
        p2: PointType,
        head_pos: HeadPos = HeadPos.END,
        head: Shape | None = None,
        line_width: float = 1,
        color: Color = colors.black,
        **kwargs: object,
    ) -> None:
        """Create a line arrow with one or more heads.

        See the class docstring for argument details.
        """
        self.p1 = p1
        self.p2 = p2
        self.head_pos = head_pos
        self.head = head
        self.line_width = line_width
        self.color = color
        self.kwargs = kwargs
        length = distance(p1, p2)
        angle = line_angle(p1, p2)
        self.line = Shape(
            [(0, 0), (length, 0)],
            line_width=line_width,
            line_color=self.color,
            **kwargs,
        )
        if head is None:
            self.head = ArrowHead()
            self.head.fill_color = color
            self.head.line_color = color
        else:
            self.head = head
        if self.head_pos == HeadPos.END:
            x = length
            self.head.translate(x - self.head.head_length, 0)
            self.head.rotate(angle)
            self.line.rotate(angle)
            self.line.translate(*p1)
            self.head.translate(*p1)
            self.heads = [self.head]
        elif self.head_pos == HeadPos.START:
            self.head.rotate(pi)
            self.head.translate(self.head.head_length, 0)
            self.head.rotate(angle)
            self.line.rotate(angle)
            self.line.translate(*p1)
            self.head.translate(*p1)
            self.heads = [self.head]
        elif self.head_pos == HeadPos.BOTH:
            self.head2 = ArrowHead()
            self.head2.rotate(pi)
            self.head2.translate(self.head2.head_length, 0)
            self.head2.rotate(angle)
            self.head2.translate(*p1)
            x = length
            self.head.translate(x - self.head.head_length, 0)
            self.head.rotate(angle)
            self.line.rotate(angle)
            self.line.translate(*p1)
            self.head.translate(*p1)
            self.heads = [self.head, self.head2]
        elif self.head_pos == HeadPos.NONE:
            self.heads = [None]

        items = [self.line] + self.heads
        super().__init__(items, subtype=Types.ARROW, **kwargs)

    def __repr__(self) -> str:
        """Return an Arrow string from this arrow's elements.

        Examples:
            >>> import simetri.graphics as sg
            >>> arrow = sg.Arrow((0, 0), (40, 0))
            >>> repr(arrow).startswith("Arrow(")
            True
            >>> str(arrow).startswith("Group")
            True
        """
        if self.elements is None or len(self.elements) == 0:
            return "Arrow()"
        if len(self.elements) in [1, 2]:
            return f"Arrow({self.elements})"
        return f"Arrow({self.elements[0]}...{self.elements[-1]})"


def vec_arrow(
    vec: VecType,
    *,
    start: PointType | None = None,
    end: PointType | None = None,
    **kwargs: object,
) -> Arrow:
    """Return an ``Arrow`` from a displacement and a start or end point.

    ``vec`` is a displacement. It is not stored on the ``Arrow``. Extra
    keyword arguments are passed to ``Arrow``.

    Args:
        vec (VecType): Displacement from tail to tip.
        start (PointType, optional): Tail position. Defaults to None.
        end (PointType, optional): Tip position. Defaults to None.
        **kwargs: Passed to ``Arrow``.

    Returns:
        Arrow: Arrow from ``start`` to ``start + vec``, or from
        ``end - vec`` to ``end``.

    Raises:
        ValueError: If ``vec`` is zero, if neither ``start`` nor ``end``
            is given, or if both are given and ``end - start`` does not
            match ``vec``.

    Examples:
        >>> import simetri.graphics as sg
        >>> arrow = sg.vec_arrow(sg.Vector(60, 80), start=(40, 20))
        >>> arrow.p1
        (40, 20)
        >>> arrow.p2
        (100, 100)
        >>> arrow = sg.vec_arrow(sg.Vector(60, 80), end=(13, 24))
        >>> arrow.p1
        (-47, -56)
        >>> arrow.p2
        (13, 24)
        >>> sg.vec_arrow(sg.Vector(0, 0), start=(40, 40))
        Traceback (most recent call last):
        ValueError: Cannot create an Arrow from a zero-length Vector.
        >>> sg.vec_arrow(sg.Vector(60, 80), start=(0, 0), end=(40, 0))
        Traceback (most recent call last):
        ValueError: start and end are not consistent with the Vector displacement (60, 80).
    """
    vector_x, vector_y = vec[:2]
    if vector_x == 0 and vector_y == 0:
        raise ValueError("Cannot create an Arrow from a zero-length Vector.")
    if start is None and end is None:
        raise ValueError("vec_arrow requires start or end.")
    if start is not None and end is not None:
        start_x, start_y = start[:2]
        end_x, end_y = end[:2]
        displacement_x = end_x - start_x
        displacement_y = end_y - start_y
        offset = hypot(displacement_x - vector_x, displacement_y - vector_y)
        if offset <= runtime_defaults["abs_tol"]:
            issue_warning(
                f"Duplicate position used for Vector({start}, {end}).",
                warning_type=WarningType.vector.duplicate,
            )
            return Arrow((start_x, start_y), (end_x, end_y), **kwargs)
        raise ValueError(
            "start and end are not consistent with the Vector "
            f"displacement {(vector_x, vector_y)}."
        )
    if start is not None:
        start_x, start_y = start[:2]
        return Arrow(
            (start_x, start_y),
            (start_x + vector_x, start_y + vector_y),
            **kwargs,
        )
    end_x, end_y = end[:2]
    return Arrow(
        (end_x - vector_x, end_y - vector_y),
        (end_x, end_y),
        **kwargs,
    )
