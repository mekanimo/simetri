"""Dimensioning related objects."""

from __future__ import annotations

from math import cos, hypot, pi, sin

from ..base.all_enums import Align, Anchor, HeadPos, Types
from ..base.common import PointType, get_defaults
from ..config.settings import runtime_defaults
from ..geom.geom_utils import midpoint
from ..geom.geometry import polar_to_cartesian
from ..geom.points.point_utils import distance
from ..geom.segments.line_utils import (
    extended_line,
    line_angle,
    line_by_point_angle_length,
)
from ..group.batch import Group
from ..shapes.geom_items import Line
from .arrows import ArcArrow, Arrow
from .illustration import Tag


def _format_dim_value(value: float) -> str:
    """Round a measured dimension value for its label.

    Uses ``runtime_defaults["n_dim_digits"]``.
    """
    ndigits = runtime_defaults["n_dim_digits"]
    if ndigits < 0:
        raise ValueError("n_dim_digits must be a nonnegative integer.")
    return str(round(float(value), ndigits))


class RadialDimension(Group):
    """A RadialDimension object is a dimension that represents a radius.

    Args:
        center (PointType): The center of the circle.
        radius (float): The radius of the circle.
        angle (float): The angle of the dimension line.
        text_offset (float, optional): Offset for the dimension text.
            ``None`` uses ``runtime_defaults["text_offset"]``.
        gap (float, optional): The gap between the dimension line and the text.
            ``None`` uses ``runtime_defaults["gap"]``.
        text (str, optional): Label text. Empty string uses the radius
            rounded to ``runtime_defaults["n_dim_digits"]``.
        ext_length (float, optional): Extension length when ``reverse_arrow``
            is True. ``None`` uses ``runtime_defaults["rev_arrow_length"]``.
        reverse_arrow (bool, optional): If True, an extension line is created.
            Defaults to False.
        keep_inside (bool, optional): Defaults to True.
        **kwargs: Additional keyword arguments for radial dimension styling.

    Examples:
        >>> import simetri.graphics as sg
        >>> dim = sg.RadialDimension((0, 0), 10)
        >>> dim.text
        '10.0'
        >>> dim.center
        (0, 0)
        >>> dim.radius
        10
        >>> tuple(dim.tag.pos)
        (5.0, 0.0)
        >>> len(dim)
        2
    """

    def __init__(
        self,
        center: PointType,
        radius: float | None = None,
        angle: float = 0,
        text: str = "",
        text_offset: float | None = None,
        ext_length: float | None = None,
        reverse_arrow: bool = False,
        keep_inside: bool = True,
        gap: float | None = None,
        **kwargs: object,
    ) -> None:
        """Create a radial dimension annotation.

        See the class docstring for argument details.
        """
        text_offset, gap, ext_length = get_defaults(
            ["text_offset", "gap", "rev_arrow_length"],
            [text_offset, gap, ext_length],
        )
        self.center = center
        self.radius = radius
        self.angle = angle
        self.text = text
        self.text_offset = text_offset
        self.ext_length = ext_length
        self.reverse_arrow = reverse_arrow
        self.keep_inside = keep_inside
        self.gap = gap
        super().__init__(subtype=Types.RADIAL_DIMENSION, **kwargs)

        p2 = polar_to_cartesian(self.radius, self.angle, self.center)

        self.arrow = Arrow(center, p2)
        self.extension = None
        if self.reverse_arrow:
            self.extension = extended_line(ext_length, [center, p2])

        if self.text == "":
            if self.radius is not None:
                self.text = _format_dim_value(self.radius)
            else:
                self.text = f"r = {_format_dim_value(distance(center, p2))}"

        self.tag = Tag(self.text, midpoint(center, p2))
        self._items = [self.arrow, self.tag]

        super().__init__(self._items, subtype=Types.RADIAL_DIMENSION, **kwargs)

    def __repr__(self) -> str:
        """Return a RadialDimension string from this dimension's elements.

        Examples:
            >>> import simetri.graphics as sg
            >>> dim = sg.RadialDimension((0, 0), 10)
            >>> repr(dim).startswith("RadialDimension(")
            True
            >>> str(dim).startswith("Group")
            True
        """
        if self.elements is None or len(self.elements) == 0:
            return "RadialDimension()"
        if len(self.elements) in [1, 2]:
            return f"RadialDimension({self.elements})"
        return f"RadialDimension({self.elements[0]}...{self.elements[-1]})"
class AngularDimension(Group):
    """An angle dimension: two extension lines, an arc, and a label.

    The arc runs counter-clockwise from ``start_angle`` to
    ``end_angle`` at ``radius``. ``gap_angle`` shortens the arc at
    each end. The extension lines lie on the two sides, from ``gap``
    out to ``radius * (1 + ext_angle)``. The label is the full sweep
    in radians, outside the arc by ``text_offset``.

    Args:
        center (PointType): The vertex of the angle.
        radius (float): Radius of the dimension arc.
        start_angle (float): First side, in radians.
        end_angle (float): Second side, in radians. The sweep is
            counter-clockwise from ``start_angle`` to ``end_angle``.
        ext_angle (float, optional): Extra radius of each extension past
            the arc, as a fraction of ``radius`` via ``1 + ext_angle``.
            ``None`` uses ``runtime_defaults["ext_angle"]``.
        gap_angle (float, optional): Angle cut from each end of the
            dimension arc. ``None`` uses ``runtime_defaults["gap_angle"]``.
        text_offset (float, optional): Distance from the arc to the
            label. ``None`` uses ``runtime_defaults["text_offset"]``.
        gap (float, optional): Distance from the vertex to the start
            of each extension. ``None`` uses ``runtime_defaults["gap"]``.
        text (str, optional): Label text. ``None`` uses the sweep in
            radians, rounded to ``runtime_defaults["n_dim_digits"]``.
            Defaults to None.
        **kwargs: Additional keyword arguments for dimension styling.

    Examples:
        >>> import simetri.graphics as sg
        >>> dim = sg.AngularDimension((0, 0), 20, 0, sg.pi / 2, 0.1, 0.1)
        >>> dim.center
        (0, 0)
        >>> dim.radius
        20
        >>> dim.start_angle
        0
        >>> dim.end_angle
        1.5707963267948966
        >>> dim.text
        '1.57'
        >>> len(dim)
        3
        >>> bare = sg.AngularDimension((0, 0), 20, 0, sg.pi / 2)
        >>> bare.ext_angle
        0
        >>> bare.gap_angle
        0
        >>> bare.text_offset
        5
    """

    def __init__(
        self,
        center: PointType,
        radius: float,
        start_angle: float,
        end_angle: float,
        ext_angle: float | None = None,
        gap_angle: float | None = None,
        text_offset: float | None = None,
        gap: float | None = None,
        text: str | None = None,
        **kwargs: object,
    ) -> None:
        """Create an angular dimension with extension lines and an arc.

        See the class docstring for argument details.
        """
        text_offset, gap, ext_angle, gap_angle, font_size = get_defaults(
            ["text_offset", "gap", "ext_angle", "gap_angle", "font_size"],
            [text_offset, gap, ext_angle, gap_angle, None],
        )
        if radius <= 0:
            raise ValueError("AngularDimension radius must be positive.")
        if ext_angle < 0:
            raise ValueError("AngularDimension ext_angle must be non-negative.")
        if gap_angle < 0:
            raise ValueError("AngularDimension gap_angle must be non-negative.")
        if gap < 0:
            raise ValueError("AngularDimension gap must be non-negative.")
        if gap >= radius:
            raise ValueError("AngularDimension gap must be less than radius.")
        abs_tol = runtime_defaults["abs_tol"]
        sweep = (end_angle - start_angle) % (2 * pi)
        if sweep < abs_tol:
            raise ValueError("AngularDimension angle must be nonzero.")
        if sweep - 2 * gap_angle <= abs_tol:
            raise ValueError(
                "AngularDimension gap_angle removes the dimension arc."
            )
        if text is None:
            text = _format_dim_value(sweep)
        center_x, center_y = center[:2]
        arc_start = start_angle + gap_angle
        arc_end = start_angle + sweep - gap_angle
        mid_angle = start_angle + sweep / 2
        self.text_pos = polar_to_cartesian(
            radius + text_offset, mid_angle, (center_x, center_y)
        )

        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.ext_angle = ext_angle
        self.gap_angle = gap_angle
        self.text_offset = text_offset
        self.gap = gap
        self.text = text
        self.font_size = font_size
        self.kwargs = kwargs
        self.text_anchor = Anchor.CENTER
        self.text_align = Align.CENTER
        self.ext3 = None
        self.arrow1 = None
        self.arrow2 = None
        self.midline = None

        outer = radius * (1.0 + ext_angle)
        self.ext1 = Line(
            polar_to_cartesian(gap, start_angle, (center_x, center_y)),
            polar_to_cartesian(outer, start_angle, (center_x, center_y)),
        )
        self.ext2 = Line(
            polar_to_cartesian(gap, end_angle, (center_x, center_y)),
            polar_to_cartesian(outer, end_angle, (center_x, center_y)),
        )
        self.dim_line = ArcArrow(
            (center_x, center_y),
            radius,
            start_angle=arc_start,
            end_angle=arc_end,
        )
        self.dim_line.line = self.dim_line.arc
        self.dim_line.heads = [
            self.dim_line.arrow_head1,
            self.dim_line.arrow_head2,
        ]
        self.tag = Tag(
            text,
            pos=self.text_pos,
            fill=True,
            anchor=self.text_anchor,
            align=self.text_align,
            font_size=self.font_size,
        )
        super().__init__(
            [self.ext1, self.ext2, self.dim_line],
            subtype=Types.ANGULAR_DIMENSION,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return an AngularDimension string from this dimension's elements.

        Examples:
            >>> import simetri.graphics as sg
            >>> dim = sg.AngularDimension((0, 0), 20, 0, sg.pi / 2, 0.1, 0.1)
            >>> repr(dim).startswith("AngularDimension(")
            True
            >>> str(dim).startswith("Group")
            True
        """
        if self.elements is None or len(self.elements) == 0:
            return "AngularDimension()"
        if len(self.elements) in [1, 2]:
            return f"AngularDimension({self.elements})"
        return f"AngularDimension({self.elements[0]}...{self.elements[-1]})"


class Dimension(Group):
    """A linear dimension: extension lines, dimension line, and a label.

    ``p1`` and ``p2`` decide the kind: same ``y`` is horizontal, same
    ``x`` is vertical, otherwise the dimension is aligned to the
    segment (diagonal). The label position is stored on ``text_pos``.

    Args:
        p1 (PointType): First feature point.
        p2 (PointType): Second feature point.
        side (str): Which side of the features the dimension line is
            on. Horizontal or diagonal: ``"up"`` or ``"down"``.
            Vertical: ``"left"`` or ``"right"``.
        text_offset (float, optional): Distance from the gap to the
            dimension line. ``None`` uses ``runtime_defaults["text_offset"]``.
        text (str, optional): Label text. ``None`` uses the measured
            length, rounded to ``runtime_defaults["n_dim_digits"]``.
            Defaults to None.
        ext_line_extension (float, optional): How far each extension
            continues past the dimension line. ``None`` uses
            ``runtime_defaults["overshoot"]``.
        ext_line_offset (float, optional): Gap from the feature to the
            start of the extension. ``None`` uses ``runtime_defaults["gap"]``.
        text_horiz_offset (float, optional): Offset of the label along
            the dimension line when ``text_loc`` is ``"left"`` or
            ``"right"``. ``None`` uses ``runtime_defaults["ext_length2"]``.
        text_loc (str, optional): ``"middle"``, ``"left"``, or
            ``"right"``. Defaults to ``"middle"``.
        stub_length (float, optional): Outward shaft length when the
            label is not in the middle. ``None`` uses
            ``runtime_defaults["stub_length"]``.
        show_midline (bool, optional): Draw the line between the side
            arrows. Defaults to True.
        **kwargs: Additional keyword arguments for dimension styling.

    Examples:
        >>> import simetri.graphics as sg
        >>> dim = sg.Dimension((0, 0), (40, 0), 'up', 8)
        >>> dim.text
        '40.0'
        >>> sg.Dimension((0, 0), (1, 1), "up", 8).text
        '1.41'
        >>> dim.text_pos
        (20.0, 13)
        >>> dim.p1
        (0, 0)
        >>> dim.p2
        (40, 0)
        >>> len(dim)
        3
        >>> hidden = sg.Dimension(
        ...     (0, 0), (40, 0), "up", 8, text_loc="left", show_midline=False
        ... )
        >>> hidden.midline is not None
        True
        >>> hidden.midline in hidden.elements
        False
        >>> plain = sg.Dimension((0, 0), (40, 0), "up")
        >>> plain.text_offset
        5
        >>> plain.stub_length
        15
        >>> plain.text_pos
        (20.0, 10)
    """

    def __init__(
        self,
        p1: PointType,
        p2: PointType,
        side: str,
        text_offset: float | None = None,
        text: str | None = None,
        ext_line_extension: float | None = None,
        ext_line_offset: float | None = None,
        text_horiz_offset: float | None = None,
        text_loc: str = "middle",
        stub_length: float | None = None,
        show_midline: bool = True,
        **kwargs: object,
    ) -> None:
        """Create a linear dimension with extension lines and arrows.

        See the class docstring for argument details.
        """
        (
            text_offset,
            ext_line_extension,
            ext_line_offset,
            text_horiz_offset,
            stub_length,
            font_size,
        ) = get_defaults(
            [
                "text_offset",
                "overshoot",
                "gap",
                "ext_length2",
                "stub_length",
                "font_size",
            ],
            [
                text_offset,
                ext_line_extension,
                ext_line_offset,
                text_horiz_offset,
                stub_length,
                None,
            ],
        )
        if text is None:
            text = _format_dim_value(distance(p1, p2))

        self.p1 = p1
        self.p2 = p2
        self.side = side
        self.text_offset = text_offset
        self.text = text
        self.ext_line_extension = ext_line_extension
        self.ext_line_offset = ext_line_offset
        self.text_horiz_offset = text_horiz_offset
        self.text_loc = text_loc
        self.stub_length = stub_length
        self.show_midline = show_midline
        self.font_size = font_size
        self.kwargs = kwargs
        self.ext1 = None
        self.ext2 = None
        self.ext3 = None
        self.arrow1 = None
        self.arrow2 = None
        self.dim_line = None
        self.midline = None
        self.tag = None
        self.text_anchor = Anchor.CENTER
        self.text_align = Align.CENTER

        super().__init__(subtype=Types.DIMENSION, **kwargs)

        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        abs_tol = runtime_defaults["abs_tol"]
        if abs(x1 - x2) < abs_tol and abs(y1 - y2) < abs_tol:
            raise ValueError("Dimension points must be distinct.")

        if abs(y1 - y2) < abs_tol:
            dim_x1 = x1
            dim_x2 = x2
            if text_loc == "middle":
                text_x = (x1 + x2) / 2
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.CENTER
            elif text_loc == "left":
                text_x = x1 - text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.RIGHT
            else:
                text_x = x2 + text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.LEFT
            if side == "up":
                text_y = y1 + ext_line_offset + text_offset
                y_p1_start = y1 + ext_line_offset
                y_p1_end = text_y + ext_line_extension
                y_p2_start = y2 + ext_line_offset
                y_p2_end = (
                    y2 + ext_line_offset + text_offset + ext_line_extension
                )
            else:
                text_y = y1 - ext_line_offset - text_offset
                y_p1_start = y1 - ext_line_offset
                y_p1_end = (
                    y1 - ext_line_offset - text_offset - ext_line_extension
                )
                y_p2_start = y2 - ext_line_offset
                y_p2_end = (
                    y2 - ext_line_offset - text_offset - ext_line_extension
                )
            ext1_start = (x1, y_p1_start)
            ext1_end = (x1, y_p1_end)
            ext2_start = (x2, y_p2_start)
            ext2_end = (x2, y_p2_end)
            dim1 = (dim_x1, text_y)
            dim2 = (dim_x2, text_y)
            stub1_tail = (dim_x1 - stub_length, text_y)
            stub2_tip = (dim_x2 + stub_length, text_y)
        elif abs(x1 - x2) < abs_tol:
            dim_y1 = y1
            dim_y2 = y2
            if text_loc == "middle":
                text_y = (y1 + y2) / 2
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.CENTER
            elif text_loc == "left":
                text_y = y1 - text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.RIGHT
            else:
                text_y = y2 + text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.LEFT
            if side == "right":
                text_x = x1 + ext_line_offset + text_offset
                x_p1_start = x1 + ext_line_offset
                x_p1_end = text_x + ext_line_extension
                x_p2_start = x2 + ext_line_offset
                x_p2_end = (
                    x2 + ext_line_offset + text_offset + ext_line_extension
                )
            else:
                text_x = x1 - ext_line_offset - text_offset
                x_p1_start = x1 - ext_line_offset
                x_p1_end = (
                    x1 - ext_line_offset - text_offset - ext_line_extension
                )
                x_p2_start = x2 - ext_line_offset
                x_p2_end = (
                    x2 - ext_line_offset - text_offset - ext_line_extension
                )
            ext1_start = (x_p1_start, y1)
            ext1_end = (x_p1_end, y1)
            ext2_start = (x_p2_start, y2)
            ext2_end = (x_p2_end, y2)
            dim1 = (text_x, dim_y1)
            dim2 = (text_x, dim_y2)
            stub1_tail = (text_x, dim_y1 - stub_length)
            stub2_tip = (text_x, dim_y2 + stub_length)
        else:
            dim_angle = line_angle(p1, p2)
            if side == "up":
                normal_angle = dim_angle + pi / 2
            else:
                normal_angle = dim_angle - pi / 2
            along_x = cos(dim_angle)
            along_y = sin(dim_angle)
            dist_to_line = ext_line_offset + text_offset
            ext_end_length = dist_to_line + ext_line_extension
            ext1_start = line_by_point_angle_length(
                p1, normal_angle, ext_line_offset
            )[1]
            ext1_end = line_by_point_angle_length(
                p1, normal_angle, ext_end_length
            )[1]
            ext2_start = line_by_point_angle_length(
                p2, normal_angle, ext_line_offset
            )[1]
            ext2_end = line_by_point_angle_length(
                p2, normal_angle, ext_end_length
            )[1]
            dim1 = line_by_point_angle_length(p1, normal_angle, dist_to_line)[1]
            dim2 = line_by_point_angle_length(p2, normal_angle, dist_to_line)[1]
            dim1_x, dim1_y = dim1[:2]
            dim2_x, dim2_y = dim2[:2]
            if text_loc == "middle":
                text_x = (dim1_x + dim2_x) / 2
                text_y = (dim1_y + dim2_y) / 2
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.CENTER
            elif text_loc == "left":
                text_x = dim1_x - along_x * text_horiz_offset
                text_y = dim1_y - along_y * text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.RIGHT
            else:
                text_x = dim2_x + along_x * text_horiz_offset
                text_y = dim2_y + along_y * text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.LEFT
            stub1_tail = (
                dim1_x - along_x * stub_length,
                dim1_y - along_y * stub_length,
            )
            stub2_tip = (
                dim2_x + along_x * stub_length,
                dim2_y + along_y * stub_length,
            )

        self.text_pos = (text_x, text_y)

        self.tag = Tag(
            text,
            pos=(text_x, text_y),
            fill=True,
            anchor=self.text_anchor,
            align=self.text_align,
            font_size=self.font_size,
        )
        self.ext1 = Line(ext1_start, ext1_end)
        self.ext2 = Line(ext2_start, ext2_end)
        self.append(self.ext1)
        self.append(self.ext2)
        if text_loc == "middle":
            self.dim_line = Arrow(dim1, dim2, head_pos=HeadPos.BOTH)
            self.append(self.dim_line)
        else:
            self.arrow1 = Arrow(stub1_tail, dim1, head_pos=HeadPos.END)
            self.arrow2 = Arrow(dim2, stub2_tip, head_pos=HeadPos.START)
            self.midline = Line(dim1, dim2)
            self.append(self.arrow1)
            self.append(self.arrow2)
            if self.show_midline:
                self.append(self.midline)

    def __repr__(self) -> str:
        """Return a Dimension string from this dimension's elements.

        Examples:
            >>> import simetri.graphics as sg
            >>> dim = sg.Dimension((0, 0), (40, 0), "up", 8)
            >>> repr(dim).startswith("Dimension(")
            True
            >>> str(dim).startswith("Group")
            True
        """
        if self.elements is None or len(self.elements) == 0:
            return "Dimension()"
        if len(self.elements) in [1, 2]:
            return f"Dimension({self.elements})"
        return f"Dimension({self.elements[0]}...{self.elements[-1]})"


_ALIGNED_SIDES = {
    "lower_left": (-1.0, -1.0),
    "lower_right": (1.0, -1.0),
    "upper_left": (-1.0, 1.0),
    "upper_right": (1.0, 1.0),
}


def _aligned_side_normal(angle: float, side: str) -> tuple[float, float]:
    """Return the unit normal of ``angle`` that faces ``side``."""
    if side not in _ALIGNED_SIDES:
        raise ValueError(
            "AlignedDimension side must be upper_left, upper_right, "
            "lower_left, or lower_right."
        )
    desired_x, desired_y = _ALIGNED_SIDES[side]
    left_x = -sin(angle)
    left_y = cos(angle)
    left_dot = left_x * desired_x + left_y * desired_y
    right_dot = -left_dot
    if left_dot >= right_dot:
        return (left_x, left_y)
    return (-left_x, -left_y)


class AlignedDimension(Group):
    """A dimension whose line, arrows, and extension lines share one angle.

    ``angle`` is the direction of the dimension line, in radians. ``None``
    uses the direction from ``p1`` to ``p2``. Extension lines are
    perpendicular to that direction. ``side`` picks which perpendicular
    faces that quadrant: ``upper_left``, ``upper_right``, ``lower_left``,
    or ``lower_right``. The label is the length of the dimension line.

    Args:
        p1 (PointType): First feature point.
        p2 (PointType): Second feature point.
        side (str): ``upper_left``, ``upper_right``, ``lower_left``, or
            ``lower_right``.
        text_offset (float, optional): Distance from the gap to the
            dimension line. ``None`` uses ``runtime_defaults["text_offset"]``.
        text (str, optional): Label text. ``None`` uses the dimension-line
            length, rounded to ``runtime_defaults["n_dim_digits"]``.
            Defaults to None.
        ext_line_extension (float, optional): How far each extension
            continues past the dimension line. ``None`` uses
            ``runtime_defaults["overshoot"]``.
        ext_line_offset (float, optional): Gap from the feature to the
            start of the extension. ``None`` uses ``runtime_defaults["gap"]``.
        text_horiz_offset (float, optional): Offset of the label along
            the dimension line when ``text_loc`` is ``"left"`` or
            ``"right"``. ``None`` uses ``runtime_defaults["ext_length2"]``.
        text_loc (str, optional): ``"middle"``, ``"left"``, or
            ``"right"``. Defaults to ``"middle"``.
        stub_length (float, optional): Outward shaft length when the
            label is not in the middle. ``None`` uses
            ``runtime_defaults["stub_length"]``.
        angle (float, optional): Dimension-line direction in radians.
            ``None`` uses the direction from ``p1`` to ``p2``.
        show_midline (bool, optional): Draw the line between the side
            arrows. Defaults to True.
        **kwargs: Additional keyword arguments for dimension styling.

    Examples:
        >>> import simetri.graphics as sg
        >>> dim = sg.AlignedDimension((0, 0), (40, 0), "upper_right", 8)
        >>> dim.angle
        0.0
        >>> dim.text
        '40.0'
        >>> dim.text_pos
        (20.0, 13.0)
        >>> dim.subtype == sg.Types.ALIGNED_DIMENSION
        True
        >>> low = sg.AlignedDimension((0, 0), (40, 0), "lower_left", 8)
        >>> low.text_pos
        (20.0, -13.0)
        >>> tilted = sg.AlignedDimension(
        ...     (0, 0), (40, 30), "upper_left", 8, angle=0.0
        ... )
        >>> tilted.angle
        0.0
        >>> round(tilted.dim_line.p1[1], 6) == round(tilted.dim_line.p2[1], 6)
        True
        >>> sg.AlignedDimension((0, 0), (0, 0), "upper_right", 8)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: AlignedDimension points must be distinct.
        >>> sg.AlignedDimension((0, 0), (0, 40), "upper_right", 8, angle=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: AlignedDimension length is zero at this angle.
        >>> sg.AlignedDimension((0, 0), (40, 0), "above", 8)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: AlignedDimension side must be upper_left, upper_right, lower_left, or lower_right.
        >>> plain = sg.AlignedDimension((0, 0), (40, 0), "upper_right")
        >>> plain.text_offset
        5
        >>> plain.stub_length
        15
        >>> plain.text_pos
        (20.0, 10.0)
    """

    def __init__(
        self,
        p1: PointType,
        p2: PointType,
        side: str,
        text_offset: float | None = None,
        text: str | None = None,
        ext_line_extension: float | None = None,
        ext_line_offset: float | None = None,
        text_horiz_offset: float | None = None,
        text_loc: str = "middle",
        stub_length: float | None = None,
        angle: float | None = None,
        show_midline: bool = True,
        **kwargs: object,
    ) -> None:
        """Create an aligned dimension with extension lines and arrows.

        See the class docstring for argument details.
        """
        (
            text_offset,
            ext_line_extension,
            ext_line_offset,
            text_horiz_offset,
            stub_length,
            font_size,
        ) = get_defaults(
            [
                "text_offset",
                "overshoot",
                "gap",
                "ext_length2",
                "stub_length",
                "font_size",
            ],
            [
                text_offset,
                ext_line_extension,
                ext_line_offset,
                text_horiz_offset,
                stub_length,
                None,
            ],
        )
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        abs_tol = runtime_defaults["abs_tol"]
        if abs(x1 - x2) < abs_tol and abs(y1 - y2) < abs_tol:
            raise ValueError("AlignedDimension points must be distinct.")
        if angle is None:
            angle = line_angle(p1, p2)
        normal_x, normal_y = _aligned_side_normal(angle, side)
        height1 = x1 * normal_x + y1 * normal_y
        height2 = x2 * normal_x + y2 * normal_y
        dist_to_line = ext_line_offset + text_offset
        if height1 >= height2:
            line_height = height1 + dist_to_line
        else:
            line_height = height2 + dist_to_line
        offset1 = line_height - height1
        offset2 = line_height - height2
        dim1 = (x1 + normal_x * offset1, y1 + normal_y * offset1)
        dim2 = (x2 + normal_x * offset2, y2 + normal_y * offset2)
        span_x = dim2[0] - dim1[0]
        span_y = dim2[1] - dim1[1]
        span = hypot(span_x, span_y)
        if span < abs_tol:
            raise ValueError("AlignedDimension length is zero at this angle.")
        if text is None:
            text = _format_dim_value(distance(dim1, dim2))
        along_x = span_x / span
        along_y = span_y / span
        dim1_x, dim1_y = dim1
        dim2_x, dim2_y = dim2
        if text_loc == "middle":
            text_x = (dim1_x + dim2_x) / 2
            text_y = (dim1_y + dim2_y) / 2
            text_anchor = Anchor.CENTER
            text_align = Align.CENTER
        elif text_loc == "left":
            text_x = dim1_x - along_x * text_horiz_offset
            text_y = dim1_y - along_y * text_horiz_offset
            text_anchor = Anchor.CENTER
            text_align = Align.RIGHT
        else:
            text_x = dim2_x + along_x * text_horiz_offset
            text_y = dim2_y + along_y * text_horiz_offset
            text_anchor = Anchor.CENTER
            text_align = Align.LEFT

        self.p1 = p1
        self.p2 = p2
        self.side = side
        self.angle = angle
        self.text_offset = text_offset
        self.text = text
        self.ext_line_extension = ext_line_extension
        self.ext_line_offset = ext_line_offset
        self.text_horiz_offset = text_horiz_offset
        self.text_loc = text_loc
        self.stub_length = stub_length
        self.show_midline = show_midline
        self.font_size = font_size
        self.kwargs = kwargs
        self.text_pos = (text_x, text_y)
        self.text_anchor = text_anchor
        self.text_align = text_align
        self.ext3 = None
        self.arrow1 = None
        self.arrow2 = None
        self.dim_line = None
        self.midline = None

        super().__init__(subtype=Types.ALIGNED_DIMENSION, **kwargs)

        self.tag = Tag(
            text,
            pos=(text_x, text_y),
            fill=True,
            anchor=self.text_anchor,
            align=self.text_align,
            font_size=self.font_size,
        )
        self.ext1 = Line(
            (x1 + normal_x * ext_line_offset, y1 + normal_y * ext_line_offset),
            (
                dim1_x + normal_x * ext_line_extension,
                dim1_y + normal_y * ext_line_extension,
            ),
        )
        self.ext2 = Line(
            (x2 + normal_x * ext_line_offset, y2 + normal_y * ext_line_offset),
            (
                dim2_x + normal_x * ext_line_extension,
                dim2_y + normal_y * ext_line_extension,
            ),
        )
        self.append(self.ext1)
        self.append(self.ext2)
        if text_loc == "middle":
            self.dim_line = Arrow(dim1, dim2, head_pos=HeadPos.BOTH)
            self.append(self.dim_line)
        else:
            self.arrow1 = Arrow(
                (
                    dim1_x - along_x * stub_length,
                    dim1_y - along_y * stub_length,
                ),
                dim1,
                head_pos=HeadPos.END,
            )
            self.arrow2 = Arrow(
                dim2,
                (
                    dim2_x + along_x * stub_length,
                    dim2_y + along_y * stub_length,
                ),
                head_pos=HeadPos.START,
            )
            self.midline = Line(dim1, dim2)
            self.append(self.arrow1)
            self.append(self.arrow2)
            if self.show_midline:
                self.append(self.midline)

    def __repr__(self) -> str:
        """Return an AlignedDimension string from this dimension's elements.

        Examples:
            >>> import simetri.graphics as sg
            >>> dim = sg.AlignedDimension((0, 0), (40, 0), "upper_right", 8)
            >>> repr(dim).startswith("AlignedDimension(")
            True
        """
        if self.elements is None or len(self.elements) == 0:
            return "AlignedDimension()"
        if len(self.elements) in [1, 2]:
            return f"AlignedDimension({self.elements})"
        return f"AlignedDimension({self.elements[0]}...{self.elements[-1]})"
