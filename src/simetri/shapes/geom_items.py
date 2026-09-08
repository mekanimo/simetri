"""Factory classes and helpers for common geometric shapes.

Includes ``Line``, ``Rectangle``, ``Circle``, ``Segment``, and helpers such
as ``square``, ``circle_points``, and ``reg_poly_shape``.

Examples:
    >>> import simetri.graphics as sg
    >>> c = sg.Circle(radius=25, center=(0, 0))
    >>> c.radius
    25.0
    >>> sq = sg.square(center=(0, 0), size=40)
    >>> sq.closed
    True
"""

from collections.abc import Callable, Sequence
from math import cos, gcd, pi, sin

import numpy as np

from simetri.coloring import colors

from ..geom.homogenize import homogenize
from ..geom.nonlinear.ellipse import ellipse_points
from ..geom.points.point_utils import distance
from ..geom.geometry import (
    side_len_to_radius,
)
from ..geom.geom_utils import (
    close_points_square,
    midpoint,
    reg_poly_points as regular_polygon_points,
)
from ..geom.polygons.polygon import offset_polygon_points
from ..geom.segments.line_utils import angle_between_lines3, fillet_corners
from ..geom.vectors import v_diff, v_scale, v_sum
from ..base.all_enums import Extent, Types
from ..group.batch import Group
from ..geom.bbox import BoundingBox
from ..base.common import PointType, axis_x, get_defaults
from .shape import Shape
from ..config.settings import defaults
from ..geom.affine import rotation_matrix

Color = colors.Color


def offset_box(
    corners: Sequence[PointType],
    offsets: Sequence[float] | None = None,
    offset: float | None = None,
) -> Shape:
    """Return a rectangle ``Shape`` from four corners and edge offsets.

    Positive offsets expand the box; negative offsets deflate it.

    Args:
        corners: Four corner points defining the original box.
        offsets: Per-edge offsets ``[left, bottom, right, top]``.
        offset: Single offset applied to all four edges (alternative to ``offsets``).

    Returns:
        A closed rectangle ``Shape``.

    Raises:
        ValueError: If neither ``offsets`` nor ``offset`` is given, if ``offsets``
            has length other than four, or if ``corners`` does not have four points.

    Examples:
        >>> import simetri.graphics as sg
        >>> box = sg.offset_box([(0, 0), (10, 0), (10, 5), (0, 5)], offset=1)
        >>> box.closed
        True
        >>> [round(coord, 6) for coord in box.vertices[0][:2]]
        [-1.0, 6.0]
    """

    # Handle the single offset case
    if offset is not None:
        offsets = [offset, offset, offset, offset]

    if offsets is None:
        raise ValueError("Either offsets or offset must be provided")

    if len(offsets) != 4:
        raise ValueError(
            "offsets must have 4 values: [left, bottom, right, top]"
        )

    left, bottom, right, top = offsets

    if len(corners) != 4:
        raise ValueError("corners must contain four points")

    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    # Apply offsets (positive = expansion, negative = deflation)
    new_x_min = x_min - left
    new_y_min = y_min - bottom
    new_x_max = x_max + right
    new_y_max = y_max + top

    new_corners = [
        (new_x_min, new_y_max),
        (new_x_min, new_y_min),
        (new_x_max, new_y_min),
        (new_x_max, new_y_max),
    ]

    return Shape(new_corners, closed=True)


def square(
    center: PointType = (0, 0), size: float = 100, angle: float = 0, **kwargs
) -> Shape:
    """Return a closed square Shape.

    Args:
        center: Center of the square. Defaults to ``(0, 0)``.
        size: Side length (width and height). Defaults to 100.
        angle: Rotation angle in radians. Defaults to 0.
        **kwargs: Passed to ``Shape``.

    Returns:
        Shape: Closed square.

    Examples:
        >>> import simetri.graphics as sg
        >>> sq = sg.square(size=50)
        >>> len(sq.vertices)
        4
        >>> [round(coord, 6) for coord in sq.vertices[0][:2]]
        [-25.0, -25.0]
    """
    points = rectangle_points(center, size, size, angle)

    return Shape(points, closed=True, **kwargs)


class Line(Shape):
    """A line defined by two points.

    Rendering depends on ``extent``:

    - ``Extent.INFINITE``: drawn through both directions to canvas limits.
    - ``Extent.RAY``: drawn from ``start`` toward ``end`` to the canvas limit.
    - ``Extent.SEGMENT``: drawn as a finite segment from ``start`` to ``end``.

    General form ``Ax + By + C = 0``: use ``A``, ``B``, ``C``, or ``ABC``.
    Slope-intercept form ``y = mx + b``: use ``slope``, ``intercept``, or ``m_b``.
    Parametric evaluation: ``parametric_function`` or ``t(t)`` returns
    ``start + t * (end - start)``.

    Examples:
        >>> import simetri.graphics as sg
        >>> line = sg.Line((0, 0), (10, 0))
        >>> line.extent.name
        'SEGMENT'
    """

    def __init__(
        self,
        start: PointType,
        end: PointType,
        extent: Extent = Extent.SEGMENT,
        draw_type: Extent | None = None,
        **kwargs,
    ) -> None:
        """Initialize a Line.

        Args:
            start: Start point.
            end: End point.
            extent: Rendering mode. Defaults to ``Extent.SEGMENT``.
            draw_type: Alias for ``extent``. Used when not ``None``.
            **kwargs: Additional shape keyword arguments.

        Raises:
            ValueError: If start and end points are the same.

        Examples:
            >>> import simetri.graphics as sg
            >>> line = sg.Line((0, 0), (1, 0), draw_type=sg.Extent.RAY)
            >>> line.extent.name
            'RAY'
        """
        dist_tol2 = defaults["dist_tol"] ** 2
        if close_points_square(start, end, dist2=dist_tol2):
            raise ValueError("Line: start and end points are the same!")

        if draw_type is not None:
            extent = draw_type

        super().__init__([start, end], closed=False, **kwargs)
        self.subtype = Types.LINE
        self.extent = extent

    def __setattr__(self, name: str, value: object) -> None:
        if name == "draw_type":
            name = "extent"
        super().__setattr__(name, value)

    @property
    def draw_type(self) -> Extent:
        """Backward-compatible alias for extent."""
        return self.extent

    @draw_type.setter
    def draw_type(self, value: Extent) -> None:
        """Set the draw extent (alias for ``extent``)."""
        self.extent = value

    @property
    def start(self) -> PointType:
        """Return the start point of the line."""
        return self.vertices[0]

    @start.setter
    def start(self, point: PointType) -> None:
        """Set the start point of the line."""
        self[0] = point[:2]

    @property
    def end(self) -> PointType:
        """Return the end point of the line."""
        return self.vertices[1]

    @end.setter
    def end(self, point: PointType) -> None:
        """Set the end point of the line."""
        self[1] = point[:2]

    @property
    def length(self) -> float:
        """Return the length between start and end points."""
        return distance(self.start, self.end)

    @property
    def A(self) -> float:
        """Return A coefficient of Ax + By + C = 0."""
        _, y1 = self.start[:2]
        _, y2 = self.end[:2]
        return y1 - y2

    @property
    def B(self) -> float:
        """Return B coefficient of Ax + By + C = 0."""
        x1, _ = self.start[:2]
        x2, _ = self.end[:2]
        return x2 - x1

    @property
    def C(self) -> float:
        """Return C coefficient of Ax + By + C = 0."""
        x1, y1 = self.start[:2]
        x2, y2 = self.end[:2]
        return x1 * y2 - x2 * y1

    @property
    def ABC(self) -> tuple[float, float, float]:
        """Return line coefficients (A, B, C) in general form."""
        return (self.A, self.B, self.C)

    @property
    def slope(self) -> float:
        """Return slope m for y = mx + b.

        Raises:
            ValueError: If the line is vertical.
        """
        x1, y1 = self.start[:2]
        x2, y2 = self.end[:2]
        dx = x2 - x1
        if abs(dx) <= defaults["dist_tol"]:
            raise ValueError("Line is vertical; slope is undefined.")
        return (y2 - y1) / dx

    @property
    def intercept(self) -> float:
        """Return y-intercept b for y = mx + b.

        Raises:
            ValueError: If the line is vertical.
        """
        m = self.slope
        x1, y1 = self.start[:2]
        return y1 - (m * x1)

    @property
    def m_b(self) -> tuple[float, float]:
        """Return (slope, intercept) for y = mx + b."""
        return (self.slope, self.intercept)

    @property
    def parametric_function(self) -> Callable[[float], PointType]:
        """Return a callable ``f(t)`` that gives points on the line.

        For ``Extent.SEGMENT`` lines, meaningful values are typically ``0 <= t <= 1``.

        Returns:
            Callable mapping parameter ``t`` to a point on the line.
        """
        return lambda t: self.t(t)

    # def copy(self, **kwargs):
    #     """Return a copy of the line."""
    #     line = Line(self.start, self.end, extent=self.extent)
    #     for attrib in custom_attributes(self):
    #        if attrib in ("vertices", "draw_type", "extent"):
    #             continue
    #         setattr(line, attrib, getattr(self, attrib))
    #     for attrib in shape_style_map:
    #         value = getattr(self, attrib, defaults[attrib])
    #         if value is not None:
    #             setattr(line, attrib, value)

    #     for k, v in kwargs.items():
    #         setattr(line, k, v)

    #     return line

    def t(self, t: float) -> PointType:
        """Return point at parameter ``t`` using ``start + t * (end - start)``.

        Args:
            t: Parametric coordinate along the line.

        Returns:
            PointType: The point at parameter ``t``.

        Examples:
            >>> import simetri.graphics as sg
            >>> line = sg.Line((0, 0), (10, 0))
            >>> [round(coord, 6) for coord in line.t(0.5)[:2]]
            [5.0, 0.0]
        """
        direction = v_diff(self.end, self.start)

        return v_sum(self.start, v_scale(direction, t))


class Rectangle(Shape):
    """Axis-aligned rectangle defined by center, width, and height.

    Examples:
        >>> import simetri.graphics as sg
        >>> r = sg.Rectangle((0, 0), 40, 20)
        >>> r.subtype.name
        'RECTANGLE'
        >>> r.width
        40.0
    """

    def __init__(
        self, center: PointType, width: float, height: float, **kwargs
    ) -> None:
        """Initialize a Rectangle.

        Args:
            center: Center point of the rectangle.
            width: Width of the rectangle.
            height: Height of the rectangle.
            **kwargs: Additional shape keyword arguments.
        """
        x, y = center[:2]
        half_width = width / 2
        half_height = height / 2
        vertices = [
            (x - half_width, y - half_height),
            (x + half_width, y - half_height),
            (x + half_width, y + half_height),
            (x - half_width, y + half_height),
        ]
        super().__init__(vertices, closed=True, **kwargs)
        self.subtype = Types.RECTANGLE

    def __setattr__(self, name: str, value: object) -> None:
        """Set an attribute of the rectangle.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        if name == "center":
            self._set_center(value)
        elif name == "width":
            self._set_width(value)
        elif name == "height":
            self._set_height(value)
        else:
            super().__setattr__(name, value)

    # def scale(
    #     self,
    #     scale_x: float,
    #     scale_y: float | None = None,
    #     about: PointType = (0, 0),
    #     reps: int = 0,
    # ):
    #     """Scale the rectangle by scale_x and scale_y.
    #     Rectangles cannot be scaled non-uniformly.
    #     scale_x changes the width and scale_y changes the height.

    #     Args:
    #         scale_x (float): The scale factor for the width.
    #         scale_y (float, optional): The scale factor for the height. Defaults to None.
    #         about (PointType, optional): The point to scale about. Defaults to (0, 0).
    #         reps (int, optional): The number of repetitions. Defaults to 0.

    #     Returns:
    #         Rectangle: The scaled rectangle.
    #     """
    #     if scale_y is None:
    #         scale_y = scale_x
    #     center = self.midpoint
    #     _, rotation, _ = decompose_transformations(self.xform_matrix)
    #     rm = rotation_matrix(-rotation, center)
    #     sm = scale_in_place_matrix(scale_x, scale_y, about)
    #     inv_rm = rotation_matrix(rotation, center)
    #     transform = rm @ sm @ inv_rm

    #     return self._update(transform, reps=reps)

    @property
    def width(self) -> float:
        """Return the width of the rectangle.

        Returns:
            float: The width of the rectangle.
        """
        return distance(self.vertices[0], self.vertices[1])

    def _set_width(self, new_width: float) -> None:
        """Set the width of the rectangle.

        Args:
            new_width (float): The new width of the rectangle.
        """
        scale_x = new_width / self.width
        self.scale(scale_x, 1, about=self.center, reps=0)

    @property
    def height(self) -> float:
        """Return the height of the rectangle.

        Returns:
            float: The height of the rectangle.
        """
        return distance(self.vertices[1], self.vertices[2])

    def _set_height(self, new_height: float) -> None:
        """Set the height of the rectangle.

        Args:
            new_height (float): The new height of the rectangle.
        """
        scale_y = new_height / self.height
        self.scale(1, scale_y, about=self.center, reps=0)

    @property
    def center(self) -> PointType:
        """Return the center of the rectangle.

        Returns:
            PointType: The center of the rectangle.
        """
        return midpoint(self.vertices[0], self.vertices[2])

    def _set_center(self, new_center: PointType) -> None:
        """Set the center of the rectangle.

        Args:
            new_center (PointType): The new center of the rectangle.
        """
        center = self.center
        x_diff = new_center[0] - center[0]
        y_diff = new_center[1] - center[1]
        for i in range(4):
            x, y = self.vertices[i][:2]
            self[i] = (x + x_diff, y + y_diff)


class Rectangle2(Rectangle):
    """A rectangle defined by two opposite corners.

    Examples:
        >>> import simetri.graphics as sg
        >>> rect = sg.Rectangle2((0, 0), (10, 4))
        >>> rect.width
        10.0
        >>> rect.height
        4.0
    """

    def __init__(
        self, corner1: PointType, corner2: PointType, **kwargs
    ) -> None:
        """Initialize a Rectangle2 object.

        Args:
            corner1 (PointType): The first corner of the rectangle.
            corner2 (PointType): The second corner of the rectangle.
            **kwargs: Additional keyword arguments.
        """
        x1, y1 = corner1[:2]
        x2, y2 = corner2[:2]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        width = x_max - x_min
        height = y_max - y_min
        super().__init__(center, width, height, **kwargs)


class Circle(Shape):
    """Circle defined by center and radius.

    Stored as a one-point shape at the center; ``radius`` drives drawing and
    the bounding box.

    Attributes:
        center: Center point (alias for the single vertex).
        radius: Circle radius.
        subtype: Always ``Types.CIRCLE``.

    Examples:
        >>> import simetri.graphics as sg
        >>> c = sg.Circle(radius=10, center=(5, 5))
        >>> c.center
        (5.0, 5.0)
    """

    def __init__(
        self,
        radius: float | None = None,
        center: PointType = (0, 0),
        xform_matrix: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """Initialize a Circle.

        Args:
            radius: Circle radius. Defaults to ``defaults["circle_radius"]``.
            center: Center point. Defaults to ``(0, 0)``.
            xform_matrix: Optional initial transform.
            **kwargs: Additional shape keyword arguments.
        """
        if radius is None:
            radius = defaults["circle_radius"]

        x, y = center[:2]
        points = [[x, y]]
        super().__init__(points, xform_matrix=xform_matrix, **kwargs)
        self.subtype = Types.CIRCLE
        self._radius = radius

    def __setattr__(self, name: str, value: object) -> None:
        """Set an attribute of the circle.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        if name == "center":
            self[0] = value[:2]
        elif name == "radius":
            ratio = value / self.radius
            self.scale(ratio, about=self.center, reps=0)
        else:
            super().__setattr__(name, value)

    @property
    def b_box(self) -> BoundingBox:
        """Return the bounding box of the shape.

        Returns:
            BoundingBox: The bounding box of the shape.
        """
        x, y = self.center[:2]
        x1, y1 = x - self.radius, y - self.radius
        x2, y2 = x + self.radius, y + self.radius
        self._b_box = BoundingBox((x1, y1), (x2, y2))

        return self._b_box

    @property
    def closed(self) -> bool:
        """Return True. Circles are closed.

        Returns:
            bool: True
        """
        return True

    @closed.setter
    def closed(self, value: bool) -> None:
        """No-op setter; circles are always closed."""
        pass

    @property
    def center(self) -> PointType:
        """Return the center of the circle.

        Returns:
            PointType: The center of the circle.
        """
        return self.vertices[0]

    @center.setter
    def center(self, value: PointType) -> None:
        """Set the center of the circle.

        Args:
            value (PointType): The new center of the circle.
        """
        self[0] = value[:2]

    @property
    def radius(self) -> float:
        """Return the radius of the circle.

        Returns:
            float: The radius of the circle.
        """
        scale_x = np.linalg.norm(
            self.xform_matrix[0, :2]
        )  # only x scale is used
        return self._radius * scale_x

    def __eq__(self, other: object) -> bool:
        """Check if the circle is equal to another circle.

        Args:
            other (Circle): The other circle to compare to.

        Returns:
            bool: True if the circles are equal, False otherwise.
        """
        if not isinstance(other, Circle):
            return False
        return self.id == other.id

    # def copy(self, **kwargs):
    #     """Return a copy of the circle.

    #     Returns:
    #         Circle: A copy of the circle.
    #     """

    #     center = self.center
    #     radius = self.radius
    #     circle = Circle(center=center, radius=radius)

    #     custom_attribs = custom_attributes(self)
    #     custom_attribs.remove("center")
    #     custom_attribs.remove("_radius")
    #     custom_attribs.remove("radius")
    #     for attrib in custom_attribs:
    #         setattr(circle, attrib, getattr(self, attrib))
    #     for attrib in shape_style_map:
    #         value = getattr(self, attrib, defaults[attrib])
    #         if value is not None:
    #             setattr(circle, attrib, value)

    #     for k, v in kwargs.items():
    #         setattr(circle, k, v)

    #     return circle


class Segment(Shape):
    """Finite line segment between two points.

    Prefer ``Line`` with ``extent=Extent.SEGMENT`` for new code.

    Examples:
        >>> import simetri.graphics as sg
        >>> seg = sg.Segment((0, 0), (10, 0))
        >>> seg.subtype.name
        'SEGMENT'
    """

    def __init__(self, start: PointType, end: PointType, **kwargs) -> None:
        """Initialize a Segment.

        Args:
            start: Start point.
            end: End point.
            **kwargs: Additional shape keyword arguments.

        Raises:
            ValueError: If the start and end points are the same.
        """
        dist_tol2 = defaults["dist_tol"] ** 2
        if close_points_square(start, end, dist2=dist_tol2):
            raise ValueError("Segment: start and end points are the same!")
        points = [start, end]
        super().__init__(points, **kwargs)
        self.subtype = Types.SEGMENT

    @property
    def start(self) -> PointType:
        """Return the start point of the segment.

        Returns:
            PointType: The start point of the segment.
        """
        return self.vertices[0]

    @property
    def end(self) -> PointType:
        """Return the end point of the segment.

        Returns:
            PointType: The end point of the segment.
        """
        return self.vertices[1]

    @property
    def length(self) -> float:
        """Return the length of the segment.

        Returns:
            float: The length of the segment.
        """
        return distance(self.start, self.end)

    # def copy(self, **kwargs) -> Shape:
    #     """Return a copy of the segment.

    #     Returns:
    #         Shape: A copy of the segment.
    #     """
    #     segment = Segment(self.start, self.end)
    #     custom_attribs = custom_attributes(self)
    #     for attrib in custom_attribs:
    #         if attrib.startswith("_"):
    #             continue
    #         if hasattr(self, attrib):
    #             setattr(segment, attrib, getattr(self, attrib))
    #     for attrib in shape_style_map:
    #         value = getattr(self, attrib, defaults[attrib])
    #         if value is not None:
    #             setattr(segment, attrib, value)

    #     for k, v in kwargs.items():
    #         setattr(segment, k, v)

    #     return segment

    def __str__(self) -> str:
        """Return a string representation of the segment.

        Returns:
            str: The string representation of the segment.
        """
        return f"Segment({self.start}, {self.end})"

    def __repr__(self) -> str:
        """Return a string representation of the segment.

        Returns:
            str: The string representation of the segment.
        """
        return f"Segment({self.start}, {self.end})"

    def __eq__(self, other: object) -> bool:
        """Check if the segment is equal to another segment.

        Args:
            other (Segment): The other segment to compare to.

        Returns:
            bool: True if the segments are equal, False otherwise.
        """
        return (
            other.type == Types.SEGMENT
            and self.start == other.start
            and self.end == other.end
        )


def circle_points(
    center: PointType, radius: float, n: int = 30
) -> list[PointType]:
    """Return a list of points that form a circle with the given parameters.

    Args:
        center (PointType): The center point of the circle.
        radius (float): The radius of the circle.
        n (int, optional): The number of points in the circle. Defaults to 30.

    Returns:
        list[PointType]: A list of points that form a circle.

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.circle_points((0, 0), 1, n=4)
        >>> len(pts)
        4
        >>> [round(coord, 6) for coord in pts[0][:2]]
        [1.0, 0.0]
    """
    return arc_points(center, radius, 0, 2 * pi, n=n)


def arc_points(
    center: PointType,
    radius: float,
    start_angle: float,
    end_angle: float,
    clockwise: bool = False,
    n: int = 20,
) -> list[PointType]:
    """Return a list of points that form a circular arc with the given parameters.

    Args:
        center (PointType): The center point of the arc.
        radius (float): The radius of the arc.
        start_angle (float): The starting angle of the arc.
        end_angle (float): The ending angle of the arc.
        clockwise (bool, optional): Whether the arc is drawn clockwise. Defaults to False.
        n (int, optional): The number of points in the arc. Defaults to 20.

    Returns:
        list[PointType]: A list of points that form a circular arc.

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.arc_points((0, 0), 1, 0, sg.pi / 2, clockwise=True, n=2)
        >>> [round(coord, 6) for coord in pts[0][:2]]
        [0.0, 1.0]
    """
    x, y = center[:2]
    points = []
    if clockwise:
        start_angle, end_angle = end_angle, start_angle
    step = (end_angle - start_angle) / n
    for i in range(n):
        angle = start_angle + step * i
        points.append([x + radius * cos(angle), y + radius * sin(angle)])
    return points


def hex_points(side_length: float) -> list[PointType]:
    """Return a list of points that define a hexagon with a given side length.

    Args:
        side_length (float): The length of each side of the hexagon.

    Returns:
        list[PointType]: A list of points that define the hexagon.

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.hex_points(1)
        >>> len(pts)
        6
        >>> [round(coord, 6) for coord in pts[0][:2]]
        [1.0, 0.0]
    """
    points = []
    for i in range(6):
        x = side_length * cos(i * 2 * pi / 6)
        y = side_length * sin(i * 2 * pi / 6)
        points.append((x, y))
    return points


def rectangle_points(
    pos: PointType = (0, 0),
    width: float = 100,
    height: float = 100,
    angle: float = 0,
) -> Sequence[PointType]:
    """Return four corner points of a rectangle.

    Args:
        pos: Center of the rectangle. Defaults to ``(0, 0)``.
        width: Rectangle width. Defaults to 100.
        height: Rectangle height. Defaults to 100.
        angle: Rotation about ``pos`` in radians. Defaults to 0.

    Returns:
        Sequence[PointType]: Corner points in order (not closed).

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.rectangle_points((0, 0), 10, 4)
        >>> len(pts)
        4
        >>> [round(coord, 6) for coord in pts[0][:2]]
        [-5.0, -2.0]
    """
    from ..geom.affine import rotate

    x, y = pos[:2]
    points = []
    points.append([x - width / 2, y - height / 2])
    points.append([x + width / 2, y - height / 2])
    points.append([x + width / 2, y + height / 2])
    points.append([x - width / 2, y + height / 2])
    if angle != 0:
        points = rotate(points, angle, (x, y))
    return points


def reg_poly_points_side_length(
    n: int, side_len: float, pos: PointType = (0, 0), angle: float = 0
) -> Sequence[PointType]:
    """Return vertices of a regular polygon with a given side length.

    Args:
        n: Number of sides.
        side_len: Length of each side.
        pos: Center of the polygon. Defaults to ``(0, 0)``.
        angle: Rotation angle in radians. Defaults to 0.

    Returns:
        Sequence[PointType]: Vertices of the polygon (not closed; first vertex
        is not repeated).

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.reg_poly_points_side_length(4, 2, angle=sg.pi / 2)
        >>> abs(pts[0][0]) < 1e-9
        True
    """
    rad = side_len_to_radius(n, side_len)
    sector = 2 * pi / n
    x, y = pos[:2]
    points = [
        [cos(sector * i) * rad + x, sin(sector * i) * rad + y] for i in range(n)
    ]

    if angle != 0:
        points = homogenize(points) @ rotation_matrix(angle)
        points = [(px, py) for (px, py, _) in points]

    return points


def reg_poly_points(
    n: int, r: float = 100, pos: PointType = (0, 0), angle: float = 0
) -> Sequence[PointType]:
    """Return vertices of a regular polygon with a given circumradius.

    Args:
        n: Number of sides.
        r: Circumradius. Defaults to 100.
        pos: Center of the polygon. Defaults to ``(0, 0)``.
        angle: Rotation angle in radians. Defaults to 0.

    Returns:
        Sequence[PointType]: Vertices of the polygon (closed; first vertex is
        repeated at the end).

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.reg_poly_points(4, 1)
        >>> len(pts)
        5
        >>> pts[0] == pts[-1]
        True
    """
    points = regular_polygon_points(pos, n, r)

    if angle != 0:
        points = homogenize(points) @ rotation_matrix(angle)
        points = [(x, y) for (x, y, _) in points]

    return points


def di_star(points: Sequence[PointType], n: int) -> Group:
    """Return a dihedral star with n petals.

    Args:
        points (Sequence[PointType]): List of [x, y] points.
        n (int): Number of petals.

    Returns:
        Group: A Group instance (dihedral star with n petals).

    Examples:
        >>> import simetri.graphics as sg
        >>> star = sg.di_star([(1, 0), (0.5, 0.2)], 2)
        >>> star.type.name
        'GROUP'
    """
    group = Group(Shape(points))
    return group.mirror(axis_x, reps=1).rotate(2 * pi / n, reps=n - 1)


def hex_grid_centers(
    x: float,
    y: float,
    side_length: float,
    n_rows: int,
    n_cols: int,
) -> list[PointType]:
    """Return a list of points that define the centers of hexagons in a grid.

    Args:
        x (float): The x-coordinate of the starting point.
        y (float): The y-coordinate of the starting point.
        side_length (float): The length of each side of the hexagons.
        n_rows (int): The number of rows in the grid.
        n_cols (int): The number of columns in the grid.

    Returns:
        list[PointType]: A list of points that define the centers of the hexagons.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.hex_grid_centers(0, 0, 1, 1, 1)
        [(0, 0)]
    """
    centers = []
    for row in range(n_rows):
        for col in range(n_cols):
            x_ = col * 3 * side_length + x
            y_ = row * 2 * side_length + y
            if col % 2:
                y_ += side_length
            centers.append((x_, y_))

    return centers


def rect_grid(
    x: float,
    y: float,
    cell_width: float,
    cell_height: float,
    n_rows: int,
    n_cols: int,
    pattern: Sequence[Sequence[bool]],
) -> Group:
    """Return a grid of rectangles with the given parameters.

    Args:
        x (float): The x-coordinate of the starting point.
        y (float): The y-coordinate of the starting point.
        cell_width (float): The width of each cell in the grid.
        cell_height (float): The height of each cell in the grid.
        n_rows (int): The number of rows in the grid.
        n_cols (int): The number of columns in the grid.
        pattern (list[list[bool]]): A pattern to fill the grid.

    Returns:
        Group: A Group object representing the grid.

    Examples:
        >>> import simetri.graphics as sg
        >>> grid = sg.rect_grid(0, 0, 10, 10, 1, 1, [[True]])
        >>> len(grid) > 1
        True
    """
    width = cell_width * n_cols
    height = cell_height * n_rows
    horiz_line = line_shape((x, y), (x + width, y))
    horiz_lines = Group(horiz_line)
    horiz_lines.translate(0, cell_height, reps=n_rows)
    vert_line = line_shape((x, y), (x, y + height))
    vert_lines = Group(vert_line)
    vert_lines.translate(cell_width, 0, reps=n_cols)
    grid = Group([horiz_lines, vert_lines])
    for row in range(n_rows):
        for col in range(n_cols):
            if pattern[row][col]:
                x_, y_ = (
                    col * cell_width + x,
                    (n_rows - row - 1) * cell_height + y,
                )
                points = [
                    (x_, y_),
                    (x_ + cell_width, y_),
                    (x_ + cell_width, y_ + cell_height),
                    (x_, y_ + cell_height),
                ]
                cell = Shape(points, closed=True, fill_color=colors.gray)
                grid.append(cell)
    return grid


def reg_star_polygon(n: int, step: int, rad: float, **kwargs) -> Shape | Group:
    """Return a regular star polygon.

    Args:
        n: Number of vertices on the generating regular ``n``-gon.
        step: Step size for connecting vertices (star polygon parameter).
        rad: Circumradius of the generating regular polygon.
        **kwargs: Additional keyword arguments passed to ``Shape``.

    Returns:
        ``Shape`` or ``Group``: A single star polygon, or a ``Group`` of rotated
        copies when ``gcd(n, step) > 1``.

    Examples:
        >>> import simetri.graphics as sg
        >>> star = sg.reg_star_polygon(5, 2, 10)
        >>> star.subtype.name
        'SHAPE'
    """
    angle = 2 * pi / n
    points = [(cos(angle * i) * rad, sin(angle * i) * rad) for i in range(n)]
    if n % step:
        indices = [i % n for i in list(range(0, (n + 1) * step, step))]
    else:
        indices = [
            i % n for i in list(range(0, ((n // step) + 1) * step, step))
        ]
    vertices = [points[ind] for ind in indices]
    reps = gcd(n, step) - 1
    shape = Shape(vertices, **kwargs)
    if reps > 1:
        res = Group(shape.rotate(angle, reps=reps))
    else:
        res = shape

    return res


def star_shape(
    points: Sequence[PointType], reps: int = 5, scale: float = 1
) -> Group:
    """Return a dihedral star from a list of points.

    Args:
        points (list[PointType]): The list of points that form the star.
        reps (int, optional): The number of repetitions. Defaults to 5.
        scale (float, optional): The scale factor. Defaults to 1.

    Returns:
        Group: A Group object representing the star.

    Examples:
        >>> import simetri.graphics as sg
        >>> star = sg.star_shape([(1, 0), (0.2, 0.2)], reps=2, scale=2)
        >>> star.type.name
        'GROUP'
    """
    shape = Shape(points, subtype=Types.STAR)
    group = Group(shape)
    group.mirror(axis_x, reps=1)
    group.rotate(2 * pi / (reps), reps=reps - 1)
    group.scale(scale)
    return group


def dot_shape(
    radius: float = 1,
    pos: PointType = (0, 0),
    fill_color: Color | None = None,
    line_color: Color | None = None,
    line_width: float | None = None,
) -> Shape:
    """Return a marker dot ``Shape`` (a single point with marker radius).

    Args:
        radius: Marker radius. Defaults to 1.
        pos: Position of the dot. Defaults to ``(0, 0)``.
        fill_color: Fill color. Defaults to package defaults.
        line_color: Stroke color. Defaults to package defaults.
        line_width: Stroke width. Defaults to package defaults.

    Returns:
        Shape: A point shape with ``marker`` set to ``radius``.

    Examples:
        >>> import simetri.graphics as sg
        >>> dot = sg.dot_shape(radius=3, pos=(1, 2))
        >>> dot.marker
        3
        >>> [round(coord, 6) for coord in dot.vertices[0][:2]]
        [1.0, 2.0]
    """
    fill_color, line_color, line_width = get_defaults(
        ["fill_color", "line_color", "line_width"],
        [fill_color, line_color, line_width],
    )
    x, y = pos[:2]
    dot_shape = Shape(
        [(x, y)],
        closed=True,
        fill_color=fill_color,
        line_color=line_color,
        line_width=line_width,
        subtype=Types.DOT,
    )
    dot_shape.marker = radius
    return dot_shape


def rect_shape(
    width: float,
    height: float,
    pos: PointType = (0, 0),
    fill_color: Color | None = colors.white,
    line_color: Color | None = defaults["line_color"],
    line_width: float | None = defaults["line_width"],
    fill: bool = True,
    marker: float | None = None,
    **kwargs,
) -> Shape:
    """Return a rectangle ``Shape`` from lower-left corner, width, and height.

    Args:
        width: Rectangle width.
        height: Rectangle height.
        pos: Lower-left corner. Defaults to ``(0, 0)``.
        fill_color: Interior color. Defaults to ``colors.white``.
        line_color: Stroke color. Defaults to ``defaults["line_color"]``.
        line_width: Stroke width. Defaults to ``defaults["line_width"]``.
        fill: Whether the rectangle is filled. Defaults to True.
        marker: Optional vertex marker. Defaults to None.
        **kwargs: Additional keyword arguments passed to ``Shape``.

    Returns:
        Shape: A closed axis-aligned rectangle.

    Examples:
        >>> import simetri.graphics as sg
        >>> rect = sg.rect_shape(10, 4, pos=(1, 2), marker=2)
        >>> rect.marker
        2
        >>> [round(coord, 6) for coord in rect.vertices[0][:2]]
        [1.0, 2.0]
    """
    x, y = pos[:2]
    fill_color, line_color, line_width = get_defaults(
        ["fill_color", "line_color", "line_width"],
        [fill_color, line_color, line_width],
    )
    rect = Shape(
        [(x, y), (x + width, y), (x + width, y + height), (x, y + height)],
        closed=True,
        fill_color=fill_color,
        line_color=line_color,
        fill=fill,
        line_width=line_width,
        subtype=Types.RECTANGLE,
        **kwargs,
    )
    if marker is not None:
        rect.marker = marker
    return rect


def arc_shape(
    x: float,
    y: float,
    radius: float,
    start_angle: float,
    end_angle: float,
    clockwise: bool = False,
    n: int = 20,
) -> Shape:
    """Return a Shape object with points that form a circular arc with the given parameters.

    Args:
        x (float): The x-coordinate of the center of the arc.
        y (float): The y-coordinate of the center of the arc.
        radius (float): The radius of the arc.
        start_angle (float): The starting angle of the arc.
        end_angle (float): The ending angle of the arc.
        clockwise (bool, optional): Whether the arc is drawn clockwise. Defaults to False.
        n (int, optional): The number of points to use for the arc. Defaults to 20.

    Returns:
        Shape: A Shape object with points that form a circular arc.

    Examples:
        >>> import simetri.graphics as sg
        >>> arc = sg.arc_shape(0, 0, 1, 0, sg.pi / 2, clockwise=True, n=2)
        >>> [round(coord, 6) for coord in arc.vertices[0][:2]]
        [0.0, 1.0]
    """
    points = arc_points(
        (x, y), radius, start_angle, end_angle, clockwise=clockwise, n=n
    )
    return Shape(points, closed=False, subtype=Types.ARC)


def circle_shape(
    radius: float,
    pos: PointType = (0, 0),
    n: int = 30,
    **kwargs,
) -> Shape:
    """Return a Shape object with points that form a circle with the given parameters.

    Args:
        radius (float): The radius of the circle.
        pos (PointType, optional): The position of the center of the circle. Defaults to (0, 0).
        n (int, optional): The number of points to use for the circle. Defaults to 30.

    Returns:
        Shape: A Shape object with points that form a circle.

    Examples:
        >>> import simetri.graphics as sg
        >>> circ = sg.circle_shape(2, pos=(3, 4), n=4, fill=False)
        >>> circ.fill
        False
        >>> len(circ.vertices)
        4
    """
    x, y = pos[:2]
    points = circle_points((x, y), radius, n=n)
    return Shape(points, closed=True, **kwargs)


def reg_poly_shape(
    n: int, r: float = 100, pos: PointType = (0, 0), angle: float = 0, **kwargs
) -> Shape:
    """Return an ``n``-sided regular polygon with circumradius ``r``.

    Args:
        n: Number of sides.
        r: Circumradius. Defaults to 100.
        pos: Center of the polygon. Defaults to ``(0, 0)``.
        angle: Rotation angle in radians. Defaults to 0.
        **kwargs: Additional keyword arguments passed to ``Shape``.

    Returns:
        Shape: A closed regular polygon.

    Examples:
        >>> import simetri.graphics as sg
        >>> poly = sg.reg_poly_shape(4, r=1, fill=False)
        >>> poly.fill
        False
        >>> poly.closed
        True
    """
    x, y = pos[:2]
    points = reg_poly_points(n=n, r=r, pos=(x, y), angle=angle)

    return Shape(points, closed=True, **kwargs)


def reg_poly_shape_side_length(
    n: int, side_len: float, pos: PointType = (0, 0), angle: float = 0, **kwargs
) -> Shape:
    """Return a regular polygon with a given side length.

    Args:
        n: Number of sides.
        side_len: Length of each side.
        pos: Center of the polygon. Defaults to ``(0, 0)``.
        angle: Rotation angle in radians. Defaults to 0.
        **kwargs: Additional keyword arguments passed to ``Shape``.

    Returns:
        Shape: A closed regular polygon.

    Examples:
        >>> import simetri.graphics as sg
        >>> poly = sg.reg_poly_shape_side_length(4, 2, angle=sg.pi / 2)
        >>> abs(poly.vertices[0][0]) < 1e-9
        True
    """

    x, y = pos[:2]
    points = reg_poly_points_side_length(
        n=n, side_len=side_len, pos=(x, y), angle=angle
    )

    return Shape(points, closed=True, **kwargs)


def ellipse_shape(
    width: float,
    height: float,
    angle: float = 0,
    pos: PointType = (0, 0),
    n_points: int | None = None,
    **kwargs,
) -> Shape:
    """Return an ellipse as a ``Shape``.

    Args:
        width: Major-axis diameter (full width).
        height: Minor-axis diameter (full height).
        angle: Rotation angle in radians. Defaults to 0.
        pos: Center of the ellipse. Defaults to ``(0, 0)``.
        n_points: Number of sample points. Defaults to ``defaults["n_ellipse_points"]``.
        **kwargs: Additional keyword arguments passed to ``Shape``.

    Returns:
        Shape: An ellipse approximated by a polyline.

    Examples:
        >>> import simetri.graphics as sg
        >>> ell = sg.ellipse_shape(10, 6, pos=(1, 2), n_points=8)
        >>> len(ell.vertices) != len(sg.ellipse_shape(10, 6, n_points=16).vertices)
        True
        >>> ell.subtype.name
        'ELLIPSE'
    """
    if n_points is None:
        n_points = defaults["n_ellipse_points"]

    points = ellipse_points(pos, width, height, angle, n_points=n_points)
    return Shape(points, subtype=Types.ELLIPSE, **kwargs)


def line_shape(
    p1: PointType,
    p2: PointType,
    line_width: float = 1,
    line_color: Color = colors.black,
    **kwargs,
) -> Line:
    """Return a ``Line`` between two points.

    Args:
        p1: Start point.
        p2: End point.
        line_width: Stroke width. Defaults to 1.
        line_color: Stroke color. Defaults to ``colors.black``.
        **kwargs: Additional keyword arguments passed to ``Line``.

    Returns:
        Line: A line segment between ``p1`` and ``p2``.

    Examples:
        >>> import simetri.graphics as sg
        >>> line = sg.line_shape((0, 0), (4, 0), line_width=2)
        >>> line.line_width
        2
        >>> line.extent.name
        'SEGMENT'
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    return Line(
        (x1, y1),
        (x2, y2),
        line_color=line_color,
        line_width=line_width,
        **kwargs,
    )


def offset_polygon_shape(
    polygon_shape: Shape,
    offset: float = 1,
    dist_tol: float = defaults["dist_tol"],
) -> Shape:
    """Return a polygon ``Shape`` with offset edges.

    Args:
        polygon_shape: Source polygon ``Shape``.
        offset: Offset distance (positive expands outward). Defaults to 1.
        dist_tol: Distance tolerance for offset construction. Defaults to
            ``defaults["dist_tol"]``.

    Returns:
        Shape: A new closed polygon with offset vertices.

    Examples:
        >>> import simetri.graphics as sg
        >>> src = sg.square(size=10)
        >>> out = sg.offset_polygon_shape(src, offset=0)
        >>> out is not src
        True
        >>> len(out.vertices)
        4
    """
    vertices = offset_polygon_points(polygon_shape.vertices, offset, dist_tol)

    return Shape(vertices)


def snap(
    free_shape: Shape,
    ref1: int | float,
    fixed_shape: Shape,
    ref2: int | float,
    angle: float = 0,
) -> Shape:
    """Snap ``free_shape`` to ``fixed_shape`` at the given references.

    References are vertex indices (``int``) or barycentric edge coordinates
    (``float``). For closed polygons, vertex walks are assumed counter-clockwise.

    When ``angle`` is zero, edges meeting at the snap points align edge-to-edge.
    Otherwise ``free_shape`` is rotated about the coincident point so the angle
    between the adjacent edges matches ``angle``.

    Integer ``ref``: snap at vertex ``k``.
    Float ``ref``: snap at a point on an edge; for example ``1.5`` is the
    midpoint of the edge from vertex 1 to vertex 2 (zero-based indexing).

    Args:
        free_shape (mutated): Shape moved and rotated into place.
        ref1: Reference on ``free_shape`` (vertex index or edge coordinate).
        fixed_shape: Shape that stays fixed.
        ref2: Reference on ``fixed_shape``.
        angle: Target angle in radians between adjacent edges at the snap point.
            Defaults to 0 (edge-to-edge alignment).

    Examples:
        >>> import simetri.graphics as sg
        >>> free = sg.square(center=(0, 0), size=2)
        >>> fixed = sg.square(center=(10, 0), size=2)
        >>> snapped = sg.snap(free, 1, fixed, 0)
        >>> snapped is free
        True
        >>> abs(snapped[1][0] - fixed[0][0]) < 1e-9
        True

    Returns:
        The transformed ``free_shape`` (same object, mutated in place).
    """

    def get_edge_indices(shape: Shape, ref: int | float) -> tuple[int, int]:
        """Get the edge indices for alignment.

        For a vertex index, returns (prev_vertex, vertex).
        For a barycentric coordinate, returns the two vertices of the edge.

        Args:
            shape: The shape
            ref: The reference (int or float)

        Returns:
            Tuple of (previous_index, current_index)

        Raises:
            ValueError: If ref is at a boundary of a non-closed shape
        """
        n_vertices = len(shape.vertices)
        is_closed = shape.closed

        if isinstance(ref, int):
            # For vertex: need vertices at ref-1, ref, ref+1
            if not is_closed:
                if ref == 0:
                    raise ValueError(
                        "Cannot snap at first vertex (index 0) of a non-closed shape"
                    )
                if ref >= n_vertices - 1:
                    raise ValueError(
                        f"Cannot snap at last vertex (index {ref}) of a non-closed shape"
                    )
            prev_idx = (ref - 1) % n_vertices if is_closed else ref - 1
            return (prev_idx, ref)
        elif isinstance(ref, float):
            # For edge point: need vertices at edge_index, edge_index+1, edge_index+2
            edge_index = int(ref)
            if not is_closed and edge_index >= n_vertices - 2:
                raise ValueError(
                    f"Cannot snap at edge {edge_index} of a non-closed shape with {n_vertices} vertices"
                )
            next_index = (
                (edge_index + 1) % n_vertices if is_closed else edge_index + 1
            )
            return (edge_index, next_index)

    free = free_shape
    fixed = fixed_shape
    # Get the reference points on both shapes
    ref1_point = free[ref1]
    ref2_point = fixed[ref2]

    # Move the free object to make ref1 and ref2 coincident
    dx = ref2_point[0] - ref1_point[0]
    dy = ref2_point[1] - ref1_point[1]
    free.translate(dx, dy)

    # Update ref1_point after translation
    ref1_point = ref2_point

    # Get edge information for angle calculation and rotation
    # Get edge indices for alignment
    _, free_curr_idx = get_edge_indices(free, ref1)
    fixed_prev_idx, _ = get_edge_indices(fixed, ref2)

    # Get the direction vectors for angle calculation
    # For edge points (float): use the edge direction
    # For vertices (int): use incoming and outgoing edges

    # For the fixed shape (incoming direction to snap point)
    fixed_prev = fixed[fixed_prev_idx]

    # For the free shape (outgoing direction from snap point)
    free_next = free[free_curr_idx]

    # Calculate the current angle between the edges
    # The angle is measured from the fixed edge (incoming) to the free edge (outgoing) at ref1_point
    current_angle = angle_between_lines3(fixed_prev, ref1_point, free_next)

    # Calculate rotation needed to achieve the desired angle
    rotation_needed = angle - current_angle

    # Rotate the free object around ref1_point by the computed angle
    free.rotate(rotation_needed, about=ref1_point)

    return free


def fillet_shape_corners(
    shape: Shape, d_vert_radius: dict[int, float], n: int = 12
) -> Shape:
    """Return a copy of ``shape`` with rounded corners.

    Args:
        shape: Source polygon or polyline.
        d_vert_radius: Mapping ``vertex_index -> fillet radius`` for corners to round.
        n: Number of points per fillet arc. Defaults to 12.

    Returns:
        A copy of ``shape`` with fillet vertices substituted.

    Examples:
        >>> import simetri.graphics as sg
        >>> src = sg.square(size=10)
        >>> rounded = sg.fillet_shape_corners(src, {0: 1}, n=4)
        >>> rounded is not src
        True
        >>> len(rounded.vertices) > len(src.vertices)
        True
    """
    vertices = fillet_corners(shape.vertices, d_vert_radius, n)

    new_shape = shape.copy()
    new_shape[:] = vertices

    return new_shape
