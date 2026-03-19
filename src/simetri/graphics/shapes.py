"""Shapes module contains classes and functions for creating shapes."""

from math import pi, gcd, sin, cos, comb
from typing import List, Sequence, Union
import copy

from numpy import ndarray
import numpy as np

from ..graphics.batch import Batch
from ..graphics.bbox import BoundingBox
from ..graphics.shape import (
    Shape,
    custom_attributes,
    clip,
    trim_margins,
    all_segments,
    get_loop,
    get_partition,
    union,
    diff,
    xor,
)
from ..graphics.common import axis_x, get_defaults, PointType, LineType
from ..graphics.all_enums import Types, Extent
from ..helpers.utilities import decompose_transformations
from ..settings.settings import defaults
from .affine import scale_in_place_matrix, rotation_matrix
from ..geometry.ellipse import ellipse_points
from ..geometry.geometry import (
    side_len_to_radius,
    offset_polygon_points,
    distance,
    midpoint,
    close_points2,
    lerp_point,
    angle_between_lines2,
)
from ..geometry.vectors import v_scale, v_diff, v_sum
from ..canvas.style_map import (
    line_style_map,
    shape_style_map,
    tag_style_map,
    image_style_map,
    group_args,
    StyleObj,
)

import simetri.colors.colors as colors

Color = colors.Color


def square(
    center: PointType = (0, 0), size: float = 100, angle: float = 0, **kwargs
) -> Shape:
    """Return a square shape.

    Args:
        center (PointType): The center of the square.
        size (float): The width and height of the square.
        angle (float): The rotation angle of the square.
        **kwargs: Additional keyword arguments.

    Returns:
        Shape: A square shape.
    """
    points = rectangle_points(center, size, size, angle)

    return Shape(points, closed=True, **kwargs)


class Line(Shape):
    """A line defined by two points.
    They are drawn different depending on their types.
    line.extent == sg.Extent.INFINITE drawn to the canvas limits
    on both sides.
    line.extent == sg.Extent.RAY drawn from the first point to the
    canvas.limit.
    line.extent == sg.Extent.SEGMENT drawn from the first point to
    the second point as a line segment.
    For representing the line in Ax + By + C = 0 (general form) use:
    line.A -> A, line.B -> B, line.C -> C, and line.ABC -> (A, B, C)
    For representing the line in mx + b = 0 (slope and intercept form) use:
    line.slope -> m and line.intercept -> b, and line.m_b -> (m, b)
    line.parametric_function Return a callable f(t) that gives points on the line.
    line.t returns an interpolation using the parametric form of the line.
    """

    def __init__(
        self,
        start: PointType,
        end: PointType,
        extent: Extent = Extent.SEGMENT,
        draw_type: Extent = None,
        **kwargs,
    ) -> None:
        """Initialize a Line object.

        Args:
            start (PointType): The start point of the line.
            end (PointType): The end point of the line.
            extent (Extent, optional): Rendering mode. Defaults to Extent.SEGMENT.
            draw_type (Extent, optional): Backward-compatible alias for extent.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If start and end points are the same.
        """
        dist_tol2 = defaults["dist_tol"] ** 2
        if close_points2(start, end, dist2=dist_tol2):
            raise ValueError("Line: start and end points are the same!")

        if draw_type is not None:
            extent = draw_type

        if not isinstance(extent, Extent):
            extent = Extent(extent)

        super().__init__([start, end], closed=False, **kwargs)
        self.subtype = Types.LINE
        self.extent = extent

    def __setattr__(self, name, value):
        if name == "draw_type":
            name = "extent"
        super().__setattr__(name, value)

    @property
    def draw_type(self) -> Extent:
        """Backward-compatible alias for extent."""
        return self.extent

    @draw_type.setter
    def draw_type(self, value: Extent):
        self.extent = value

    @property
    def start(self) -> PointType:
        """Return the start point of the line."""
        return self.vertices[0]

    @start.setter
    def start(self, point: PointType):
        self[0] = point[:2]

    @property
    def end(self) -> PointType:
        """Return the end point of the line."""
        return self.vertices[1]

    @end.setter
    def end(self, point: PointType):
        self[1] = point[:2]

    @property
    def length(self) -> float:
        """Return the length between start and end points."""
        return distance(self.start, self.end)

    @property
    def A(self) -> float:
        """Return A coefficient of Ax + By + C = 0."""
        x1, y1 = self.start[:2]
        x2, y2 = self.end[:2]
        return y1 - y2

    @property
    def B(self) -> float:
        """Return B coefficient of Ax + By + C = 0."""
        x1, y1 = self.start[:2]
        x2, y2 = self.end[:2]
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
    def parametric_function(self):
        """Return a callable f(t) that gives points on the line.

        For SEGMENT lines, meaningful values are typically 0 <= t <= 1.
        """
        return lambda t: self.t(t)

    def copy(self):
        """Return a copy of the line."""
        line = Line(self.start, self.end, extent=self.extent)
        for attrib in custom_attributes(self):
            if attrib in ["vertices", "draw_type", "extent"]:
                continue
            setattr(line, attrib, getattr(self, attrib))
        for attrib in shape_style_map:
            value = getattr(self, attrib, defaults[attrib])
            if value is not None:
                setattr(line, attrib, value)
        return line

    def t(self, t: float):
        """Return point at parameter t using start + t * (end - start)."""
        direction = v_diff(self.end, self.start)

        return v_sum(self.start, v_scale(direction, t))


class Rectangle(Shape):
    """A rectangle defined by width and height."""

    def __init__(
        self, center: PointType, width: float, height: float, **kwargs
    ) -> None:
        """Initialize a Rectangle object.

        Args:
            center (PointType): The center point of the rectangle.
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.
            **kwargs: Additional keyword arguments.
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

    def __setattr__(self, name, value):
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
    #     scale_y: Union[float, None] = None,
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
    def width(self):
        """Return the width of the rectangle.

        Returns:
            float: The width of the rectangle.
        """
        return distance(self.vertices[0], self.vertices[1])

    def _set_width(self, new_width: float):
        """Set the width of the rectangle.

        Args:
            new_width (float): The new width of the rectangle.
        """
        scale_x = new_width / self.width
        self.scale(scale_x, 1, about=self.center, reps=0)

    @property
    def height(self):
        """Return the height of the rectangle.

        Returns:
            float: The height of the rectangle.
        """
        return distance(self.vertices[1], self.vertices[2])

    def _set_height(self, new_height: float):
        """Set the height of the rectangle.

        Args:
            new_height (float): The new height of the rectangle.
        """
        scale_y = new_height / self.height
        self.scale(1, scale_y, about=self.center, reps=0)

    @property
    def center(self):
        """Return the center of the rectangle.

        Returns:
            PointType: The center of the rectangle.
        """
        return midpoint(self.vertices[0], self.vertices[2])

    def _set_center(self, new_center: PointType):
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

    def copy(self):
        """Return a copy of the rectangle.

        Returns:
            Rectangle: A copy of the rectangle.
        """
        center = self.center
        width = self.width
        height = self.height
        rectangle = Rectangle(center, width, height)
        _, rotation, _ = decompose_transformations(self.xform_matrix)
        rectangle.rotate(rotation, about=center, reps=0)
        rectangle._set_aliases()
        custom_attribs = custom_attributes(self)
        for attrib in custom_attribs:
            if attrib.startswith("_"):
                continue
            if hasattr(self, attrib):
                setattr(rectangle, attrib, getattr(self, attrib))
        for attrib in shape_style_map:
            value = getattr(self, attrib, defaults[attrib])
            if value is not None:
                setattr(rectangle, attrib, value)
        return rectangle


class Rectangle2(Rectangle):
    """A rectangle defined by two opposite corners."""

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
    """A circle defined by a center point and a radius."""

    def __init__(
        self,
        radius: float = None,
        center: PointType = (0, 0),
        xform_matrix: np.array = None,
        **kwargs,
    ) -> None:
        """Initialize a Circle object.

        Args:
            center (PointType, optional): The center point of the circle. Defaults to (0, 0).
            radius (float, optional): The radius of the circle. Defaults to None.
            xform_matrix (np.array, optional): The transformation matrix. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        if radius is None:
            radius = defaults["circle_radius"]

        x, y = center[:2]
        points = [[x, y]]
        super().__init__(points, xform_matrix=xform_matrix, **kwargs)
        self.subtype = Types.CIRCLE
        self._radius = radius

    def __setattr__(self, name, value):
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
    def b_box(self):
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
    def closed(self):
        """Return True. Circles are closed.

        Returns:
            bool: True
        """
        return True

    @closed.setter
    def closed(self, value: bool):
        pass

    @property
    def center(self):
        """Return the center of the circle.

        Returns:
            PointType: The center of the circle.
        """
        return self.vertices[0]

    @center.setter
    def center(self, value: PointType):
        """Set the center of the circle.

        Args:
            value (PointType): The new center of the circle.
        """
        self[0] = value[:2]

    @property
    def radius(self):
        """Return the radius of the circle.

        Returns:
            float: The radius of the circle.
        """
        scale_x = np.linalg.norm(
            self.xform_matrix[0, :2]
        )  # only x scale is used
        return self._radius * scale_x

    def copy(self):
        """Return a copy of the circle.

        Returns:
            Circle: A copy of the circle.
        """

        center = self.center
        radius = self.radius
        circle = Circle(center=center, radius=radius)
        # style = copy.deepcopy(self.style)
        # circle.style = style
        circle._set_aliases()

        custom_attribs = custom_attributes(self)
        custom_attribs.remove("center")
        custom_attribs.remove("_radius")
        custom_attribs.remove("radius")
        for attrib in custom_attribs:
            setattr(circle, attrib, getattr(self, attrib))
        for attrib in shape_style_map:
            value = getattr(self, attrib, defaults[attrib])
            if value is not None:
                setattr(circle, attrib, value)

        return circle


class Segment(Shape):
    """A line segment defined by two points.
    This is not used in the code-base, but is here for the API.
    """

    def __init__(self, start: PointType, end: PointType, **kwargs) -> None:
        """Initialize a Segment object.

        Args:
            start (PointType): The start point of the segment.
            end (PointType): The end point of the segment.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the start and end points are the same.
        """
        dist_tol2 = defaults["dist_tol"] ** 2
        if close_points2(start, end, dist2=dist_tol2):
            raise ValueError("Segment: start and end points are the same!")
        points = [start, end]
        super().__init__(points, **kwargs)
        self.subtype = Types.SEGMENT

    @property
    def start(self):
        """Return the start point of the segment.

        Returns:
            PointType: The start point of the segment.
        """
        return self.vertices[0]

    @property
    def end(self):
        """Return the end point of the segment.

        Returns:
            PointType: The end point of the segment.
        """
        return self.vertices[1]

    @property
    def length(self):
        """Return the length of the segment.

        Returns:
            float: The length of the segment.
        """
        return distance(self.start, self.end)

    def copy(self) -> Shape:
        """Return a copy of the segment.

        Returns:
            Shape: A copy of the segment.
        """
        segment = Segment(self.start, self.end)
        custom_attribs = custom_attributes(self)
        for attrib in custom_attribs:
            if attrib.startswith("_"):
                continue
            if hasattr(self, attrib):
                setattr(segment, attrib, getattr(self, attrib))
        for attrib in shape_style_map:
            value = getattr(self, attrib, defaults[attrib])
            if value is not None:
                setattr(segment, attrib, value)
        return segment

    def __str__(self):
        """Return a string representation of the segment.

        Returns:
            str: The string representation of the segment.
        """
        return f"Segment({self.start}, {self.end})"

    def __repr__(self):
        """Return a string representation of the segment.

        Returns:
            str: The string representation of the segment.
        """
        return f"Segment({self.start}, {self.end})"

    def __eq__(self, other):
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


class Mask(Shape):
    """A mask is a closed shape that is used to clip other shapes.
    All it has is points and a transformation matrix.
    """

    def __init__(self, points, reverse=False, xform_matrix=None):
        """Initialize a Mask object.

        Args:
            points (Sequence[PointType]): The points that make up the mask.
            reverse (bool, optional): Whether to reverse the mask. Defaults to False.
            xform_matrix (np.array, optional): The transformation matrix. Defaults to None.
        """
        super().__init__(
            points=points,
            closed=True,
            xform_matrix=xform_matrix,
            subtype=Types.MASK,
        )
        self.reverse: bool = reverse
        # mask should be between \begin{scope} and \end{scope}
        # canvas, batch, and shapes can have scope


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
    """
    x, y = center[:2]
    points = []
    if clockwise:
        start_angle, end_angle = end_angle, start_angle
    step = (end_angle - start_angle) / n
    for i in np.arange(start_angle, end_angle + 1, step):
        points.append([x + radius * cos(i), y + radius * sin(i)])
    return points


def hex_points(side_length: float) -> List[List[float]]:
    """Return a list of points that define a hexagon with a given side length.

    Args:
        side_length (float): The length of each side of the hexagon.

    Returns:
        list[list[float]]: A list of points that define the hexagon.
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
    """Return a list of points that form a rectangle with the given parameters.

    Args:
        pos (PointType): The position of the rectangle.
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
        angle (float, optional): The rotation angle of the rectangle. Defaults to 0.

    Returns:
        Sequence[PointType]: A list of points that form the rectangle.
    """
    from ..graphics.affine import rotate

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
    pos: PointType, n: int, side_len: float
) -> Sequence[PointType]:
    """Return a regular polygon points list with n sides and side_len length.

    Args:
        pos (PointType): The position of the center of the polygon.
        n (int): The number of sides of the polygon.
        side_len (float): The length of each side of the polygon.

    Returns:
        Sequence[PointType]: A list of points that form the polygon.
    """
    rad = side_len_to_radius(n, side_len)
    angle = 2 * pi / n
    x, y = pos[:2]
    points = [
        [cos(angle * i) * rad + x, sin(angle * i) * rad + y] for i in range(n)
    ]
    points.append(points[0])
    return points


def reg_poly_points(pos: PointType, n: int, r: float) -> Sequence[PointType]:
    """Return a regular polygon points list with n sides and radius r.

    Args:
        pos (PointType): The position of the center of the polygon.
        n (int): The number of sides of the polygon.
        r (float): The radius of the polygon.

    Returns:
        Sequence[PointType]: A list of points that form the polygon.
    """
    angle = 2 * pi / n
    x, y = pos[:2]
    points = [
        [cos(angle * i) * r + x, sin(angle * i) * r + y] for i in range(n)
    ]
    points.append(points[0])
    return points


def di_star(points: Sequence[PointType], n: int) -> Batch:
    """Return a dihedral star with n petals.

    Args:
        points (Sequence[PointType]): List of [x, y] points.
        n (int): Number of petals.

    Returns:
        Batch: A Batch instance (dihedral star with n petals).
    """
    batch = Batch(Shape(points))
    return batch.mirror(axis_x, reps=1).rotate(2 * pi / n, reps=n - 1)


def hex_grid_centers(x, y, side_length, n_rows, n_cols):
    """Return a list of points that define the centers of hexagons in a grid.

    Args:
        x (float): The x-coordinate of the starting point.
        y (float): The y-coordinate of the starting point.
        side_length (float): The length of each side of the hexagons.
        n_rows (int): The number of rows in the grid.
        n_cols (int): The number of columns in the grid.

    Returns:
        list[PointType]: A list of points that define the centers of the hexagons.
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


def rect_grid(x, y, cell_width, cell_height, n_rows, n_cols, pattern):
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
        Batch: A Batch object representing the grid.
    """
    width = cell_width * n_cols
    height = cell_height * n_rows
    horiz_line = line_shape((x, y), (x + width, y))
    horiz_lines = Batch(horiz_line)
    horiz_lines.translate(0, cell_height, reps=n_rows)
    vert_line = line_shape((x, y), (x, y + height))
    vert_lines = Batch(vert_line)
    vert_lines.translate(cell_width, 0, reps=n_cols)
    grid = Batch(horiz_lines, *vert_lines)
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


def reg_star_polygon(n, step, rad, **kwargs) -> Shape | Batch:
    """
    Return a regular star polygon with the given parameters.

    :param n: The number of vertices of the star polygon.
    :type n: int
    :param step: The step size for connecting vertices.
    :type step: int
    :param rad: The radius of the star polygon.
    :type rad: float
    :return: A Batch object representing the star polygon.
    :rtype: Batch
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
        res = Batch(shape.rotate(angle, reps=reps))
    else:
        res = shape

    return res


def star_shape(points, reps=5, scale=1):
    """Return a dihedral star from a list of points.

    Args:
        points (list[PointType]): The list of points that form the star.
        reps (int, optional): The number of repetitions. Defaults to 5.
        scale (float, optional): The scale factor. Defaults to 1.

    Returns:
        Batch: A Batch object representing the star.
    """
    shape = Shape(points, subtype=Types.STAR)
    batch = Batch(shape)
    batch.mirror(axis_x, reps=1)
    batch.rotate(2 * pi / (reps), reps=reps - 1)
    batch.scale(scale)
    return batch


def dot_shape(
    radius=1,
    pos=(0, 0),
    fill_color=None,
    line_color=None,
    line_width=None,
):
    """Return a Shape object with a single point.

    Args:
        radius (float, optional): The radius of the point. Defaults to 1.
        pos (PointType, optional): The position of the point. Defaults to (0, 0).
        fill_color (Color, optional): The fill color of the point. Defaults to None.
        line_color (Color, optional): The line color of the point. Defaults to None.
        line_width (float, optional): The line width of the point. Defaults to None.

    Returns:
        Shape: A Shape object with a single point.
    """
    fill_color, line_color, line_width = get_defaults(
        ["fill_color", "line_color", "line_width"],
        [fill_color, line_color, line_width],
    )
    dot_shape = Shape(
        [(x, y)],
        closed=True,
        fill_color=fill_color,
        line_color=line_color,
        line_width=line_width,
        subtype=Types.D_o_t,
    )
    dot_shape.marker = radius
    return dot_shape


def rect_shape(
    width: float,
    height: float,
    pos: PointType = (0, 0),
    fill_color: Color = colors.white,
    line_color: Color = defaults["line_color"],
    line_width: float = defaults["line_width"],
    fill: bool = True,
    marker: "Marker" = None,
    **kwargs,
) -> Shape:
    """Given lower left corner position, width, and height,
    return a Shape object with points that form a rectangle.

    Args:
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
        pos (PointType, optional): The position of the lower left corner of the rectangle. Defaults to (0, 0).
        fill_color (Color, optional): The fill color of the rectangle. Defaults to colors.white.
        line_color (Color, optional): The line color of the rectangle. Defaults to defaults["line_color"].
        line_width (float, optional): The line width of the rectangle. Defaults to defaults["line_width"].
        fill (bool, optional): Whether to fill the rectangle. Defaults to True.
        marker (Marker, optional): The marker for the rectangle. Defaults to None.

    Returns:
        Shape: A Shape object with points that form a rectangle.
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


def arc_shape(x, y, radius, start_angle, end_angle, clockwise=False, n=20):
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
    """
    points = arc_points(
        (x, y), radius, start_angle, end_angle, clockwise=clockwise, n=n
    )
    return Shape(points, closed=False, subtype=Types.ARC)


def circle_shape(radius, pos=(0, 0), n=30, **kwargs):
    """Return a Shape object with points that form a circle with the given parameters.

    Args:
        radius (float): The radius of the circle.
        pos (PointType, optional): The position of the center of the circle. Defaults to (0, 0).
        n (int, optional): The number of points to use for the circle. Defaults to 30.

    Returns:
        Shape: A Shape object with points that form a circle.
    """
    x, y = pos[:2]
    points = circle_points((x, y), radius, n=n)
    return Shape(points, closed=True, **kwargs)


def reg_poly_shape(n, r=100, pos=(0, 0), **kwargs):
    """Return a regular polygon.

    Args:
        n (int): The number of sides of the polygon.
        r (float, optional): The radius of the polygon. Defaults to 100.
        kwargs (dict): Additional keyword arguments.
        pos (PointType): The position of the center of the polygon.

    Returns:
        Shape: A Shape object with points that form a regular polygon.
    """
    x, y = pos[:2]
    points = reg_poly_points((x, y), n=n, r=r)
    return Shape(points, closed=True, **kwargs)


def ellipse_shape(width, height, angle=0, pos=(0, 0), n_points=None, **kwargs):
    """Return a Shape object with points that form an ellipse with the given parameters.

    Args:
        width (float): The width of the ellipse.
        height (float): The height of the ellipse.
        angle (float, optional): The rotation angle of the ellipse. Defaults to 0.
        pos (PointType, optional): The position of the center of the ellipse. Defaults to (0, 0).
        n_points (int, optional): The number of points to use for the ellipse. Defaults to 30.

    Returns:
        Shape: A Shape object with points that form an ellipse.
    """
    if n_points is None:
        n_points = defaults["n_ellipse_points"]

    points = ellipse_points(pos, width, height, angle, n_points=n_points)
    return Shape(points, subtype=Types.ELLIPSE, **kwargs)


def line_shape(p1, p2, line_width=1, line_color=colors.black, **kwargs):
    """Return a Shape object with two points p1 and p2.

    Args:
        p1 (PointType): The first point of the line.
        p2 (PointType): The second point of the line.
        line_width (float, optional): The width of the line. Defaults to 1.
        line_color (Color, optional): The color of the line. Defaults to colors.black.
        kwargs (dict): Additional keyword arguments.

    Returns:
        Shape: A Shape object with two points that form a line.
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
    polygon_shape, offset: float = 1, dist_tol: float = defaults["dist_tol"]
) -> list[PointType]:
    """Return a copy of a polygon with offset edges.

    Args:
        polygon_shape (Shape): The original polygon shape.
        offset (float, optional): The offset distance. Defaults to 1.
        dist_tol (float, optional): The distance tolerance. Defaults to defaults["dist_tol"].

    Returns:
        list[PointType]: A list of points that form the offset polygon.
    """
    vertices = offset_polygon_points(polygon_shape.vertices, offset, dist_tol)

    return Shape(vertices)


def snap(
    free_shape: Shape,
    ref1: Union[int, float],
    fixed_shape: Shape,
    ref2: Union[int, float],
    angle: float = 0,
):
    """Snaps the given free Shape to the fixed Shape using integer indices (for vertices) or floating point numbers for Barycentric edge coordinates.
    For closed shapes (polygons), indices need to be in counterclockwise order.
    Both references need to be vertices or points on edges.
    When the angle is zero then the objects are attached with edge to edge
    condition.
    Example:
    sg.snap(free=triangle, 0, square, 1, angle=sg.pi/4)
    angle is computed from the triangle[1], square[1], square[0]
    A floating point number indicates a point on an edge in Barycentric
    coordinates. For example,
    1.5 indicates the midpoint between the second and third vertices (zero
    based indexing) or the midpoint of the second edge.
    """

    def get_edge_indices(
        shape: Shape, ref: Union[int, float]
    ) -> tuple[int, int]:
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
        else:
            raise TypeError(f"Invalid reference type: {type(ref)}")

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
    free_prev_idx, free_curr_idx = get_edge_indices(free, ref1)
    fixed_prev_idx, fixed_curr_idx = get_edge_indices(fixed, ref2)

    # Get the vertices needed for angle calculation
    # For free shape: get the next vertex after curr (the outgoing edge)
    n_free_vertices = len(free.vertices)
    if free.closed:
        free_next_idx = (free_curr_idx + 1) % n_free_vertices
    else:
        free_next_idx = free_curr_idx + 1
    free_next = free[free_next_idx]
    fixed_prev = fixed[fixed_prev_idx]

    # Calculate the current angle between the edges
    # The angle is measured from the fixed edge (incoming) to the free edge (outgoing) at ref1_point
    current_angle = angle_between_lines2(fixed_prev, ref1_point, free_next)

    # Calculate rotation needed to achieve the desired angle
    rotation_needed = angle - current_angle

    # Rotate the free object around ref1_point by the computed angle
    free.rotate(rotation_needed, about=ref1_point)

    return free
