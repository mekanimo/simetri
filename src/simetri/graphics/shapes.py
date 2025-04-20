"""Shapes module contains classes and functions for creating shapes."""

from math import pi, gcd, sin, cos, comb
from typing import List, Sequence, Union
import copy

from numpy import ndarray
import numpy as np

from ..graphics.batch import Batch
from ..graphics.bbox import BoundingBox
from ..graphics.shape import Shape, custom_attributes
from ..graphics.common import axis_x, get_defaults, Sequence, Point
from ..graphics.all_enums import Types
from ..helpers.utilities import decompose_transformations
from ..settings.settings import defaults
from .affine import scale_in_place_matrix, rotation_matrix
from ..geometry.ellipse import ellipse_points
from ..geometry.geometry import (
    side_len_to_radius,
    offset_polygon_points,
    distance,
    mid_point,
    close_points2,
)

import simetri.colors.colors as colors

Color = colors.Color


class Rectangle(Shape):
    """A rectangle defined by width and height."""

    def __init__(self, center: Point, width: float, height: float, **kwargs) -> None:
        """Initialize a Rectangle object.

        Args:
            center (Point): The center point of the rectangle.
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

    def scale(
        self,
        scale_x: float,
        scale_y: Union[float, None] = None,
        about: Point = (0, 0),
        reps: int = 0,
    ):
        """Scale the rectangle by scale_x and scale_y.
        Rectangles cannot be scaled non-uniformly.
        scale_x changes the width and scale_y changes the height.

        Args:
            scale_x (float): The scale factor for the width.
            scale_y (float, optional): The scale factor for the height. Defaults to None.
            about (Point, optional): The point to scale about. Defaults to (0, 0).
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Rectangle: The scaled rectangle.
        """
        if scale_y is None:
            scale_y = scale_x
        center = self.center
        _, rotation, _ = decompose_transformations(self.xform_matrix)
        rm = rotation_matrix(-rotation, center)
        sm = scale_in_place_matrix(scale_x, scale_y, about)
        inv_rm = rotation_matrix(rotation, center)
        transform = rm @ sm @ inv_rm

        return self._update(transform, reps=reps)

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
            Point: The center of the rectangle.
        """
        return mid_point(self.vertices[0], self.vertices[2])

    def _set_center(self, new_center: Point):
        """Set the center of the rectangle.

        Args:
            new_center (Point): The new center of the rectangle.
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
        style = copy.copy(self.style)
        rectangle.style = style
        rectangle._set_aliases()
        custom_attribs = custom_attributes(self)
        for attrib in custom_attribs:
            setattr(rectangle, attrib, getattr(self, attrib))

        return rectangle


class Rectangle2(Rectangle):
    """A rectangle defined by two opposite corners."""

    def __init__(self, corner1: Point, corner2: Point, **kwargs) -> None:
        """Initialize a Rectangle2 object.

        Args:
            corner1 (Point): The first corner of the rectangle.
            corner2 (Point): The second corner of the rectangle.
            **kwargs: Additional keyword arguments.
        """
        x1, y1 = corner1
        x2, y2 = corner2
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
        center: Point = (0, 0),
        radius: float = None,
        xform_matrix: np.array = None,
        **kwargs,
    ) -> None:
        """Initialize a Circle object.

        Args:
            center (Point, optional): The center point of the circle. Defaults to (0, 0).
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
            Point: The center of the circle.
        """
        return self.vertices[0]

    @center.setter
    def center(self, value: Point):
        """Set the center of the circle.

        Args:
            value (Point): The new center of the circle.
        """
        self[0] = value[:2]

    @property
    def radius(self):
        """Return the radius of the circle.

        Returns:
            float: The radius of the circle.
        """
        scale_x = np.linalg.norm(self.xform_matrix[0, :2])  # only x scale is used
        return self._radius * scale_x

    def copy(self):
        """Return a copy of the circle.

        Returns:
            Circle: A copy of the circle.
        """
        center = self.center
        radius = self.radius
        circle = Circle(center=center, radius=radius)
        style = copy.deepcopy(self.style)
        circle.style = style
        circle._set_aliases()

        custom_attribs = custom_attributes(self)
        custom_attribs.remove("center")
        custom_attribs.remove("_radius")
        custom_attribs.remove("radius")
        for attrib in custom_attribs:
            setattr(circle, attrib, getattr(self, attrib))

        return circle


class Segment(Shape):
    """A line segment defined by two points.
    This is not used in the code-base, but is here for the API.
    """

    def __init__(self, start: Point, end: Point, **kwargs) -> None:
        """Initialize a Segment object.

        Args:
            start (Point): The start point of the segment.
            end (Point): The end point of the segment.
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
            Point: The start point of the segment.
        """
        return self.vertices[0]

    @property
    def end(self):
        """Return the end point of the segment.

        Returns:
            Point: The end point of the segment.
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
        return Segment(self.start, self.end, **self.kwargs)

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
            points (Sequence[Point]): The points that make up the mask.
            reverse (bool, optional): Whether to reverse the mask. Defaults to False.
            xform_matrix (np.array, optional): The transformation matrix. Defaults to None.
        """
        super().__init__(points, xform_matrix, subtype=Types.MASK, closed=True)
        self.reverse: bool = reverse
        # mask should be between \begin{scope} and \end{scope}
        # canvas, batch, and shapes can have scope


def circle_points(center: Point, radius: float, n: int = 30) -> list[Point]:
    """Return a list of points that form a circle with the given parameters.

    Args:
        center (Point): The center point of the circle.
        radius (float): The radius of the circle.
        n (int, optional): The number of points in the circle. Defaults to 30.

    Returns:
        list[Point]: A list of points that form a circle.
    """
    return arc_points(center, radius, 0, 2 * pi, n=n)


def arc_points(
    center: Point,
    radius: float,
    start_angle: float,
    end_angle: float,
    clockwise: bool = False,
    n: int = 20,
) -> list[Point]:
    """Return a list of points that form a circular arc with the given parameters.

    Args:
        center (Point): The center point of the arc.
        radius (float): The radius of the arc.
        start_angle (float): The starting angle of the arc.
        end_angle (float): The ending angle of the arc.
        clockwise (bool, optional): Whether the arc is drawn clockwise. Defaults to False.
        n (int, optional): The number of points in the arc. Defaults to 20.

    Returns:
        list[Point]: A list of points that form a circular arc.
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
    x: float, y: float, width: float, height: float, angle: float = 0
) -> Sequence[Point]:
    """Return a list of points that form a rectangle with the given parameters.

    Args:
        x (float): The x-coordinate of the center of the rectangle.
        y (float): The y-coordinate of the center of the rectangle.
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
        angle (float, optional): The rotation angle of the rectangle. Defaults to 0.

    Returns:
        Sequence[Point]: A list of points that form the rectangle.
    """
    from affine import rotate

    points = []
    points.append([x - width / 2, y - height / 2])
    points.append([x + width / 2, y - height / 2])
    points.append([x + width / 2, y + height / 2])
    points.append([x - width / 2, y + height / 2])
    if angle != 0:
        points = rotate(points, angle, (x, y))
    return points


def reg_poly_points_side_length(pos: Point, n: int, side_len: float) -> Sequence[Point]:
    """Return a regular polygon points list with n sides and side_len length.

    Args:
        pos (Point): The position of the center of the polygon.
        n (int): The number of sides of the polygon.
        side_len (float): The length of each side of the polygon.

    Returns:
        Sequence[Point]: A list of points that form the polygon.
    """
    rad = side_len_to_radius(n, side_len)
    angle = 2 * pi / n
    x, y = pos[:2]
    points = [[cos(angle * i) * rad + x, sin(angle * i) * rad + y] for i in range(n)]
    points.append(points[0])
    return points


def reg_poly_points(pos: Point, n: int, r: float) -> Sequence[Point]:
    """Return a regular polygon points list with n sides and radius r.

    Args:
        pos (Point): The position of the center of the polygon.
        n (int): The number of sides of the polygon.
        r (float): The radius of the polygon.

    Returns:
        Sequence[Point]: A list of points that form the polygon.
    """
    angle = 2 * pi / n
    x, y = pos[:2]
    points = [[cos(angle * i) * r + x, sin(angle * i) * r + y] for i in range(n)]
    points.append(points[0])
    return points


def di_star(points: Sequence[Point], n: int) -> Batch:
    """Return a dihedral star with n petals.

    Args:
        points (Sequence[Point]): List of [x, y] points.
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
        list[Point]: A list of points that define the centers of the hexagons.
    """
    centers = []
    for row in range(n_rows):
        for col in range(n_cols):
            x_ = col * 3 * side_length + x
            y_ = row * 2 * side_length + y
            if col % 2:
                y_ += side_length
            centers.append((x_, y_))

    centers = []
    # first row
    origin = Point(x, y)
    grid = Batch(Point)
    grid.transform()
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
                x_, y_ = (col * cell_width + x, (n_rows - row - 1) * cell_height + y)
                points = [
                    (x_, y_),
                    (x_ + cell_width, y_),
                    (x_ + cell_width, y_ + cell_height),
                    (x_, y_ + cell_height),
                ]
                cell = Shape(points, closed=True, fill_color=colors.gray)
                grid.append(cell)
    return grid


def regular_star_polygon(n, step, rad):
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
        indices = [i % n for i in list(range(0, ((n // step) + 1) * step, step))]
    vertices = [points[ind] for ind in indices]
    return Batch(Shape(vertices)).rotate(angle, reps=gcd(n, step) - 1)


def star_shape(points, reps=5, scale=1):
    """Return a dihedral star from a list of points.

    Args:
        points (list[Point]): The list of points that form the star.
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
    x,
    y,
    radius=1,
    fill_color=None,
    line_color=None,
    line_width=None,
):
    """Return a Shape object with a single point.

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        radius (float, optional): The radius of the point. Defaults to 1.
        fill_color (Color, optional): The fill color of the point. Defaults to None.
        line_color (Color, optional): The line color of the point. Defaults to None.
        line_width (float, optional): The line width of the point. Defaults to None.

    Returns:
        Shape: A Shape object with a single point.
    """
    fill_color, line_color, line_width = get_defaults(
        ["fill_color", "line_color", "line_width"], [fill_color, line_color, line_width]
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
    x: float,
    y: float,
    width: float,
    height: float,
    fill_color: Color = colors.white,
    line_color: Color = defaults["line_color"],
    line_width: float = defaults["line_width"],
    fill: bool = True,
    marker: "Marker" = None,
) -> Shape:
    """Given lower left corner position, width, and height,
    return a Shape object with points that form a rectangle.

    Args:
        x (float): The x-coordinate of the lower left corner.
        y (float): The y-coordinate of the lower left corner.
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
        fill_color (Color, optional): The fill color of the rectangle. Defaults to colors.white.
        line_color (Color, optional): The line color of the rectangle. Defaults to defaults["line_color"].
        line_width (float, optional): The line width of the rectangle. Defaults to defaults["line_width"].
        fill (bool, optional): Whether to fill the rectangle. Defaults to True.
        marker (Marker, optional): The marker for the rectangle. Defaults to None.

    Returns:
        Shape: A Shape object with points that form a rectangle.
    """
    return Shape(
        [(x, y), (x + width, y), (x + width, y + height), (x, y + height)],
        closed=True,
        fill_color=fill_color,
        line_color=line_color,
        fill=fill,
        line_width=line_width,
        marker=marker,
        subtype=Types.RECTANGLE,
    )


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
    points = arc_points(x, y, radius, start_angle, end_angle, clockwise=clockwise, n=n)
    return Shape(points, closed=False, subtype=Types.ARC)


def circle_shape(x, y, radius, n=30):
    """Return a Shape object with points that form a circle with the given parameters.

    Args:
        x (float): The x-coordinate of the center of the circle.
        y (float): The y-coordinate of the center of the circle.
        radius (float): The radius of the circle.
        n (int, optional): The number of points to use for the circle. Defaults to 30.

    Returns:
        Shape: A Shape object with points that form a circle.
    """
    circ = arc_shape(x, y, radius, 0, 2 * pi, n=n)
    circ.subtype = Types.CIRCLE
    return circ


def reg_poly_shape(pos, n, r=100, **kwargs):
    """Return a regular polygon.

    Args:
        pos (Point): The position of the center of the polygon.
        n (int): The number of sides of the polygon.
        r (float, optional): The radius of the polygon. Defaults to 100.
        kwargs (dict): Additional keyword arguments.

    Returns:
        Shape: A Shape object with points that form a regular polygon.
    """
    x, y = pos[:2]
    points = reg_poly_points((x, y), n=n, r=r)
    return Shape(points, closed=True, **kwargs)


def ellipse_shape(x, y, width, height, angle, n_points=None):
    """Return a Shape object with points that form an ellipse with the given parameters.

    Args:
        x (float): The x-coordinate of the center of the ellipse.
        y (float): The y-coordinate of the center of the ellipse.
        width (float): The width of the ellipse.
        height (float): The height of the ellipse.
        n (int, optional): The number of points to use for the ellipse. Defaults to 30.

    Returns:
        Shape: A Shape object with points that form an ellipse.
    """
    if n_points is None:
        n_points = defaults["n_ellipse_points"]

    points = ellipse_points((x, y), width, height, n_points=n)
    return Shape(points, subtype=Types.ELLIPSE)


def line_shape(p1, p2, line_width=1, line_color=colors.black, **kwargs):
    """Return a Shape object with two points p1 and p2.

    Args:
        p1 (Point): The first point of the line.
        p2 (Point): The second point of the line.
        line_width (float, optional): The width of the line. Defaults to 1.
        line_color (Color, optional): The color of the line. Defaults to colors.black.
        kwargs (dict): Additional keyword arguments.

    Returns:
        Shape: A Shape object with two points that form a line.
    """
    x1, y1 = p1
    x2, y2 = p2
    return Shape(
        [(x1, y1), (x2, y2)],
        closed=False,
        line_color=line_color,
        line_width=line_width,
        subtype=Types.L_i_n_e,
        **kwargs,
    )


def offset_polygon_shape(
    polygon_shape, offset: float = 1, dist_tol: float = defaults["dist_tol"]
) -> list[Point]:
    """Return a copy of a polygon with offset edges.

    Args:
        polygon_shape (Shape): The original polygon shape.
        offset (float, optional): The offset distance. Defaults to 1.
        dist_tol (float, optional): The distance tolerance. Defaults to defaults["dist_tol"].

    Returns:
        list[Point]: A list of points that form the offset polygon.
    """
    vertices = offset_polygon_points(polygon_shape.vertices, offset, dist_tol)

    return Shape(vertices)
