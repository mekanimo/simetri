"""Provides facilities for working with grids of cells."""

from itertools import product
from math import sin, cos, pi, sqrt
from typing import Sequence

from ..helpers.utilities import reg_poly_points
from ..geometry.geometry import (
    intersect,
    cartesian_to_polar,
    polar_to_cartesian,
    lerp_point,
)
from ..geometry.circle import Circle
from ..graphics.common import Point, common_properties
from ..graphics.batch import Batch
from ..graphics.shape import Shape
from ..graphics.all_enums import Types, GridType
from ..colors.colors import gray


d_grid_types = {
    GridType.CIRCULAR: Types.CIRCULAR_GRID,
    GridType.SQUARE: Types.SQUARE_GRID,
    GridType.HEXAGONAL: Types.HEX_GRID,
    GridType.MIXED: Types.MIXED_GRID,
}


class Grid(Batch):
    """A base-class for all grids."""

    def __init__(
        self,
        grid_type: GridType,
        center: Point = (0, 0),
        n: int = 12,
        radius: float = 100,
        points: Sequence[Point] = None,
        n_circles=1,
    ):
        if grid_type not in d_grid_types:
            raise ValueError(f"Invalid grid type: {grid_type}.")
        super().__init__(subtype=d_grid_types[grid_type])
        common_properties(self)
        self.center = center
        self.radius = radius
        self.n = n
        self.n_circles = n_circles
        pairs = product(points, repeat=2)
        for point1, point2 in pairs:
            if point1 != point2:
                self.append(Shape([point1, point2], line_color=gray))
        self.points = points if points else []
        for i, point in enumerate(self.points):
            next_point = self.points[(i + 1) % len(self.points)]
            self.append(Shape([point, next_point]))

    def intersect(self, line1: Sequence[int], line2: Sequence[int]):
        """
        Returns the intersection of the lines connecting the given indices.

        Args:
            line1 (Sequence[int]): A sequence containing two indices (ind1, ind2).
            line2 (Sequence[int]): A sequence containing two indices (ind3, ind4).

        Returns:
            tuple: (x, y) intersection point of the lines.
        """
        ind1, ind2 = line1
        ind3, ind4 = line2

        line1 = (self.points[ind1], self.points[ind2])
        line2 = (self.points[ind3], self.points[ind4])

        return intersect(line1, line2)

    def line(self, ind1: int, ind2: int) -> tuple:
        """
        Returns the line connecting the given indices.

        Args:
            ind1 (int): The first index.
            ind2 (int): The second index.

        Returns:
            tuple: The line connecting the two points.
        """
        return (self.points[ind1], self.points[ind2])

    def radial_point(self, radius, index: int):
        """
        Returns the point on the line connecting the center of the grid to the given index.
        radius is the distance from the center to the point.
        The index is the index of the point in the grid.

        Args:
            radius (float): The radius.
            index (int): The index of the point.

        Returns:
            tuple: The polar point.
        """
        return polar_to_cartesian(radius, index * (2 * pi / self.n))

    def between(self, ind1: int, ind2, t: float = 0.5) -> Point:
        """
        Returns the point on the line connecting the given indices interpolated
        by using the given t parameter.

        Args:
            ind1 (int): The first index.
            ind2 (int): The second index.
            t (float): The parameter used for interpolation. Default is 0.5.

        Returns:
            Point: The point on the line connecting the two points.
        """
        if t < 0 or t > 1:
            raise ValueError("t must be between 0 and 1.")
        if ind1 < 0 or ind1 >= len(self.points):
            raise ValueError(f"ind1 must be between 0 and {len(self.points) - 1}.")
        if ind1 == ind2:
            raise ValueError("ind1 and ind2 must be different.")

        return lerp_point(self.points[ind1], self.points[ind2], t)


class CircularGrid(Grid):
    """A grid formed by connections of regular polygon points."""

    def __init__(
        self, center: Point = (0, 0), n: int = 12, radius: float = 100, n_circles=1
    ):
        """
        Initializes the grid with the given center, radius, number of rows, and number of columns.

        Args:
            center (Point): The center point of the grid.
            n (int): The number of points in the regular polygon.
            radius (float): The radius of the grid.
            n_circles (int): The number of circles in the grid. Used for drawing the grid.
        """
        points = reg_poly_points(center, n, radius)
        super().__init__(GridType.CIRCULAR, center, n, radius, points, n_circles)

        self.append(Circle(center, radius, fill=False))

class HexGrid(Grid):
    """A grid formed by connections of regular polygon points."""

    def __init__(self, center: Point = (0, 0), radius: float = 100, n_circles=1):
        """
        Initializes the grid with the given center, radius, number of rows, and number of columns.

        Args:
            center (Point): The center point of the hexagon.
            radius (float): The circumradius of the hexagon.
            n_circles (int): The number of circles in the grid. Used for drawing the grid.
        """
        points = reg_poly_points(center, 6, radius)
        super().__init__(GridType.HEXAGONAL, center, 6, radius, points, n_circles)


class SquareGrid(Grid):
    """A grid formed by connections of square cells."""

    def __init__(self, center: Point = (0, 0), n: int = 16, cell_size: float = 25):
        """
        Initializes the grid with the given center, number of rows, number of columns, and cell size.

        Args:
            center (Point): The center point of the grid.
            n (int): The number of points in the grid. Square of an even integer.
            cell_size (float): The size of each cell in the grid.
        """
        self.cell_size = cell_size
        hs = int(sqrt(n) // 2)  # half size
        c = cell_size
        vals = [c * x for x in range(-hs, hs + 1)]
        coords = list(product(vals, repeat=2))

        def sort_key(coord):
            r, _ = cartesian_to_polar(*coord)
            return r

        def sort_key2(coord):
            _, theta = cartesian_to_polar(*coord)
            return theta

        coords.sort(key=sort_key, reverse=True)
        coords = coords[:n]
        coords.sort(key=sort_key2)
        points = coords
        radius = (2 * (cell_size * sqrt(n))**2)**.5
        super().__init__(GridType.SQUARE, center, n, radius, points)

# change of basis conversion


def convert_basis(x: float, y: float, basis: tuple):
    """
    Converts the given (x, y) coordinates from the standard basis to the given basis.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        basis (tuple): The basis to convert to.

    Returns:
        tuple: The converted (x, y) coordinates.
    """
    return basis[0][0] * x + basis[0][1] * y, basis[1][0] * x + basis[1][1] * y


def convert_to_cartesian(x: float, y: float, basis: tuple):
    """
    Converts the given (x, y) coordinates from the given basis to the standard basis.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        basis (tuple): The basis to convert from.

    Returns:
        tuple: The converted (x, y) coordinates.
    """
    return basis[0][0] * x + basis[1][0] * y, basis[0][1] * x + basis[1][1] * y


def cartesian_to_isometric(x: float, y: float):
    """
    Converts the given (x, y) coordinates to isometric coordinates.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        tuple: The isometric (x, y) coordinates.
    """
    return convert_basis(x, y, ((1, 0), (cos(pi / 3), sin(pi / 3))))


def isometric_to_cartesian(x: float, y: float):
    """
    Converts the given isometric (x, y) coordinates to Cartesian coordinates.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        tuple: The Cartesian (x, y) coordinates.
    """
    return convert_to_cartesian(x, y, ((1, 0), (cos(pi / 3), sin(pi / 3))))
