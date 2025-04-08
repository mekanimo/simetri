"""Provides facilities for working with grids of cells."""

from itertools import product
from math import sin, cos, pi, sqrt
from typing import Sequence

from ..helpers.utilities import reg_poly_points
from ..geometry.geometry import intersect, cartesian_to_polar
from ..graphics.common import Point


class CircularGrid:
    """A grid formed by connections of regular polygon points."""

    def __init__(self, center: Point = (0, 0), n: int = 12, radius: float = 100):
        """
        Initializes the grid with the given center, radius, number of rows, and number of columns.

        Args:
            center (Point): The center point of the grid.
            n (int): The number of points in the regular polygon.
            radius (float): The radius of the grid.
        """
        self.center = center
        self.radius = radius
        self.n = n
        self.points = reg_poly_points(center, n, radius)

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

    def line(self, ind1:int, ind2:int)->tuple:
        """
        Returns the line connecting the given indices.

        Args:
            ind1 (int): The first index.
            ind2 (int): The second index.

        Returns:
            tuple: The line connecting the two points.
        """
        return (self.points[ind1], self.points[ind2])


class SquareGrid:
    """A grid formed by connections of square cells."""

    def __init__(self, center: Point = (0, 0), n: int = 16, cell_size: float = 25):
        """
        Initializes the grid with the given center, number of rows, number of columns, and cell size.

        Args:
            center (Point): The center point of the grid.
            n (int): The number of points in the grid. Square of an even integer.
            cell_size (float): The size of each cell in the grid.
        """
        self.center = center
        self.n = n
        self.cell_size = cell_size

        self.points = []
        # points are around the grid, not in the grid

        hs = int(sqrt(n)//2) # half size
        c = cell_size
        vals = [c*x for x in range(-hs, hs+1)]
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
        self.points = coords


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

    def line(self, ind1:int, ind2:int)->tuple:
        """
        Returns the line connecting the given indices.

        Args:
            ind1 (int): The first index.
            ind2 (int): The second index.

        Returns:
            tuple: The line connecting the two points.
        """
        return (self.points[ind1], self.points[ind2])

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


# basis = ((1, 0), (cos(pi/3), sin(pi/3)))

# print(convert_basis(1, 0, basis))  # (1.0, 0.0)

# basis = ((1, 0), (cos(pi/3), sin(pi/3)))
# print(convert_to_cartesian(1, 1, basis))  # (1.0, 0.0)

# print(cartesian_to_isometric(1, 0))  # (1, 0.5)

# print(isometric_to_cartesian(1, 1)) # (1.5, 0.866)
