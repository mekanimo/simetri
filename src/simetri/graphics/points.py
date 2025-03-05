"""Shape object uses the Points class to store the coordinates of the points that make up the shape.
The Points class is a container for coordinates of multiple points.
It provides conversion to homogeneous coordinates in nd_arrays.
Shape.final_coords is computed by using the Points.homogen_coords property."""

import logging
import copy
from typing import Sequence

from numpy import allclose, ndarray
from typing_extensions import Self

from ..geometry.geometry import homogenize
from .common import Point, common_properties
from .all_enums import *
from ..settings.settings import defaults


class Points:
    """Container for coordinates of multiple points. They provide conversion to homogeneous
    coordinates in nd_arrays. Used in Shape objects.
    """

    def __init__(self, coords: Sequence[Point] = None) -> None:
        """Initialize a Points object.

        Args:
            coords (Sequence[Point], optional): The coordinates of the points. Defaults to None.
        """
        # coords are a list of (x, y) values
        if coords is None:
            coords = []
            n_coords = 0
        else:
            n_coords = len(coords)
            coords = [tuple(x) for x in coords]
        if len(coords) != n_coords:
            msg = "Points.__init__: Degenerate points found!"
            logging.warning(msg)
        self.coords = coords
        if self.coords:
            self.nd_array = homogenize(coords)  # homogeneous coordinates
        self.type = Types.POINTS
        self.subtype = Types.POINTS
        self.dist_tol = defaults["dist_tol"]
        self.dist_tol2 = self.dist_tol**2
        common_properties(self, False)

    def __str__(self):
        """Return a string representation of the points.

        Returns:
            str: The string representation of the points.
        """
        return f"Points({self.coords})"

    def __repr__(self):
        """Return a string representation of the points.

        Returns:
            str: The string representation of the points.
        """
        return f"Points({self.coords})"

    def __getitem__(self, subscript):
        """Get the point(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to get the point(s) from.

        Returns:
            Point or list[Point]: The point(s) at the given subscript.

        Raises:
            TypeError: If the subscript type is invalid.
        """
        if isinstance(subscript, slice):
            res = self.coords[subscript.start : subscript.stop : subscript.step]
        elif isinstance(subscript, int):
            res = self.coords[subscript]
        else:
            raise TypeError("Invalid subscript type")
        return res

    def _update_coords(self):
        """Update the homogeneous coordinates of the points."""
        self.nd_array = homogenize(self.coords)

    def __setitem__(self, subscript, value):
        """Set the point(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to set the point(s) at.
            value (Point or list[Point]): The value to set the point(s) to.

        Raises:
            TypeError: If the subscript type is invalid.
        """
        if isinstance(subscript, slice):
            self.coords[subscript.start : subscript.stop : subscript.step] = value
            self._update_coords()
        elif isinstance(subscript, int):
            self.coords[subscript] = value
            self._update_coords()
        else:
            raise TypeError("Invalid subscript type")

    def __eq__(self, other):
        """Check if the points are equal to another Points object.

        Args:
            other (Points): The other Points object to compare against.

        Returns:
            bool: True if the points are equal, False otherwise.
        """
        return (
            other.type == Types.POINTS
            and len(self.coords) == len(other.coords)
            and allclose(
                self.nd_array,
                other.nd_array,
                rtol=defaults["rtol"],
                atol=defaults["atol"],
            )
        )

    def append(self, item: Point) -> Self:
        """Append a point to the points.

        Args:
            item (Point): The point to append.

        Returns:
            Self: The updated Points object.
        """
        self.coords.append(item)
        self._update_coords()
        return self

    def extend(self, items: Sequence[Point]) -> Self:
        """Extend the points with a given sequence of points.

        Args:
            items (Sequence[Point]): The sequence of points to add.

        Returns:
            Self: The updated Points object.
        """
        self.coords.extend(items)
        self._update_coords()
        return self

    def pop(self, index: int = -1) -> Point:
        """Remove the point at the given index and return it.

        Args:
            index (int, optional): The index of the point to remove. Defaults to -1.

        Returns:
            Point: The removed point.
        """
        value = self.coords.pop(index)
        self._update_coords()
        return value

    def __delitem__(self, subscript) -> Self:
        """Delete the point(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to delete the point(s) from.

        Raises:
            TypeError: If the subscript type is invalid.
        """
        coords = self.coords
        if isinstance(subscript, slice):
            del coords[subscript.start : subscript.stop : subscript.step]
        elif isinstance(subscript, int):
            del coords[subscript]
        else:
            raise TypeError("Invalid subscript type")
        self._update_coords()

    def remove(self, value):
        """Remove the first occurrence of the given point.

        Args:
            value (Point): The point value to remove.
        """
        self.coords.remove(value)
        self._update_coords()

    def insert(self, index, points):
        """Insert a point at the specified index.

        Args:
            index (int): The index to insert the point at.
            points (Point): The point to insert.
        """
        self.coords.insert(index, points)
        self._update_coords()

    def clear(self):
        """Clear all points."""
        self.coords.clear()
        self.nd_array = ndarray((0, 3))

    def reverse(self):
        """Reverse the order of the points."""
        self.coords.reverse()
        self._update_coords()

    def __iter__(self):
        """Return an iterator over the points.

        Returns:
            Iterator[Point]: An iterator over the points.
        """
        return iter(self.coords)

    def __len__(self):
        """Return the number of points.

        Returns:
            int: The number of points.
        """
        return len(self.coords)

    def __bool__(self):
        """Return whether the Points object has any points.

        Returns:
            bool: True if there are points, False otherwise.
        """
        return bool(self.coords)

    @property
    def homogen_coords(self):
        """Return the homogeneous coordinates of the points.

        Returns:
            ndarray: The homogeneous coordinates.
        """
        return self.nd_array

    def copy(self):
        """Return a copy of the Points object.

        Returns:
            Points: A copy of the Points object.
        """
        points = Points(copy.copy(self.coords))
        points.nd_array = ndarray.copy(self.nd_array)
        return points

    @property
    def pairs(self):
        """Return a list of consecutive pairs of points.

        Returns:
            list[tuple[Point, Point]]: A list where each element is a tuple containing two consecutive points.
        """
        return list(zip(self.coords[:-1], self.coords[1:]))
