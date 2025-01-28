"""Shape object uses the Points class to store the coordinates of the points that make up the shape.
The Points class is a container for coordinates of multiple points.
It provides conversion to homogeneous coordinates in nd_arrays.
Shape.final_coords is computed by using the Points.homogen_coords property."""

import logging
import copy
from typing import Sequence

from numpy import allclose, ndarray
from typing_extensions import Self


from ..helpers.geometry import homogenize
from .common import Point, common_properties
from .all_enums import *
from ..settings.settings import defaults


class Points:
    """Container for coordinates of multiple points. They provide
    conversion to homogeneous coordinates in nd_arrays.
    Used in Shape objects."""

    def __init__(self, coords: Sequence[Point] = None) -> None:
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
            self.nd_array = homogenize(coords)
        self.type = Types.POINTS
        self.subtype = Types.POINTS
        self._coords = copy.deepcopy(coords)
        self.dist_tol = defaults["dist_tol"]
        self.dist_tol2 = self.dist_tol**2
        common_properties(self, False)

    def __str__(self):
        return f"Points({self.coords})"

    def __repr__(self):
        return f"Points({self.coords})"

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            res = self.coords[subscript.start : subscript.stop : subscript.step]
        elif isinstance(subscript, int):
            res = self.coords[subscript]
        else:
            raise TypeError("Invalid subscript type")

        return res

    def _update_coords(self):
        self.nd_array = homogenize(self.coords)
        self._coords = copy.deepcopy(self.coords)

    def __setitem__(self, subscript, value):
        if isinstance(subscript, slice):
            self.coords[subscript.start : subscript.stop : subscript.step] = value
            self._update_coords()
        elif isinstance(subscript, int):
            self.coords[subscript] = value
            self._update_coords()
        else:
            raise TypeError("Invalid subscript type")

    def __eq__(self, other):
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
        """Appends a point to the points."""
        self.coords.append(item)
        self._update_coords()

    def pop(self, index: int = -1) -> Point:
        """Removes the point at the given index and returns it."""
        value = self.coords.pop(index)
        self._update_coords()
        return value

    def __delitem__(self, subscript) -> Self:
        coords = self.coords
        if isinstance(subscript, slice):
            del coords[subscript.start : subscript.stop : subscript.step]
        elif isinstance(subscript, int):
            del coords[subscript]
        else:
            raise TypeError("Invalid subscript type")
        self._update_coords()

    def remove(self, value):
        """Removes the first occurrence of the value from the points."""
        self.coords.remove(value)
        self._update_coords()

    def insert(self, index, points):
        """Inserts the points at the given index."""
        self.coords.insert(index, points)
        self._update_coords()

    def reverse(self):
        """Reverses the order of the points."""
        self.coords.reverse()
        self._update_coords()

    def __iter__(self):
        return iter(self.coords)

    def __len__(self):
        return len(self.coords)

    def __bool__(self):
        return bool(self.coords)

    @property
    def homogen_coords(self):
        """Returns the homogeneous coordinates of the points."""
        return self.nd_array

    def copy(self):
        """Returns a copy of the points."""
        points = Points(copy.deepcopy(self.coords))
        points.nd_array = ndarray.copy(self.nd_array)
        points._coords = copy.deepcopy(self._coords)
        return points

    @property
    def pairs(self):
        """Returns a list of pairs of points."""
        return list(zip(self.coords[:-1], self.coords[1:]))
