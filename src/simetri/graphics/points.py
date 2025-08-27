"""Shape object uses the Points class to store the coordinates of the points that make up the shape.
The Points class is a container for coordinates of multiple points.
It provides conversion to homogeneous coordinates in nd_arrays.
Shape.final_coords is computed by using the Points.homogen_coords property."""

import copy
from typing import Sequence

from numpy import allclose, ndarray
from typing_extensions import Self

from ..geometry.geometry import homogenize
from .common import Point, common_properties
from .all_enums import Types
from ..settings.settings import defaults


class _BatchUpdateContext:
    """Context manager for batch operations on Points to avoid redundant cache invalidations."""

    def __init__(self, points_obj):
        self.points_obj = points_obj
        self.original_invalidate = None

    def __enter__(self):
        # Replace the invalidate method with a no-op during batch operations
        self.original_invalidate = self.points_obj._invalidate_cache
        self.points_obj._invalidate_cache = lambda: None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original method and invalidate cache once
        self.points_obj._invalidate_cache = self.original_invalidate
        self.points_obj._invalidate_cache()


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
        else:
            coords = [tuple(x) for x in coords]
        self.coords = coords

        # Initialize cache variables for lazy evaluation of homogeneous coordinates
        self._nd_array_cache = None
        self._coords_dirty = True

        self.type = Types.POINTS
        self.subtype = Types.POINTS
        self.dist_tol = defaults["dist_tol"]
        self.dist_tol2 = self.dist_tol**2
        common_properties(self, False)
        self.nd_array_changed = False

    def __str__(self):
        """Return a string representation of the points.

        Returns:
            str: The string representation of the points.
        """
        return f"Points({self.coords})"

    @property
    def nd_array(self):
        """Get the homogeneous coordinates of the points (computed lazily).

        Returns:
            ndarray: The homogeneous coordinates.
        """
        if self._coords_dirty or self._nd_array_cache is None:
            if self.coords:
                self._nd_array_cache = homogenize(self.coords)
            else:
                self._nd_array_cache = ndarray((0, 3))
            self._coords_dirty = False
        return self._nd_array_cache

    @nd_array.setter
    def nd_array(self, value):
        """Set the homogeneous coordinates directly and mark as clean.

        Args:
            value: The homogeneous coordinate array to set.
        """
        self._nd_array_cache = value
        self._coords_dirty = False

    def _invalidate_cache(self):
        """Mark the homogeneous coordinates cache as dirty and notify shape if needed."""
        self._coords_dirty = True
        self.nd_array_changed = True

    def batch_update(self):
        """Context manager for batch operations to avoid redundant cache invalidations.

        Usage:
            with points.batch_update():
                points.append(point1)
                points.append(point2)
                # Cache invalidation happens only once when exiting the context
        """
        return _BatchUpdateContext(self)

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
        """Mark homogeneous coordinates as needing update (replaced with lazy evaluation)."""
        self._invalidate_cache()

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
                rtol=defaults["rel_tol"],
                atol=defaults["abs_tol"],
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
        self._invalidate_cache()

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
        # Copy the cached homogeneous coordinates if they exist
        if not self._coords_dirty and self._nd_array_cache is not None:
            points._nd_array_cache = ndarray.copy(self._nd_array_cache)
            points._coords_dirty = False
        return points

    @property
    def pairs(self):
        """Return a list of consecutive pairs of points.

        Returns:
            list[tuple[Point, Point]]: A list where each element is a tuple containing two consecutive points.
        """
        return list(zip(self.coords[:-1], self.coords[1:]))
