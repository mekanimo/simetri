"""Point container used by Shape geometry.

``Points`` stores ``(x, y)`` vertices and lazily builds a homogeneous
``ndarray`` for affine transforms.

Examples:
    >>> import simetri.graphics as sg
    >>> pts = sg.Points([(0, 0), (1, 0), (1, 1)])
    >>> len(pts)
    3
    >>> pts.nd_array.shape
    (3, 3)
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from types import TracebackType
from typing import Self

from numpy import allclose, ndarray

from ..base.all_enums import Types
from ..base.common import PointType
from ..config.settings import runtime_defaults as defaults
from ..geom.homogenize import homogenize
from ..helpers.utilities import format_data, register_format_handler


class _GroupUpdateContext:
    """Context manager for batch operations on Points to avoid redundant cache invalidations."""

    def __init__(self, points_obj: Points) -> None:
        """Bind to a ``Points`` instance for deferred cache invalidation.

        Args:
            points_obj: The ``Points`` object whose cache invalidation is deferred.
        """
        self.points_obj = points_obj
        self.original_invalidate = None

    def __enter__(self) -> Self:
        # Replace the invalidate method with a no-op during group operations
        self.original_invalidate = self.points_obj._invalidate_cache
        self.points_obj._invalidate_cache = lambda: None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Restore original method and invalidate cache once
        self.points_obj._invalidate_cache = self.original_invalidate
        self.points_obj._invalidate_cache()


class Points:
    """Mutable sequence of 2D points with lazy homogeneous coordinates.

    Used by ``Shape`` as ``primary_points``.
    Affinely transformed coordinates are obtained via ``homogen_coords`` /
    ``nd_array``.

    Attributes:
        coords: List of ``(x, y)`` tuples.
        type: Always ``Types.POINTS``.
        nd_array_changed: Set when the cache should be refreshed by Shape.

    Examples:
        >>> pts = sg.Points([(0, 0), (10, 0)])
        >>> _ = pts.append((10, 10))
        >>> list(pts)
        [(0, 0), (10, 0), (10, 10)]
"""

    def __init__(self, coords: Sequence[PointType] | None = None) -> None:
        """Initialize a Points container.

        Args:
            coords: Optional sequence of points. Defaults to an empty list.
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
        self.nd_array_changed = False

    def __str__(self) -> str:
        """Return a string representation of the points.

        Returns:
            str: The string representation of the points.
        """
        return f"Points({self.coords})"

    @property
    def nd_array(self) -> ndarray:
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
    def nd_array(self, value: ndarray) -> None:
        """Set the homogeneous coordinates directly and mark as clean.

        Args:
            value: The homogeneous coordinate array to set.
        """
        self._nd_array_cache = value
        self._coords_dirty = False

    def _invalidate_cache(self) -> None:
        """Mark the homogeneous coordinates cache as dirty and notify shape if needed."""
        self._coords_dirty = True
        self.nd_array_changed = True

    def group_update(self) -> _GroupUpdateContext:
        """Context manager for group operations to avoid redundant cache invalidations.

        Usage:
            with points.group_update():
                points.append(point1)
                points.append(point2)
                # Cache invalidation happens only once when exiting the context
        """
        return _GroupUpdateContext(self)

    def __repr__(self) -> str:
        """Return a string representation of the points.

        Returns:
            str: The string representation of the points.
        """
        return f"Points({self.coords})"

    def __getitem__(
        self, subscript: int | slice
    ) -> PointType | list[PointType]:
        """Get the point(s) at the given subscript.

        Args:
            subscript: Index or slice into ``coords``.

        Returns:
            One point or a list of points at ``subscript``.

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

    def _update_coords(self) -> None:
        """Mark homogeneous coordinates as needing update (replaced with lazy evaluation)."""
        self._invalidate_cache()

    def __setitem__(
        self,
        subscript: int | slice,
        value: PointType | list[PointType],
    ) -> None:
        """Set the point(s) at the given subscript.

        Args:
            subscript: Index or slice into ``coords``.
            value: Point or list of points to assign.

        Raises:
            TypeError: If the subscript type is invalid.
        """
        if isinstance(subscript, slice):
            self.coords[subscript.start : subscript.stop : subscript.step] = (
                value
            )
            self._update_coords()
        elif isinstance(subscript, int):
            self.coords[subscript] = value
            self._update_coords()
        else:
            raise TypeError("Invalid subscript type")

    def __eq__(self, other: object) -> bool:
        """Check if the points are equal to another Points object.

        Args:
            other: Object to compare (must be ``Points`` with matching coords).

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

    def append(self, item: PointType) -> Self:
        """Append a point to the points.

        Args:
            item (PointType): The point to append.

        Returns:
            Self: The updated Points object.
        """
        self.coords.append(item)
        self._update_coords()
        return self

    def extend(self, items: Sequence[PointType]) -> Self:
        """Extend the points with a given sequence of points.

        Args:
            items (Sequence[PointType]): The sequence of points to add.

        Returns:
            Self: The updated Points object.
        """
        self.coords.extend(items)
        self._update_coords()
        return self

    def pop(self, index: int = -1) -> PointType:
        """Remove the point at the given index and return it.

        Args:
            index (int, optional): The index of the point to remove. Defaults to -1.

        Returns:
            PointType: The removed point.
        """
        value = self.coords.pop(index)
        self._update_coords()
        return value

    def __delitem__(self, subscript: int | slice) -> Self:
        """Delete the point(s) at the given subscript.

        Args:
            subscript: Index or slice into ``coords``.

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

    def remove(self, value: PointType) -> None:
        """Remove the first occurrence of the given point.

        Args:
            value (PointType): The point value to remove.
        """
        self.coords.remove(value)
        self._update_coords()

    def insert(self, index: int, points: PointType) -> None:
        """Insert a point at the specified index.

        Args:
            index: Index at which to insert.
            points: Point to insert.
        """
        self.coords.insert(index, points)
        self._update_coords()

    def clear(self) -> None:
        """Clear all points."""
        self.coords.clear()
        self._invalidate_cache()

    def reverse(self) -> None:
        """Reverse the order of the points."""
        self.coords.reverse()
        self._update_coords()

    def __iter__(self) -> Iterator[PointType]:
        """Return an iterator over the points.

        Returns:
            Iterator over ``coords``.
        """
        return iter(self.coords)

    def __len__(self) -> int:
        """Return the number of points.

        Returns:
            int: The number of points.
        """
        return len(self.coords)

    def __bool__(self) -> bool:
        """Return whether the Points object has any points.

        Returns:
            bool: True if there are points, False otherwise.
        """
        return bool(self.coords)

    @property
    def homogen_coords(self) -> ndarray:
        """Return the homogeneous coordinates of the points.

        Returns:
            ndarray: The homogeneous coordinates.
        """
        return self.nd_array

    def copy(self) -> Points:
        """Return a copy of the Points object.

        Returns:
            A shallow copy with duplicated homogeneous cache when valid.
        """
        points = Points(copy.copy(self.coords))
        # Copy the cached homogeneous coordinates if they exist
        if not self._coords_dirty and self._nd_array_cache is not None:
            points._nd_array_cache = ndarray.copy(self._nd_array_cache)
            points._coords_dirty = False
        return points

    @property
    def pairs(self) -> list[tuple[PointType, PointType]]:
        """Return consecutive vertex pairs along ``coords``.

        Returns:
            ``[(p0, p1), (p1, p2), ...]``.
        """
        return list(zip(self.coords[:-1], self.coords[1:]))


def _format_points_for_display(
    points: Points,
    *,
    n_digits: int | None = None,
    n_sig_digits: int | None = None,
) -> str:
    formatted_coords = format_data(
        points.coords,
        n_digits=n_digits,
        n_sig_digits=n_sig_digits,
    )
    return f"Points({formatted_coords})"


register_format_handler(Points, _format_points_for_display)
