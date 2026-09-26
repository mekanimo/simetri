"""Line-segment container used by Shape geometry.

``Lines`` stores segments as parallel start/end ``Points`` sequences.

Examples:
    >>> from simetri.shapes.lines import Lines
    >>> lines = Lines([((0, 0), (1, 0)), ((1, 0), (1, 1))])
    >>> len(lines)
    2
"""

from collections.abc import Iterator, Sequence
from typing import Self

import numpy as np

from ..base.all_enums import Types
from ..base.common import PointType
from ..geom.homogenize import homogenize
from .points import Points


class Lines:
    """Mutable container of line segments as start/end point pairs.

    Attributes:
        starts / ends: Parallel ``Points`` for segment endpoints.
        type: Always ``Types.LINES`` when set by callers.

    Examples:
        >>> from simetri.shapes.lines import Lines
        >>> lines = Lines([((0, 0), (1, 0)), ((1, 0), (1, 1))])
        >>> len(lines)
        2
"""

    def __init__(
        self,
        point_pairs: Sequence[tuple[PointType, PointType]] | None = None,
        points: Points | Sequence[PointType] | None = None,
        start_points: Points | Sequence[PointType] | None = None,
        end_points: Points | Sequence[PointType] | None = None,
    ) -> Self:
        """Initialize lines from pairs, interleaved points, or start/end sequences.

        Provide exactly one of ``point_pairs``, ``points``, or both
        ``start_points`` and ``end_points``.

        Args:
            point_pairs: Sequence of ``(start, end)`` point pairs.
            points: Interleaved endpoints ``[s0, e0, s1, e1, ...]``.
            start_points: Segment start points (with ``end_points``).
            end_points: Segment end points (with ``start_points``).

        Raises:
            ValueError: If start and end sequences differ in length.
        """
        if point_pairs:
            self.start_points = Points([p[0] for p in point_pairs])
            self.end_points = Points([p[1] for p in point_pairs])
        elif points:
            self.start_points = Points(points[::2])
            self.end_points = Points(points[1::2])
        else:
            if isinstance(start_points, Points):
                self.start_points = start_points
            else:
                self.start_points = Points(start_points)

            if isinstance(end_points, Points):
                self.end_points = end_points
            else:
                self.end_points = Points(end_points)

        if len(self.start_points) != len(self.end_points):
            raise ValueError(
                "start_points and end_points must have the same length"
            )

        self.type = Types.LINE
        self.subtype = Types.LINE

    def __str__(self) -> str:
        """Return a string representation of the lines."""
        return f"Lines({self.point_pairs})"

    def __repr__(self) -> str:
        """Return a string representation of the lines."""
        return f"Lines({self.point_pairs})"

    def __getitem__(
        self, subscript: int | slice
    ) -> tuple[PointType, PointType] | list[tuple[PointType, PointType]]:
        """Return one segment or a slice of segments by index."""
        if isinstance(subscript, slice):
            return list(
                zip(
                    self.start_points[
                        subscript.start : subscript.stop : subscript.step
                    ],
                    self.end_points[
                        subscript.start : subscript.stop : subscript.step
                    ],
                )
            )
        if isinstance(subscript, int):
            return self.start_points[subscript], self.end_points[subscript]
        raise TypeError("Invalid subscript type")

    def __setitem__(
        self,
        subscript: int | slice,
        value: tuple[PointType, PointType] | Sequence[tuple[PointType, PointType]],
    ) -> None:
        """Assign one segment or a slice of segments."""
        if isinstance(subscript, slice):
            self.start_points[subscript] = [point[0] for point in value]
            self.end_points[subscript] = [point[1] for point in value]
            return
        if isinstance(subscript, int):
            self.start_points[subscript] = value[0]
            self.end_points[subscript] = value[1]
            return
        raise TypeError("Invalid subscript type")

    def __eq__(self, other: object) -> bool:
        """Return whether ``other`` is a ``Lines`` with the same endpoints."""
        return (
            isinstance(other, Lines)
            and self.start_points == other.start_points
            and self.end_points == other.end_points
        )

    def append(self, item: tuple[PointType, PointType]) -> Self:
        """Append a line segment."""
        self.start_points.append(item[0])
        self.end_points.append(item[1])
        return self

    def extend(self, items: Sequence[tuple[PointType, PointType]]) -> Self:
        """Extend the lines with the given line segments."""
        self.start_points.extend([point[0] for point in items])
        self.end_points.extend([point[1] for point in items])
        return self

    def pop(self, index: int = -1) -> tuple[PointType, PointType]:
        """Remove the line at the given index and return it."""
        start_point = self.start_points.pop(index)
        end_point = self.end_points.pop(index)
        return start_point, end_point

    def __delitem__(self, subscript: int | slice) -> Self:
        """Delete segment(s) at ``subscript``."""
        del self.start_points[subscript]
        del self.end_points[subscript]

    def remove(self, value: tuple[PointType, PointType]) -> None:
        """Remove the first segment equal to ``value``."""
        index = self.point_pairs.index(value)
        del self.start_points[index]
        del self.end_points[index]

    def insert(self, index: int, line: tuple[PointType, PointType]) -> None:
        """Insert a segment at ``index``."""
        self.start_points.insert(index, line[0])
        self.end_points.insert(index, line[1])

    def clear(self) -> None:
        """Clear all lines."""
        self.start_points.clear()
        self.end_points.clear()

    def reverse(self) -> None:
        """Reverse the order of the lines."""
        self.start_points.reverse()
        self.end_points.reverse()

    def __iter__(self) -> Iterator[tuple[PointType, PointType]]:
        """Iterate over ``(start, end)`` segment pairs."""
        return iter(self.point_pairs)

    def __len__(self) -> int:
        """Return the number of lines."""
        return len(self.start_points)

    def __bool__(self) -> bool:
        """Return whether the Lines object has any lines."""
        return bool(self.start_points)

    def copy(self) -> Self:
        """Return a shallow copy with duplicated ``Points`` containers."""
        return Lines(
            start_points=self.start_points.copy(),
            end_points=self.end_points.copy(),
        )

    @property
    def point_pairs(self) -> list[tuple[PointType, PointType]]:
        """Return segments as ``(start, end)`` pairs."""
        return list(zip(self.start_points, self.end_points))

    @property
    def points(self) -> list[PointType]:
        """Return endpoints interleaved ``[s0, e0, s1, e1, ...]``."""
        points = []
        for start_point, end_point in self.point_pairs:
            points.extend([start_point, end_point])
        return points

    @property
    def homogen_coords(self) -> np.ndarray:
        """Return homogeneous coordinates for flattened ``points``."""
        return homogenize(self.points)

    def homogenize(self) -> tuple[np.ndarray, np.ndarray]:
        """Return homogeneous coordinates for start and end ``Points``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Start and end homogeneous arrays.
        """
        return (
            self.start_points.homogen_coords,
            self.end_points.homogen_coords,
        )
