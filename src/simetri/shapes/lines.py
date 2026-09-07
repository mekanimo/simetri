"""Line-segment container used by Shape geometry.

``Lines`` stores segments as parallel start/end ``Points`` sequences.

Examples:
    >>> from simetri.shapes.lines import Lines
    >>> lines = Lines([((0, 0), (1, 0)), ((1, 0), (1, 1))])
    >>> len(lines)
    2
"""

from collections.abc import Sequence
from typing import Self

from ..geom.homogenize import homogenize
from ..base.all_enums import Types
from ..base.common import PointType
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
        point_pairs: Sequence[tuple[PointType, PointType]] = None,
        points: Points | Sequence[PointType] = None,
        start_points: Points | Sequence[PointType] = None,
        end_points: Points | Sequence[PointType] = None,
    ) -> Self:
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

    def __str__(self):
        """Return a string representation of the lines."""
        return f"Lines({self.point_pairs})"

    def __repr__(self):
        """Return a string representation of the lines."""
        return f"Lines({self.point_pairs})"

    def __getitem__(self, subscript):
        """Get the line(s) at the given subscript."""
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

    def __setitem__(self, subscript, value):
        """Set the line(s) at the given subscript."""
        if isinstance(subscript, slice):
            self.start_points[subscript] = [point[0] for point in value]
            self.end_points[subscript] = [point[1] for point in value]
            return
        if isinstance(subscript, int):
            self.start_points[subscript] = value[0]
            self.end_points[subscript] = value[1]
            return
        raise TypeError("Invalid subscript type")

    def __eq__(self, other):
        """Check if the lines are equal to another Lines object."""
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

    def __delitem__(self, subscript) -> Self:
        """Delete the line(s) at the given subscript."""
        del self.start_points[subscript]
        del self.end_points[subscript]

    def remove(self, value):
        """Remove the first occurrence of the given line."""
        index = self.point_pairs.index(value)
        del self.start_points[index]
        del self.end_points[index]

    def insert(self, index, line):
        """Insert a line at the specified index."""
        self.start_points.insert(index, line[0])
        self.end_points.insert(index, line[1])

    def clear(self):
        """Clear all lines."""
        self.start_points.clear()
        self.end_points.clear()

    def reverse(self):
        """Reverse the order of the lines."""
        self.start_points.reverse()
        self.end_points.reverse()

    def __iter__(self):
        """Return an iterator over the lines."""
        return iter(self.point_pairs)

    def __len__(self):
        """Return the number of lines."""
        return len(self.start_points)

    def __bool__(self):
        """Return whether the Lines object has any lines."""
        return bool(self.start_points)

    def copy(self):
        """Return a copy of the Lines object."""
        return Lines(
            start_points=self.start_points.copy(),
            end_points=self.end_points.copy(),
        )

    @property
    def point_pairs(self):
        """Return line segments as point pairs."""
        return list(zip(self.start_points, self.end_points))

    @property
    def points(self):
        """Return the flattened line endpoints."""
        points = []
        for start_point, end_point in self.point_pairs:
            points.extend([start_point, end_point])
        return points

    @property
    def homogen_coords(self):
        """Return flattened homogeneous coordinates for all line endpoints."""
        return homogenize(self.points)

    def homogenize(self):
        return (
            self.start_points.homogen_coords,
            self.end_points.homogen_coords,
        )
