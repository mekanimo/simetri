"""Dot and Dots classes for creating circular markers.

Examples:
    >>> import simetri.graphics as sg
    >>> d = sg.Dot((10, 20), radius=3)
    >>> d.pos
    (10, 20)
    >>> cluster = sg.Dots((0, 0), radius=2)
"""

__all__ = ["Dot", "Dots"]

import numpy as np

from ..canvas.style_map import shape_args
from ..colors.colors import Color
from ..geometry.points.point_utils import close_points2
from ..helpers.validation import validate_args
from ..settings.settings import defaults
from ..core.all_enums import Types
from ..group.batch import Group
from ..core.common import PointType
from .shapes import Shape


class Dot(Shape):
    """A filled circular marker at a single point.

    Geometry is a one-point shape; ``radius`` is used only for drawing.
    The primary style property is ``color``.

    Attributes:
        pos: Position of the dot (same as ``vertices[0]``).
        radius: Draw radius.
        color: Fill/stroke color for the marker.
        subtype: Always ``Types.DOT``.

    Examples:
        >>> import simetri.graphics as sg
        >>> dot = sg.Dot((5, 5), radius=2)
        >>> dot.subtype.name
        'DOT'
    """

    def __init__(
        self,
        pos: PointType = (0, 0),
        radius: float = 1,
        color: Color = None,
        **kwargs,
    ) -> None:
        """Initialize a Dot.

        Args:
            pos: Position of the dot. Defaults to ``(0, 0)``.
            radius: Draw radius. Defaults to 1.
            color: Marker color. Defaults to ``defaults["dot_color"]``.
            **kwargs: Additional shape style keyword arguments.
        """
        valid_args = shape_args
        validate_args(kwargs, valid_args)
        super().__init__([(0, 0)], **kwargs)
        self.move_to(pos)
        self.subtype = Types.DOT
        self.radius = radius  # for drawing
        if color is not None:
            self.color = color
        else:
            self.color = defaults["dot_color"]

    @property
    def pos(self) -> PointType:
        """Return the position of the dot.

        Returns:
            PointType: The single vertex of the dot.
        """
        return self.vertices[0]

    @pos.setter
    def pos(self, new_pos: PointType):
        """Set the position of the dot.

        Args:
            new_pos: New ``(x, y)`` position.

        Raises:
            TypeError: If ``new_pos`` is not a list, tuple, or ndarray.
        """
        if not isinstance(new_pos, (list, tuple, np.ndarray)):
            raise TypeError("Name must be a string")
        self.move_to(new_pos)

    def copy(self, **kwargs) -> Shape:
        """Return a deep-enough copy of the dot.

        Args:
            **kwargs: Attributes to override on the copy.

        Returns:
            Shape: A new ``Dot`` at the same position with copied color.
        """
        color = self.color.copy()
        dot = Dot(self.pos, self.radius, color)

        for k, v in kwargs.items():
            setattr(dot, k, v)

        return dot

    def __str__(self):
        """Return a human-readable representation."""
        return f"Dot({self.pos}, {self.radius}, {self.color})"

    def __repr__(self):
        """Return a developer-oriented representation."""
        return f"Dot({self.pos}, {self.radius}, {self.color})"

    def __eq__(self, other):
        """Return True if ``other`` is a Dot at nearly the same position.

        Args:
            other: Object to compare.

        Returns:
            bool: True when types match and positions are within tolerance.
        """
        return other.type == Types.DOT and close_points2(
            self.pos, other.pos, self.dtol2
        )


class Dots(Group):
    """Group that starts with a single Dot and can hold more.

    Attributes:
        elements: List of contained ``Dot`` (and nested) elements.
        subtype: Always ``Types.DOTS``.

    Examples:
        >>> import simetri.graphics as sg
        >>> dots = sg.Dots((0, 0), radius=1)
        >>> len(dots)
        1
    """

    def __init__(
        self,
        pos: PointType = (0, 0),
        radius: float = 1,
        color: Color = None,
        **kwargs,
    ) -> None:
        """Initialize a Dots group with one Dot.

        Args:
            pos: Position of the initial dot. Defaults to ``(0, 0)``.
            radius: Draw radius for the initial dot. Defaults to 1.
            color: Color for the initial dot. Defaults to None (library default).
            **kwargs: Additional keyword arguments passed to the Dot.
        """
        dot = Dot(pos=pos, radius=radius, color=color, **kwargs)
        super().__init__([dot], subtype=Types.DOTS, **kwargs)
