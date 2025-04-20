"""Dot and Dots classes for creating dots.
"""

__all__ = ["Dot", "Dots"]

import numpy as np

from .shapes import Shape
from .batch import Batch
from ..helpers.validation import validate_args
from ..canvas.style_map import shape_args
from ..settings.settings import defaults
from .common import Point
from .all_enums import Types
from ..colors.colors import Color
from ..geometry.geometry import close_points2
from ..canvas.style_map import batch_args


class Dot(Shape):
    """A dot defined by a single point.
    The radius is for drawing. The only style property is the color."""

    def __init__(
        self, pos: Point = (0, 0), radius: float = 1, color: Color = None, **kwargs
    ) -> None:
        """Initialize a Dot object.

        Args:
            pos (Point, optional): The position of the dot. Defaults to (0, 0).
            radius (float, optional): The radius of the dot. Defaults to 1.
            color (Color, optional): The color of the dot. Defaults to None.
            **kwargs: Additional keyword arguments.
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
    def pos(self) -> Point:
        """Return the point of the dot.

        Returns:
            Point: The point of the dot.
        """
        return self.vertices[0]

    @pos.setter
    def pos(self, new_pos: Point):
        """Set the position of the dot.

        Args:
            new_pos (Point): The new position of the dot.

        Raises:
            TypeError: If the new position is not a list, tuple, or ndarray.
        """
        if not isinstance(new_pos, (list, tuple, np.ndarray)):
            raise TypeError("Name must be a string")
        self.move_to(new_pos)

    def copy(self) -> Shape:
        """Return a copy of the dot.

        Returns:
            Shape: A copy of the dot.
        """
        color = self.color.copy()
        return Dot(self.pos, self.radius, color)

    def __str__(self):
        """Return a string representation of the dot.

        Returns:
            str: The string representation of the dot.
        """
        return f"Dot({self.pos}, {self.radius}, {self.color})"

    def __repr__(self):
        """Return a string representation of the dot.

        Returns:
            str: The string representation of the dot.
        """
        return f"Dot({self.pos}, {self.radius}, {self.color})"

    def __eq__(self, other):
        """Check if the dot is equal to another dot.

        Args:
            other (Dot): The other dot to compare to.

        Returns:
            bool: True if the dots are equal, False otherwise.
        """
        return other.type == Types.DOT and close_points2(
            self.pos, other.pos, self.dtol2
        )


class Dots(Batch):
    """For creating multiple dots. Initially there is only one dot."""

    def __init__(
        self, pos: Point = (0, 0), radius: float = 1, color: Color = None, **kwargs
    ) -> None:
        """Initialize a Dots object.

        Args:
            pos (Point, optional): The position of the dots. Defaults to (0, 0).
            radius (float, optional): The radius of the dots. Defaults to 1.
            color (Color, optional): The color of the dots. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        valid_args = batch_args + shape_args + ["radius", "color", "pos"]
        dot = Dot(pos=pos, radius=radius, color=color, **kwargs)
        super().__init__([dot], subtype=Types.DOTS, **kwargs)
