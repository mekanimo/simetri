"""Turtle graphics variant built on Simetri ``Group`` objects.

Provides a familiar pen-based drawing API (forward, turn, pen up/down) that
records polylines as shapes. Also includes helpers such as ``spirolateral``
and ``spiral``.
"""

from math import pi, radians
from typing import Sequence, Any
from dataclasses import dataclass

from ..graphics.batch import Group

from ..geometry.geometry import line_by_point_angle_length as get_pos


@dataclass
class State:
    """A snapshot of turtle position, heading, and pen state.

    Attributes:
        pos: Turtle position as ``(x, y)``.
        angle: Heading angle in degrees or radians.
        pen_is_down: Whether the pen is currently drawing.
    """

    pos: tuple
    angle: float
    pen_is_down: bool


class Turtle(Group):
    """Turtle graphics that records drawn paths as Simetri shapes.

    Args:
        *args: Positional arguments forwarded to ``Group``.
        in_degrees: If True, angles are in degrees; otherwise radians.
        **kwargs: Keyword arguments forwarded to ``Group``.

    Examples:
        >>> import simetri.graphics as sg
        >>> t = sg.Turtle(in_degrees=True)
        >>> t.forward(40)
        >>> t.right(90)
        >>> t.forward(40)
    """

    def __init__(self, *args: Any, in_degrees: bool = False, **kwargs: Any) -> None:
        self.pos = (0, 0)
        self.stack = []
        self.lists = []
        self.current_list = [self.pos]
        super().__init__([], *args, **kwargs)
        self.pen_is_down = True
        self._set_aliases()
        self.in_degrees = in_degrees
        if in_degrees:
            self.def_angle = 90
        else:
            self.def_angle = pi / 2
        self.def_dist = 20
        # start facing north
        if in_degrees:
            self._angle = 90
        else:
            self._angle = pi / 2

    @property
    def angle(self) -> float:
        """Return the current angle of the turtle.

        The angle is clamped to the appropriate range based on the angle mode.

        Returns:
            float: The current angle, normalized to 0-360 degrees or 0-2π radians.
        """
        if self.in_degrees:
            res = self._angle % 360
        else:
            res = self._angle % (2 * pi)

        return res

    @angle.setter
    def angle(self, value: float) -> None:
        """Set the angle of the turtle.

        Args:
            value (float): The new angle value in degrees or radians
                based on the turtle's configuration.
        """
        self._angle = value

    def _forward_pos(self, dist: float = None) -> tuple:
        """Calculate the position after moving forward by the given distance.

        Args:
            dist (float, optional): The distance to move. Defaults to self.def_dist.

        Returns:
            tuple: The new position coordinates.
        """
        if self.in_degrees:
            angle = radians(self._angle)
        else:
            angle = self._angle

        if dist is None:
            dist = self.def_dist

        return get_pos(self.pos, angle, dist)[1]

    def forward(self, dist: float = None) -> None:
        """Move the turtle forward by the given distance.

        Moves the turtle and draws a line if the pen is down.

        Args:
            dist (float, optional): The distance to move forward. Defaults to self.def_dist.
        """
        x, y = self._forward_pos(dist)[:2]
        self.pos = (x, y)
        if self.pen_is_down:
            self.current_list.append(self.pos)

    def go(self, dist: float = None) -> None:
        """Move the turtle forward without drawing.

        Moves the turtle regardless of the pen state, and doesn't draw.
        Saves the current path and starts a new one.

        Args:
            dist (float, optional): The distance to move forward. Defaults to self.def_dist.
        """
        x, y = self._forward_pos(dist)[:2]
        self.pos = (x, y)
        self.lists.append(self.current_list)
        self.current_list = [self.pos]

    def backward(self, dist: float = None) -> None:
        """Move the turtle backward by the given distance.

        Args:
            dist (float, optional): The distance to move backward. Defaults to self.def_dist.
        """
        if dist is None:
            dist = self.def_dist
        self.forward(-dist)

    def left(self, angle: float = None) -> None:
        """Turn the turtle left by the given angle.

        Args:
            angle (float, optional): The angle to turn left. Defaults to self.def_angle.
        """
        if angle is None:
            angle = self.def_angle
        self._angle += angle

    def right(self, angle: float = None) -> None:
        """Turn the turtle right by the given angle.

        Args:
            angle (float, optional): The angle to turn right. Defaults to self.def_angle.
        """
        if angle is None:
            angle = self.def_angle
        self._angle -= angle

    def turn_around(self) -> None:
        """Turn the turtle around by 180 degrees.

        Rotates the turtle 180 degrees from its current direction.
        """
        if self.in_degrees:
            self._angle += 180
        else:
            self._angle += pi

    def pen_up(self) -> None:
        """Lift the pen.

        Stops drawing and saves the current path when the turtle moves.
        """
        self.pen_is_down = False
        self.lists.append(self.current_list)
        self.current_list = []

    def pen_down(self) -> None:
        """Lower the pen.

        Enables drawing when the turtle moves and adds the current position
        to the current path.
        """
        self.pen_is_down = True
        self.current_list.append(self.pos)

    def move_to(self, pos: tuple) -> None:
        """Move the turtle to the given position.

        Args:
            pos (tuple): The target position as (x, y) coordinates.
        """
        self.pos = pos
        if self.pen_is_down:
            self.current_list.append(self.pos)

    def push(self) -> None:
        """Save the current state of the turtle.

        Stores the current position, angle, and pen state for later retrieval.
        """
        state = State(self.pos, self._angle, self.pen_is_down)
        self.stack.append(state)

    def pop(self) -> None:
        """Restore the last saved state of the turtle.

        Retrieves the most recently saved state and restores the turtle to it.
        """
        state = self.stack.pop()
        self.pos = state.pos
        self._angle = state.angle
        self.pen_is_down = state.pen_is_down
        self.lists.append(self.current_list)
        self.current_list = [self.pos]

    def reset(self) -> None:
        """Reset the turtle to its initial state.

        Appends the current shape to the batch and resets position, angle,
        and pen state.
        """
        self.append(self.current_shape)
        self.pos = (0, 0)
        if self.in_degrees:
            self._angle = 0
        else:
            self._angle = pi / 2
        self.current_list = [self.pos]
        self.pen_is_down = True

    # aliases
    def _set_aliases(self) -> None:
        """Set up aliases for turtle methods.

        Creates shorthand method names commonly used in turtle graphics.
        """
        self.fd = self.forward
        self.bk = self.backward
        self.lt = self.left
        self.rt = self.right
        self.pu = self.pen_up
        self.pd = self.pen_down
        self.goto = self.move_to


def add_digits(n: int) -> int:
    """Return the sum of the digits of n.

    Spirolateral helper function that adds all digits in a number.

    Args:
        n (int): The number to process.

    Returns:
        int: The sum of all digits in n.

    Examples:
        >>> add_digits(10)
        1
        >>> add_digits(123)
        6
    """
    return sum((int(x) for x in str(n)))


def spirolateral(
    sequence: Sequence, angle: float, cycles: int = 15, multiplier: float = 50
) -> Turtle:
    """Draw a spirolateral with the given sequence and turn angle.

    Args:
        sequence: Sequence of numbers determining segment lengths.
        angle: Exterior turn angle in degrees.
        cycles: Number of forward/turn steps to draw.
        multiplier: Scaling factor for segment lengths.

    Returns:
        The ``Turtle`` used for drawing.

    Examples:
        >>> import simetri.graphics as sg
        >>> t = sg.spirolateral([1, 2, 3], angle=90, cycles=12)
    """
    turtle = Turtle(in_degrees=True)
    count = 0
    while count < cycles:
        for i in sequence:
            turtle.forward(multiplier * add_digits(i))
            turtle.right(180 - angle)
            count += 1
    return turtle


def spiral(
    turtle: Turtle, side: float, angle: float, delta: float, cycles: int = 15
) -> Turtle:
    """Draw a spiral by repeatedly moving forward and turning.

    Args:
        turtle: Turtle used for drawing.
        side: Initial segment length.
        angle: Turn angle after each segment.
        delta: Amount added to ``side`` after each step.
        cycles: Number of segments to draw.

    Returns:
        The turtle used for drawing.

    Examples:
        >>> import simetri.graphics as sg
        >>> t = sg.Turtle(in_degrees=True)
        >>> sg.spiral(t, side=10, angle=20, delta=2, cycles=30)
    """
    t = turtle
    count = 0
    while count < cycles:
        t.forward(side)
        t.right(angle)
        side += delta
        count += 1
    return t
