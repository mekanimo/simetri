"""Turtle graphics variant, with a twist."""

from math import pi, radians
from typing import Sequence
from dataclasses import dataclass

from ..graphics.batch import Batch
from ..graphics.shape import Shape

from ..geometry.geometry import line_by_point_angle_length as get_pos


@dataclass
class State:
    """A state of the turtle."""

    pos: tuple
    angle: float
    pen_is_down: bool

class Turtle(Batch):
    """A Turtle graphics variant, with a twist."""

    def __init__(self, *args, in_degrees: bool = False, **kwargs):

        self.pos = (0, 0)
        self.stack = []
        self.lists = []
        self.current_list = [self.pos]
        super().__init__([], *args, **kwargs)
        self.pen_is_down = True
        self._set_aliasess()
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
    def angle(self):
        """Return the current angle of the turtle.
        Clamp the angle to 0 <= angle < 360 if in degrees.
        Clamp the angle to 0 <= angle < 2*pi if in radians.
        """
        if self.in_degrees:
            res = self._angle % 360
        else:
            res = self._angle % (2 * pi)

        return res

    @angle.setter
    def angle(self, value):
        """Set the angle of the turtle."""
        self._angle = value

    def _forward_pos(self, dist: float=None):
        """Return the position after moving forward by the given distance."""
        if self.in_degrees:
            angle = radians(self._angle)
        else:
            angle = self._angle

        if dist is None:
            dist = self.def_dist

        return get_pos(self.pos, angle, dist)[1]

    def forward(self, dist: float=None):
        """Move the turtle forward by the given distance."""
        x, y = self._forward_pos(dist)[:2]
        self.pos = (x, y)
        if self.pen_is_down:
            self.current_list.append(self.pos)

    def go(self, dist: float=None):
        """Move the turtle forward.
        Don't draw regardless of the pen state.
        """
        x, y = self._forward_pos(dist)[:2]
        self.pos = (x, y)
        self.lists.append(self.current_list)
        self.current_list = [self.pos]

    def backward(self, dist: float=None):
        """Move the turtle backward by the given distance."""
        if dist is None:
            dist = self.def_dist
        self.forward(-dist)

    def left(self, angle: float=None):
        """Turn the turtle left by the given angle."""
        if angle is None:
            angle = self.def_angle
        self._angle += angle

    def right(self, angle: float=None):
        """Turn the turtle right by the given angle."""
        if angle is None:
            angle = self.def_angle
        self._angle -= angle

    def turn_around(self):
        """Turn the turtle around by 180 degrees."""
        if self.in_degrees:
            self._angle += 180
        else:
            self._angle += pi


    def pen_up(self):
        """Lift the pen."""
        self.pen_is_down = False
        self.lists.append(self.current_list)
        self.current_list = []

    def pen_down(self):
        """Lower the pen."""
        self.pen_is_down = True
        self.current_list.append(self.pos)

    def move_to(self, pos):
        """Move the turtle to the given position."""
        self.pos = pos
        if self.pen_is_down:
            self.current_list.append(self.pos)


    def push(self):
        """Save the current state of the turtle."""
        state = State(self.pos, self._angle, self.pen_is_down)
        self.stack.append(state)

    def pop(self):
        """Restore the last saved state of the turtle."""
        state = self.stack.pop()
        self.pos = state.pos
        self._angle = state.angle
        self.pen_is_down = state.pen_is_down
        self.lists.append(self.current_list)
        self.current_list = [self.pos]

    def reset(self):
        """Reset the turtle to its initial state."""
        self.append(self.current_shape)
        self.pos = (0, 0)
        if self.in_degrees:
            self._angle = 0
        else:
            self._angle = pi / 2
        self.current_list = [self.pos]
        self.pen_is_down = True

    # aliases
    def _set_aliasess(self):
        self.fd = self.forward
        self.bk = self.backward
        self.lt = self.left
        self.rt = self.right
        self.pu = self.pen_up
        self.pd = self.pen_down
        self.goto = self.move_to


def add_digits(n):
    """Return the sum of the digits of n.
    spirolateral helper function.
    10 -> 1 + 0 -> 1
    123 -> 1 + 2 + 3 -> 6
    """
    return sum((int(x) for x in str(n)))


def spirolateral(
    sequence: Sequence, angle: float, cycles: int = 15, multiplier: float = 50
):
    """Draw a spirolateral with the given sequence and angle.
    Angle is in degrees."""
    turtle = Turtle(in_degrees=True)
    count = 0
    while count < cycles:
        for i in sequence:
            turtle.forward(multiplier * add_digits(i))
            turtle.right(180 - angle)
            count += 1
    return turtle


def spiral(turtle, side, angle, delta, cycles=15):
    """Draw a spiral with the given side, angle, delta, and cycles."""
    t = turtle
    count = 0
    while count < cycles:
        t.forward(side)
        t.right(angle)
        side += delta
        count += 1
    return t
