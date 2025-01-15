from math import cos, sin
from typing import Tuple, Union

import numpy as np


Vec2 = Tuple[float, float]


class Vector2D:
    """A 2D vector class.
    a + b vector addition
    a - b vector subtraction
    a * b cross product if b is a vector
    a.dot(b) dot product
    s * a and a * s multiplication with scalar
    |a| norm of the vector
    a.rotate(angle) rotation
    """

    def __init__(self, x: float, y: float):
        self.vector = np.array([x, y])

    def __add__(self, other: Vec2) -> Vec2:
        return Vector2D(*(self.vector + other.vector))

    def __sub__(self, other: Vec2) -> Vec2:
        return Vector2D(*(self.vector - other.vector))

    def __mul__(self, other: Union[Vec2, float]) -> Union[float, Vec2]:
        if isinstance(other, Vector2D):
            res = np.cross(self.vector, other.vector)
        elif isinstance(other, (int, float)):
            res = Vector2D(*(other * self.vector))
        else:
            res = NotImplemented

        return res

    def __neg__(self) -> Vec2:
        return Vector2D(*(-self.vector))

    def __abs__(self) -> float:
        return np.linalg.norm(self.vector)

    def norm(self) -> float:
        """Return the norm of the vector."""
        return np.linalg.norm(self.vector)

    def dot(self, other: Vec2) -> float:
        """Return the dot product of self and other."""
        return np.dot(self.vector, other.vector)

    def cross(self, other: Vec2) -> float:
        """Return the cross product of self and other."""
        return np.cross(self.vector, other.vector)

    def rotate(self, angle: float) -> Vec2:
        """Rotate self counterclockwise by angle (in degrees)"""
        angle_rad = np.radians(angle)
        rotation_matrix = np.array(
            [[cos(angle_rad), -sin(angle_rad)], [sin(angle_rad), cos(angle_rad)]]
        )
        rotated_vector = rotation_matrix @ self.vector
        return Vector2D(*rotated_vector)

    def __repr__(self) -> str:
        return f"({self.vector[0]:.2f}, {self.vector[1]:.2f})"
