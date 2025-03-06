from math import cos, sin
from typing import Tuple, Union

import numpy as np


Vec2 = Tuple[float, float]


class Vector2D:
    """A 2D vector class.

    Attributes:
        vector (np.ndarray): The vector represented as a numpy array.
    """

    def __init__(self, x: float, y: float):
        """Initializes a 2D vector with x and y coordinates.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
        """
        self.vector = np.array([x, y])

    def __add__(self, other: Vec2) -> Vec2:
        """Adds two vectors.

        Args:
            other (Vec2): The vector to add.

        Returns:
            Vec2: The resulting vector.
        """
        return Vector2D(*(self.vector + other.vector))

    def __sub__(self, other: Vec2) -> Vec2:
        """Subtracts two vectors.

        Args:
            other (Vec2): The vector to subtract.

        Returns:
            Vec2: The resulting vector.
        """
        return Vector2D(*(self.vector - other.vector))

    def __mul__(self, other: Union[Vec2, float]) -> Union[float, Vec2]:
        """Multiplies the vector with another vector or a scalar.

        Args:
            other (Union[Vec2, float]): The vector or scalar to multiply with.

        Returns:
            Union[float, Vec2]: The resulting vector or scalar.
        """
        if isinstance(other, Vector2D):
            res = np.cross(self.vector, other.vector)
        elif isinstance(other, (int, float)):
            res = Vector2D(*(other * self.vector))
        else:
            res = NotImplemented

        return res

    def __neg__(self) -> Vec2:
        """Negates the vector.

        Returns:
            Vec2: The negated vector.
        """
        return Vector2D(*(-self.vector))

    def __abs__(self) -> float:
        """Returns the norm of the vector.

        Returns:
            float: The norm of the vector.
        """
        return np.linalg.norm(self.vector)

    def norm(self) -> float:
        """Returns the norm of the vector.

        Returns:
            float: The norm of the vector.
        """
        return np.linalg.norm(self.vector)

    def dot(self, other: Vec2) -> float:
        """Returns the dot product of self and other.

        Args:
            other (Vec2): The vector to dot with.

        Returns:
            float: The dot product.
        """
        return np.dot(self.vector, other.vector)

    def cross(self, other: Vec2) -> float:
        """Returns the cross product of self and other.

        Args:
            other (Vec2): The vector to cross with.

        Returns:
            float: The cross product.
        """
        return np.cross(self.vector, other.vector)

    def rotate(self, angle: float) -> Vec2:
        """Rotates the vector counterclockwise by a given angle.

        Args:
            angle (float): The angle in degrees.

        Returns:
            Vec2: The rotated vector.
        """
        angle_rad = np.radians(angle)
        rotation_matrix = np.array(
            [[cos(angle_rad), -sin(angle_rad)], [sin(angle_rad), cos(angle_rad)]]
        )
        rotated_vector = rotation_matrix @ self.vector
        return Vector2D(*rotated_vector)

    def __repr__(self) -> str:
        """Returns a string representation of the vector.

        Returns:
            str: The string representation.
        """
        return f"({self.vector[0]:.2f}, {self.vector[1]:.2f})"
