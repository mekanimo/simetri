'''Vector objects and vector operations.
Any array/list can be used as arguments to the vector operations. This is the
most light-weight way to operate on high number of vectors.
Vector object instances can be used in a similar fashion. Some of the Vector
methods can be chained together.
This module borrows ideas from both PiScript by Bill Casselman and
VPython by Bruce Sherwood.
'''

from math import acos, atan2, cos, hypot, sin
from typing import List, Optional, Sequence, Tuple, Union
import warnings


Vec = Union[List[float], Tuple[float, ...]]


def v_copy(vec: Vec) -> List[float]:
    """
    Create a copy of the vector.

    Example:
        >>> v = [1, 2, 3]
        >>> v_copy(v)
        [1, 2, 3]
    """
    return [x for x in vec]


def v_minus(vec: Vec) -> List[float]:
    """
    Return the negation of the vector.

    Example:
        >>> v = [1, -2]
        >>> v_minus(v)
        [-1, 2]
    """
    return [-x for x in vec]


def v_neg(vec: Vec) -> List[float]:
    """
    Return the negation of the vector.

    Example:
        >>> v = [1, -2]
        >>> v_neg(v)
        [-1, 2]
    """
    return v_minus(vec)


def v_mul(vec1: Vec, vec2: Union[Vec, float]) -> Union[float, List[float]]:
    """
    Calculate the dot product of two vectors or multiply a vector by a scalar.

    Example:
        >>> v1 = [1, 2]
        >>> v2 = [3, 4]
        >>> v_mul(v1, v2)
        11
        >>> v_mul(v1, 2)
        [2, 4]
    """
    if isarray(vec2):
        return sum(x * y for x, y in zip(vec1, vec2))
    else:
        return [x * vec2 for x in vec1]


def v_dot(vec1: Vec, vec2: Union[Vec, float]) -> Union[float, List[float]]:
    """
    Calculate the dot product of two vectors or multiply a vector by a scalar.

    Example:
        >>> v1 = [1, 2]
        >>> v2 = [3, 4]
        >>> v_dot(v1, v2)
        11
        >>> v_dot(v1, 2)
        [2, 4]
    """
    return v_mul(vec1, vec2)


def v_div(vec: Vec, c: float) -> List[float]:
    """
    Divide a vector by a scalar.

    Example:
        >>> v = [2, 4]
        >>> v_div(v, 2)
        [1.0, 2.0]
    """
    return [x / c for x in vec]


def v_sum(vec1: Vec, vec2: Vec) -> List[float]:
    """
    Add two vectors.

    Example:
        >>> v1 = [1, 2]
        >>> v2 = [3, 4]
        >>> v_sum(v1, v2)
        [4, 6]
    """
    return [x + y for x, y in zip(vec1, vec2)]


def v_diff(vec1: Vec, vec2: Vec) -> List[float]:
    """
    Subtract the second vector from the first.

    Example:
        >>> v1 = [3, 4]
        >>> v2 = [1, 2]
        >>> v_diff(v1, v2)
        [2, 2]
    """
    return [x - y for x, y in zip(vec1, vec2)]


def v_equals(vec1: Vec, vec2: Vec) -> bool:
    """Check if two vectors are equal.
    Return True if they are equal, False otherwise.

    Example:
        >>> v1 = [1, 0, 0]
        >>> v2 = [0, 1, 0]
        >>> v_equals(v1, v2)
        False
        >>> v3 = [1.0, 0]
        >>> v4 = [1, 0]
        >>> v_equals(v3, v4)
        True
    """


def v_cross(vec1: Vec, vec2: Vec) -> Union[List[float], float]:
    """
    Calculate the cross product of two vectors.
    For 3D vectors, returns a 3D vector.
    For 2D vectors, returns the scalar magnitude of the cross product (z-component).

    Example:
        >>> v1 = [1, 0, 0]
        >>> v2 = [0, 1, 0]
        >>> v_cross(v1, v2)
        [0, 0, 1]
        >>> v3 = [1, 0]
        >>> v4 = [0, 1]
        >>> v_cross(v3, v4)
        1
    """
    if len(vec1) == 3 and len(vec2) == 3:
        return [
            vec1[1] * vec2[2] - vec1[2] * vec2[1],
            vec1[2] * vec2[0] - vec1[0] * vec2[2],
            vec1[0] * vec2[1] - vec1[1] * vec2[0],
        ]
    elif len(vec1) == 2 and len(vec2) == 2:
        return vec1[0] * vec2[1] - vec1[1] * vec2[0]
    else:
        raise ValueError(
            "Vectors must be both 2D or both 3D for cross product."
        )


def v_length(vec: Vec) -> float:
    """
    Calculate the Euclidean length (magnitude) of the vector.

    Example:
        >>> v = [3, 4]
        >>> v_length(v)
        5.0
    """
    return hypot(*vec)


def v_angle_between(vec1: Vec, vec2: Vec) -> float:
    """
    Calculate the angle in radians between two vectors.

    Example:
        >>> v1 = [1, 0]
        >>> v2 = [0, 1]
        >>> v_angle_between(v1, v2)
        1.5707963267948966
    """
    ru = v_length(vec1)
    rv = v_length(vec2)
    if ru == 0 or rv == 0:
        return 0.0
    # Clamp value to [-1, 1] to avoid domain errors due to float precision
    cos_val = v_mul(vec1, vec2) / (ru * rv)
    cos_val = max(min(cos_val, 1.0), -1.0)
    return acos(cos_val)


def v_arg(vec: Vec) -> float:
    """
    Calculate the argument (angle) of a 2D vector.

    Example:
        >>> v = [1, 1]
        >>> v_arg(v)
        0.7853981633974483
    """
    if len(vec) != 2:
        raise ValueError("v_arg is only defined for 2D vectors.")
    return atan2(vec[1], vec[0])


def v_perp(vec: Vec) -> List[float]:
    """
    Return a vector perpendicular to the given 2D vector (rotated 90 degrees counter-clockwise).

    Example:
        >>> v = [1, 0]
        >>> v_perp(v)
        [0, 1]
    """
    if len(vec) != 2:
        raise ValueError("v_perp is only defined for 2D vectors.")
    return [-vec[1], vec[0]]


def v_rotated(
    vec: Vec, angle: float, axis: Optional[Vec] = None
) -> List[float]:
    """
    Rotate a vector by an angle (in radians).
    If axis is provided, rotates a 3D vector around the axis.
    Otherwise, rotates a 2D vector.

    Example:
        >>> v = [1, 0]
        >>> v_rotated(v, pi / 2)
        [6.123233995736766e-17, 1.0]
    """
    c = cos(angle)
    s = sin(angle)

    if axis is None:
        if len(vec) != 2:
            raise ValueError("2D rotation requires a 2D vector.")
        return [c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]]
    else:
        if len(vec) != 3 or len(axis) != 3:
            raise ValueError("3D rotation requires 3D vector and axis.")

        # Normalize axis
        r = v_length(axis)
        if r == 0:
            return list(vec)
        u = [x / r for x in axis]

        # Rodrigues' rotation formula
        k_dot_v = v_mul(u, vec)
        k_cross_v = v_cross(u, vec)

        return [
            vec[i] * c + k_cross_v[i] * s + u[i] * k_dot_v * (1 - c)
            for i in range(3)
        ]


def v_reflect(f: Vec, vec1: Vec, vec2: Vec) -> List[float]:
    """
    Reflect vec2 across a line/plane defined by f and vec1.
    """
    r = f[0] * vec1[0] + f[1] * vec1[1]
    if r == 0:
        return list(vec2)

    val = f[0] * vec2[0] + f[1] * vec2[1] + f[2]
    c = 2 * val / float(r)
    return [vec2[0] - c * vec1[0], vec2[1] - c * vec1[1]]


def v_evaluate(line: Vec, point: Vec) -> float:
    """
    Evaluate a line equation at a point.
    line: [A, B, C] for Ax + By + C = 0
    point: [x, y]

    Example:
        >>> l = [1, -1, 0]
        >>> p = [2, 2]
        >>> v_evaluate(l, p)
        0
    """
    return line[0] * point[0] + line[1] * point[1] + line[2]


def v_line_through(point1: Vec, point2: Vec) -> List[float]:
    """
    Return the line equation [A, B, C] passing through two points.
    Normalized by the length of the normal vector (A, B).

    Example:
        >>> p1 = [0, 0]
        >>> p2 = [1, 1]
        >>> v_line_through(p1, p2)
        [-0.7071067811865475, 0.7071067811865475, -0.0]
    """
    A = -(point2[1] - point1[1])
    B = point2[0] - point1[0]
    C = -A * point1[0] - B * point1[1]
    r = v_length((A, B))
    if r == 0:
        return [0.0, 0.0, 0.0]
    return [A / r, B / r, C / r]


def v_intersection(line1: Vec, line2: Vec) -> List[float]:
    """
    Find the intersection point of two lines.
    Lines are given as [A, B, C] for Ax + By + C = 0.

    Example:
        >>> l1 = [1, -1, 0]  # y = x
        >>> l2 = [1, 1, -2]  # y = -x + 2
        >>> v_intersection(l1, l2)
        [1.0, 1.0]
    """
    det = line1[0] * line2[1] - line2[0] * line1[1]
    if det == 0:
        raise ValueError("Lines are parallel")
    return [
        (-line2[1] * line1[2] + line1[1] * line2[2]) / det,
        (line2[0] * line1[2] - line1[0] * line2[2]) / det,
    ]


def v_linethrough(point1: Vec, point2: Vec) -> List[float]:
    """
    Return the line equation [A, B, C] passing through two points.
    Normalized by the distance between points.
    """
    v = [point2[0] - point1[0], point2[1] - point1[1]]
    r = v_length(v)
    if r == 0:
        return [0.0, 0.0, 0.0]
    A = -v[1] / r
    B = v[0] / r
    C = A * point1[0] + B * point1[1]
    return [A, B, -C]


def v_string(vec: Vec) -> str:
    """
    Return a string representation of the vector.

    Example:
        >>> v = [1.2345, 2.3456]
        >>> v_string(v)
        '[ 1.2345, 2.3456 ]'
    """
    return "[ " + ", ".join(str(x) for x in vec) + " ]"


def v_dim(vec: Vec) -> int:
    """
    Return the dimension of the vector.

    Example:
        >>> v = [1, 2, 3]
        >>> v_dim(v)
        3
    """
    return len(vec)


def v_rotate(vec: List[float], angle: float) -> None:
    """
    Rotate a 2D vector in-place by an angle (in radians).

    Example:
        >>> v = [1.0, 0.0]
        >>> v_rotate(v, pi / 2)
        >>> v
        [6.123233995736766e-17, 1.0]
    """
    if len(vec) != 2:
        raise ValueError("v_rotate is only defined for 2D vectors.")
    c = cos(angle)
    s = sin(angle)
    x = c * vec[0] - s * vec[1]
    y = s * vec[0] + c * vec[1]
    vec[0] = x
    vec[1] = y


def v_interpolated(vec1: Vec, vec2: Vec, t: float) -> List[float]:
    """
    Linear interpolation between two vectors.

    Example:
        >>> v1 = [0, 0]
        >>> v2 = [10, 10]
        >>> v_interpolated(v1, v2, 0.5)
        [5.0, 5.0]
    """
    s = 1 - t
    return [s * p1 + t * p2 for p1, p2 in zip(vec1, vec2)]


def isarray(a) -> bool:
    """Check if object is array-like (has __getitem__)."""
    return hasattr(a, "__getitem__")


def distance(point1: Vec, point2: Vec) -> float:
    """
    Calculate the Euclidean distance between two points.

    Example:
        >>> p1 = [0, 0]
        >>> p2 = [3, 4]
        >>> distance(p1, p2)
        5.0
    """
    return hypot(*(q - p for p, q in zip(point1, point2)))


class Vector:
    """A 2D/3D vector class.

    Wraps the functional vector operations for an object-oriented interface.
    """

    def __init__(self, *args):
        """Initialize a vector.

        Can be initialized with:
        - Separate components: Vector(1, 2) or Vector(1, 2, 3)
        - A sequence: Vector([1, 2]) or Vector((1, 2, 3))

        Example:
            >>> v1 = Vector(1, 2)
            >>> v2 = Vector([3, 4, 5])
        """
        if len(args) == 1 and isarray(args[0]):
            self.data = list(args[0])
        else:
            self.data = list(args)

    @property
    def x(self) -> float:
        """Return the x component."""
        return self.data[0]

    @property
    def y(self) -> float:
        """Return the y component."""
        return self.data[1]

    @property
    def z(self) -> float:
        """Return the z component (0.0 if 2D)."""
        return self.data[2] if len(self.data) > 2 else 0.0

    def __repr__(self) -> str:
        return f"Vector({', '.join(f'{x:.2f}' for x in self.data)})"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> float:
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __add__(self, other: Union['Vector', Sequence[float]]) -> 'Vector':
        """Add two vectors."""
        if isinstance(other, Vector):
            return Vector(v_sum(self.data, other.data))
        warnings.warn(
            "Vector objects are being used with lists/tuples!", UserWarning
        )
        return Vector(v_sum(self.data, other))

    def __sub__(self, other: Union['Vector', Sequence[float]]) -> 'Vector':
        """Subtract two vectors."""
        if isinstance(other, Vector):
            return Vector(v_diff(self.data, other.data))
        warnings.warn(
            "Vector objects are being used with lists/tuples!", UserWarning
        )
        return Vector(v_diff(self.data, other))

    def __mul__(
        self, other: Union['Vector', Sequence[float], float, int]
    ) -> Union[float, 'Vector']:
        """Dot product if other is vector, scalar multiplication if scalar."""
        if isinstance(other, (int, float)):
            return Vector(v_mul(self.data, other))
        if isinstance(other, Vector):
            return v_mul(self.data, other.data)
        warnings.warn(
            "Vector objects are being used with lists/tuples!", UserWarning
        )
        return v_mul(self.data, other)

    def __rmul__(self, other: Union[float, int]) -> 'Vector':
        """Reverse scalar multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: float) -> 'Vector':
        """Divide by scalar."""
        return Vector(v_div(self.data, other))

    def __neg__(self) -> 'Vector':
        """Negate vector."""
        return Vector(v_minus(self.data))

    def __pos__(self) -> 'Vector':
        """Self vector."""
        return self

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if isinstance(other, Vector):
            return self.data == other.data
        if isarray(other):
            return self.data == list(other)
        return False

    def equals(self, other: object) -> bool:
        """Check equality."""
        return self.__eq__(other)

    def perp(self) -> 'Vector':
        """Return a perpendicular vector (2D only)."""
        return Vector(v_perp(self.data))

    def distance_to(self, other: Union['Vector', Sequence[float]]) -> float:
        """Distance to another point."""
        if isinstance(other, Vector):
            other_data = other.data
        else:
            warnings.warn(
                "Vector objects are being used with lists/tuples!", UserWarning
            )
            other_data = other
        return distance(self.data, other_data)

    def mag(self) -> float:
        """Magnitude (length) of the vector."""
        return v_length(self.data)

    def mag_sq(self) -> float:
        """Squared magnitude."""
        return sum(x * x for x in self.data)

    def normalize(self) -> 'Vector':
        """Return a normalized copy of the vector."""
        l = self.mag()
        if l == 0:
            return Vector(self.data)
        return self / l

    def dot(self, other: Union['Vector', Sequence[float]]) -> float:
        """Dot product."""
        if isinstance(other, Vector):
            return v_mul(self.data, other.data)
        warnings.warn(
            "Vector objects are being used with lists/tuples!", UserWarning
        )
        return v_mul(self.data, other)

    def cross(
        self, other: Union['Vector', Sequence[float]]
    ) -> Union[float, 'Vector']:
        """Cross product."""
        if isinstance(other, Vector):
            other_data = other.data
        else:
            warnings.warn(
                "Vector objects are being used with lists/tuples!", UserWarning
            )
            other_data = other
        res = v_cross(self.data, other_data)
        if isinstance(res, list):
            return Vector(res)
        return res

    def angle(self) -> float:
        """Angle of 2D vector."""
        return v_arg(self.data)

    def angle_between(self, other: Union['Vector', Sequence[float]]) -> float:
        """Angle between two vectors."""
        if isinstance(other, Vector):
            other_data = other.data
        else:
            warnings.warn(
                "Vector objects are being used with lists/tuples!", UserWarning
            )
            other_data = other
        return v_angle_between(self.data, other_data)

    def rotate(
        self,
        angle: float,
        axis: Optional[Union['Vector', Sequence[float]]] = None,
    ) -> 'Vector':
        """Rotate vector by angle (radians)."""
        if isinstance(axis, Vector):
            axis_data = axis.data
        else:
            if axis is not None:
                warnings.warn(
                    "Vector objects are being used with lists/tuples!",
                    UserWarning,
                )
            axis_data = axis
        return Vector(v_rotated(self.data, angle, axis_data))

    def project(self, other: Union['Vector', Sequence[float]]) -> 'Vector':
        """Project this vector onto other."""
        if isinstance(other, Vector):
            other_vec = other
        else:
            warnings.warn(
                "Vector objects are being used with lists/tuples!", UserWarning
            )
            other_vec = Vector(other)
        b_mag_sq = other_vec.mag_sq()
        if b_mag_sq == 0:
            return Vector([0.0] * len(self.data))
        scale = self.dot(other_vec) / b_mag_sq
        return other_vec * scale

    def reflect(self, normal: Union['Vector', Sequence[float]]) -> 'Vector':
        """Reflect vector across a normal."""
        if isinstance(normal, Vector):
            n = normal
        else:
            warnings.warn(
                "Vector objects are being used with lists/tuples!", UserWarning
            )
            n = Vector(normal)
        n = n.normalize()
        return self - n * (2 * self.dot(n))

    def lerp(
        self, other: Union['Vector', Sequence[float]], t: float
    ) -> 'Vector':
        """Linear interpolation."""
        if isinstance(other, Vector):
            other_data = other.data
        else:
            warnings.warn(
                "Vector objects are being used with lists/tuples!", UserWarning
            )
            other_data = other
        return Vector(v_interpolated(self.data, other_data, t))

    # Aliases
    norm = mag
    __abs__ = mag
