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


Vec = Union[Sequence[float], Vector]


def _as_data(vec: Vec) -> Sequence[float]:
    """Return the underlying sequence for a Vector or sequence input."""
    return vec.data if isinstance(vec, Vector) else vec


def _result_like(vec: Vec, values: Sequence[float]) -> Vec:
    """Return values as Vector if vec is Vector, otherwise as a list."""
    materialized = [x for x in values]
    return Vector(materialized) if isinstance(vec, Vector) else materialized


def v_copy(vec: Vec) -> Vec:
    """Return a shallow copy of the vector preserving output type."""
    return _result_like(vec, _as_data(vec))


def v_minus(vec: Vec) -> Vec:
    """Return the additive inverse of the vector."""
    return _result_like(vec, (-x for x in _as_data(vec)))


def v_neg(vec: Vec) -> Vec:
    """Alias for v_minus."""
    return v_minus(vec)


def v_mul(vec1: Vec, vec2: Union[Vec, float]) -> Union[float, Vec]:
    """Return dot product for vector input, or scale vec1 by scalar vec2."""
    v1 = _as_data(vec1)
    if isarray(vec2):
        v2 = _as_data(vec2)
        return sum(x * y for x, y in zip(v1, v2))
    return _result_like(vec1, (x * vec2 for x in v1))


def v_dot(vec1: Vec, vec2: Union[Vec, float]) -> Union[float, Vec]:
    """Alias for v_mul."""
    return v_mul(vec1, vec2)


def v_div(vec: Vec, c: float) -> Vec:
    """Divide a vector by scalar c and preserve output type."""
    return _result_like(vec, (x / c for x in _as_data(vec)))


def v_sum(vec1: Vec, vec2: Vec) -> Vec:
    """Return element-wise sum of two vectors."""
    v1 = _as_data(vec1)
    v2 = _as_data(vec2)
    return _result_like(vec1, (x + y for x, y in zip(v1, v2)))


def v_diff(vec1: Vec, vec2: Vec) -> Vec:
    """Return element-wise difference vec1 - vec2."""
    v1 = _as_data(vec1)
    v2 = _as_data(vec2)
    return _result_like(vec1, (x - y for x, y in zip(v1, v2)))


def v_equals(vec1: Vec, vec2: Vec) -> bool:
    """Return True when both vectors have equal components in order."""
    return list(_as_data(vec1)) == list(_as_data(vec2))


def v_cross(vec1: Vec, vec2: Vec) -> Union[Vec, float]:
    """Return 3D cross product vector or 2D scalar cross value."""
    v1 = _as_data(vec1)
    v2 = _as_data(vec2)
    if len(v1) == 3 and len(v2) == 3:
        res = [
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ]
        return _result_like(vec1, res)
    if len(v1) == 2 and len(v2) == 2:
        return v1[0] * v2[1] - v1[1] * v2[0]
    raise ValueError("Vectors must be both 2D or both 3D for cross product.")


def v_length(vec: Vec) -> float:
    """Return Euclidean norm of the vector."""
    return hypot(*_as_data(vec))


def v_angle_between(vec1: Vec, vec2: Vec) -> float:
    """Return angle in radians between vec1 and vec2."""
    ru = v_length(vec1)
    rv = v_length(vec2)
    if ru == 0 or rv == 0:
        return 0.0
    cos_val = v_mul(vec1, vec2) / (ru * rv)
    cos_val = max(min(cos_val, 1.0), -1.0)
    return acos(cos_val)


def v_arg(vec: Vec) -> float:
    """Return polar argument (atan2) of a 2D vector."""
    v = _as_data(vec)
    if len(v) != 2:
        raise ValueError("v_arg is only defined for 2D vectors.")
    return atan2(v[1], v[0])


def v_perp(vec: Vec) -> Vec:
    """Return a 2D vector rotated 90 degrees counterclockwise."""
    v = _as_data(vec)
    if len(v) != 2:
        raise ValueError("v_perp is only defined for 2D vectors.")
    return _result_like(vec, (-v[1], v[0]))


def v_rotated(vec: Vec, angle: float, axis: Optional[Vec] = None) -> Vec:
    """Return rotated vector (2D) or axis-angle rotation result (3D)."""
    c = cos(angle)
    s = sin(angle)
    v = _as_data(vec)

    if axis is None:
        if len(v) != 2:
            raise ValueError("2D rotation requires a 2D vector.")
        return _result_like(vec, (c * v[0] - s * v[1], s * v[0] + c * v[1]))

    ax = _as_data(axis)
    if len(v) != 3 or len(ax) != 3:
        raise ValueError("3D rotation requires 3D vector and axis.")

    r = v_length(ax)
    if r == 0:
        return _result_like(vec, v)

    u = [x / r for x in ax]
    k_dot_v = v_mul(u, v)
    k_cross_v = v_cross(u, v)
    return _result_like(
        vec,
        (v[i] * c + k_cross_v[i] * s + u[i] * k_dot_v * (1 - c) for i in range(3)),
    )


def v_reflect(f: Vec, vec1: Vec, vec2: Vec) -> Vec:
    """Reflect vec2 using line/plane parameters derived from f and vec1."""
    f_ = _as_data(f)
    v1 = _as_data(vec1)
    v2 = _as_data(vec2)
    r = f_[0] * v1[0] + f_[1] * v1[1]
    if r == 0:
        return _result_like(vec2, v2)

    val = f_[0] * v2[0] + f_[1] * v2[1] + f_[2]
    c = 2 * val / float(r)
    return _result_like(vec2, (v2[0] - c * v1[0], v2[1] - c * v1[1]))


def v_evaluate(line: Vec, point: Vec) -> float:
    """Evaluate implicit line coefficients [A, B, C] at a 2D point."""
    ln = _as_data(line)
    p = _as_data(point)
    return ln[0] * p[0] + ln[1] * p[1] + ln[2]


def v_line_through(point1: Vec, point2: Vec) -> List[float]:
    """Return normalized line coefficients [A, B, C] through two points."""
    p1 = _as_data(point1)
    p2 = _as_data(point2)
    A = -(p2[1] - p1[1])
    B = p2[0] - p1[0]
    C = -A * p1[0] - B * p1[1]
    r = v_length((A, B))
    if r == 0:
        return [0.0, 0.0, 0.0]
    return [A / r, B / r, C / r]


def v_intersection(line1: Vec, line2: Vec) -> List[float]:
    """Return intersection point [x, y] of two implicit lines."""
    l1 = _as_data(line1)
    l2 = _as_data(line2)
    det = l1[0] * l2[1] - l2[0] * l1[1]
    if det == 0:
        raise ValueError("Lines are parallel")
    return [
        (-l2[1] * l1[2] + l1[1] * l2[2]) / det,
        (l2[0] * l1[2] - l1[0] * l2[2]) / det,
    ]


def v_linethrough(point1: Vec, point2: Vec) -> List[float]:
    """Compatibility variant that returns [A, B, C] for a line through points."""
    p1 = _as_data(point1)
    p2 = _as_data(point2)
    v = [p2[0] - p1[0], p2[1] - p1[1]]
    r = v_length(v)
    if r == 0:
        return [0.0, 0.0, 0.0]
    A = -v[1] / r
    B = v[0] / r
    C = A * p1[0] + B * p1[1]
    return [A, B, -C]


def v_scale(vec: Vec, k: float) -> Vec:
    """Scale vector by factor k preserving output type."""
    return v_mul(vec, k)


def v_string(vec: Vec) -> str:
    """Return vector formatted as a bracketed comma-separated string."""
    return "[ " + ", ".join(str(x) for x in _as_data(vec)) + " ]"


def v_dim(vec: Vec) -> int:
    """Return number of components in the vector."""
    return len(_as_data(vec))


def v_rotate(vec: Vec, angle: float) -> Vec:
    """Rotate a 2D vector in place for sequences, or return rotated Vector."""
    if isinstance(vec, Vector):
        return v_rotated(vec, angle)

    if len(vec) != 2:
        raise ValueError("v_rotate is only defined for 2D vectors.")
    c = cos(angle)
    s = sin(angle)
    x = c * vec[0] - s * vec[1]
    y = s * vec[0] + c * vec[1]
    vec[0] = x
    vec[1] = y
    return vec


def v_interpolated(vec1: Vec, vec2: Vec, t: float) -> Vec:
    """Return linear interpolation between vec1 and vec2 at parameter t."""
    v1 = _as_data(vec1)
    v2 = _as_data(vec2)
    s = 1 - t
    return _result_like(vec1, (s * p1 + t * p2 for p1, p2 in zip(v1, v2)))


def isarray(a) -> bool:
    """Check if object is array-like (has __getitem__)."""
    return hasattr(a, "__getitem__")


def distance(point1: Vec, point2: Vec) -> float:
    """Return Euclidean distance between two points/vectors."""
    p1 = _as_data(point1)
    p2 = _as_data(point2)
    return hypot(*(q - p for p, q in zip(p1, p2)))
