"""Vector objects and vector operations.
Any array/list can be used as arguments to the vector operations. This is the
most light-weight way to operate on high number of vectors.
Vector object instances can be used in a similar fashion. Some of the Vector
methods can be chained together.
This module borrows ideas from both PiScript by Bill Casselman and
VPython by Bruce Sherwood.
"""

from collections.abc import Sequence
from math import acos, atan2, cos, hypot, sin, sqrt
from numbers import Real

from ..base.common import LineType, PointType, VecType, axis_x, axis_y
from ..helpers.validation import check_position
from ..config.settings import issue_warning


class Vector:
    """A 2D/3D vector with an object-oriented interface over sequence ops.

    Wraps the functional ``v_*`` helpers. Many methods return new ``Vector``
    instances and can be chained.

    Examples:
        ::

            from simetri.geom.vectors import Vector

            v = Vector(3, 4)
            v.mag()  # 5.0
            (v + Vector(1, 0)).normalize()
    """

    def __init__(self, *args):
        """Initialize a vector.

        Can be initialized with:

        - Separate components: ``Vector(1, 2)`` or ``Vector(1, 2, 3)``
        - A sequence: ``Vector([1, 2])`` or ``Vector((1, 2, 3))``
        - Two points: ``Vector(p1, p2)`` (from ``p1`` to ``p2``)

        Args:
            *args: Components, a sequence, or two points.

        Raises:
            ValueError: If the arguments do not form a valid 2D/3D vector.
        """
        if not args:
            raise ValueError(
                "Vector requires 1, 2, or 3 numeric components, or two points."
            )

        if len(args) == 1:
            if not isarray(args[0]):
                raise ValueError(
                    "Vector requires a sequence of 2 or 3 numeric components."
                )
            data = list(args[0])
            if len(data) not in (2, 3):
                raise ValueError(
                    "Vector sequence input must have 2 or 3 components."
                )
            if not all(isinstance(component, Real) for component in data):
                raise ValueError("Vector components must be numeric.")
            self.data = data
            return

        if (
            len(args) == 2
            and check_position(args[0])
            and check_position(args[1])
        ):
            point_1 = args[0]
            point_2 = args[1]
            if len(point_1) != len(point_2):
                raise ValueError(
                    "Vector point inputs must have the same dimension."
                )
            self.data = [
                coord_2 - coord_1 for coord_1, coord_2 in zip(point_1, point_2)
            ]
            return

        if len(args) not in (2, 3):
            raise ValueError(
                "Vector requires 2 or 3 numeric components, or two points."
            )

        if not all(isinstance(component, Real) for component in args):
            raise ValueError("Vector components must be numeric.")

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
        return f"Vector({', '.join(str(component) for component in self.data)})"

    def __str__(self) -> str:
        return f"<{', '.join(str(component) for component in self.data)}>"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> float:
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __add__(self, other: "Vector | Sequence[float]") -> "Vector":
        """Add two vectors.

        Args:
            other: Vector or sequence to add.

        Returns:
            Vector: Element-wise sum.
        """
        if isinstance(other, Vector):
            return Vector(v_sum(self.data, other.data))
        issue_warning("Vector objects are being used with lists/tuples!")
        return Vector(v_sum(self.data, other))

    def __sub__(self, other: "Vector | Sequence[float]") -> "Vector":
        """Subtract two vectors.

        Args:
            other: Vector or sequence to subtract.

        Returns:
            Vector: Element-wise difference ``self - other``.
        """
        if isinstance(other, Vector):
            return Vector(v_diff(self.data, other.data))
        issue_warning("Vector objects are being used with lists/tuples!")
        return Vector(v_diff(self.data, other))

    def __mul__(
        self, other: "Vector | Sequence[float] | float | int"
    ) -> "float | Vector":
        """Dot product if ``other`` is a vector; otherwise scale by scalar.

        Args:
            other: Vector/sequence (dot) or scalar (scale).

        Returns:
            float | Vector: Dot product or scaled vector.
        """
        if isinstance(other, (int, float)):
            return Vector(v_mul(self.data, other))
        if isinstance(other, Vector):
            return v_mul(self.data, other.data)
        issue_warning("Vector objects are being used with lists/tuples!")
        return v_mul(self.data, other)

    def __rmul__(self, other: float) -> "Vector":
        """Reverse scalar multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "Vector":
        """Divide by scalar."""
        return Vector(v_div(self.data, other))

    def __neg__(self) -> "Vector":
        """Negate vector."""
        return Vector(v_minus(self.data))

    def __pos__(self) -> "Vector":
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

    def perp(self) -> "Vector":
        """Return a perpendicular vector (2D only)."""
        return Vector(v_perp(self.data))

    def distance_to(self, other: "Vector | Sequence[float]") -> float:
        """Return Euclidean distance to another point/vector.

        Args:
            other: Target point or vector.

        Returns:
            float: Distance between ``self`` and ``other``.
        """
        if isinstance(other, Vector):
            other_data = other.data
        else:
            issue_warning("Vector objects are being used with lists/tuples!")
            other_data = other
        return distance(self.data, other_data)

    def mag(self) -> float:
        """Magnitude (length) of the vector."""
        return v_length(self.data)

    def magnitude(self) -> float:
        """Magnitude (length) of the vector."""
        return self.mag

    def mag_sq(self) -> float:
        """Squared magnitude."""
        return sum(x * x for x in self.data)

    def normalize(self) -> "Vector":
        """Return a normalized copy of the vector."""
        mag = self.mag()
        if mag == 0:
            return Vector(self.data)
        return self / mag

    def dot(self, other: "Vector | Sequence[float]") -> float:
        """Return the dot product with ``other``.

        Args:
            other: Other vector or sequence.

        Returns:
            float: Dot product.
        """
        if isinstance(other, Vector):
            return v_mul(self.data, other.data)
        issue_warning("Vector objects are being used with lists/tuples!")
        return v_mul(self.data, other)

    def cross(self, other: "Vector | Sequence[float]") -> "float | Vector":
        """Return the cross product with ``other``.

        Args:
            other: Other vector or sequence (same dimension).

        Returns:
            float | Vector: 2D scalar cross or 3D cross-product vector.
        """
        if isinstance(other, Vector):
            other_data = other.data
        else:
            issue_warning("Vector objects are being used with lists/tuples!")
            other_data = other
        res = v_cross(self.data, other_data)
        if isinstance(res, list):
            return Vector(res)
        return res

    def angle(self) -> float:
        """Angle of 2D vector."""
        return v_arg(self.data)

    def angle_between(self, other: "Vector | Sequence[float]") -> float:
        """Return the angle in radians between ``self`` and ``other``.

        Args:
            other: Other vector or sequence.

        Returns:
            float: Angle in radians.
        """
        if isinstance(other, Vector):
            other_data = other.data
        else:
            issue_warning("Vector objects are being used with lists/tuples!")
            other_data = other
        return v_angle_between(self.data, other_data)

    def bisector(self, other: "Vector | Sequence[float]") -> "Vector":
        """Return the angle bisector of ``self`` and ``other``.

        Computed as ``self.normalize() + other.normalize()``.

        Args:
            other: Other vector or sequence.

        Returns:
            Vector: Bisector direction (not necessarily unit length).
        """
        if not isinstance(other, Vector):
            other = Vector(other)

        return self.normalize() + other.normalize()

    def rotate(
        self,
        angle: float,
        axis: "Vector | Sequence[float] | None" = None,
    ) -> "Vector":
        """Rotate this vector by ``angle`` radians.

        Args:
            angle: Rotation angle in radians.
            axis: Optional 3D axis; if None, performs 2D rotation in the plane.

        Returns:
            Vector: Rotated copy.
        """
        if isinstance(axis, Vector):
            axis_data = axis.data
        else:
            if axis is not None:
                issue_warning(
                    "Vector objects are being used with lists/tuples!"
                )
            axis_data = axis
        return Vector(v_rotated(self.data, angle, axis_data))

    def project(self, other: "Vector | Sequence[float]") -> "Vector":
        """Project this vector onto ``other``.

        Args:
            other: Direction to project onto.

        Returns:
            Vector: Parallel projection of ``self`` onto ``other``.
        """
        if isinstance(other, Vector):
            other_vec = other
        else:
            issue_warning("Vector objects are being used with lists/tuples!")
            other_vec = Vector(other)
        b_mag_sq = other_vec.mag_sq()
        if b_mag_sq == 0:
            return Vector([0.0] * len(self.data))
        scale = self.dot(other_vec) / b_mag_sq
        return other_vec * scale

    def reflect(self, normal: "Vector | Sequence[float]") -> "Vector":
        """Reflect this vector across a normal.

        Args:
            normal: Surface normal (will be normalized).

        Returns:
            Vector: Reflected vector.
        """
        if isinstance(normal, Vector):
            n = normal
        else:
            issue_warning("Vector objects are being used with lists/tuples!")
            n = Vector(normal)
        n = n.normalize()
        return self - n * (2 * self.dot(n))

    def lerp(self, other: "Vector | Sequence[float]", t: float) -> "Vector":
        """Linearly interpolate between ``self`` and ``other``.

        Args:
            other: Target vector.
            t: Interpolation parameter (0 = self, 1 = other).

        Returns:
            Vector: Interpolated vector.
        """
        if isinstance(other, Vector):
            other_data = other.data
        else:
            issue_warning("Vector objects are being used with lists/tuples!")
            other_data = other
        return Vector(v_interpolated(self.data, other_data, t))

    # Aliases
    norm = mag
    __abs__ = mag


Vec = Sequence[float] | Vector


def _as_data(vec: Vec) -> Sequence[float]:
    """Return the underlying sequence for a Vector or sequence input."""
    return vec.data if isinstance(vec, Vector) else vec


def _result_like(vec: Vec, values: Sequence[float]) -> Vec:
    """Return values as Vector if vec is Vector, otherwise as a list."""
    materialized = [x for x in values]
    return Vector(materialized) if isinstance(vec, Vector) else materialized


def v_bisector(vec1: Vec, vec2: Vec) -> Vec:
    """Returns the bisector of the given vectors.
    Vector sum of vec1 unit-vector and vec2 unit-vector.
    """
    return Vector(vec1).bisector(vec2)


def v_copy(vec: Vec) -> Vec:
    """Return a shallow copy of the vector preserving output type."""
    return _result_like(vec, _as_data(vec))


def v_minus(vec: Vec) -> Vec:
    """Return the additive inverse of the vector."""
    return _result_like(vec, (-x for x in _as_data(vec)))


def v_neg(vec: Vec) -> Vec:
    """Alias for v_minus."""
    return v_minus(vec)


def v_mul(vec1: Vec, vec2: Vec | float) -> float | Vec:
    """Return the dot product, or scale ``vec1`` by a scalar.

    Args:
        vec1: First vector.
        vec2: Second vector (dot) or scalar (scale).

    Returns:
        float | Vec: Dot product or scaled vector (type matches ``vec1``).
    """
    v1 = _as_data(vec1)
    if isarray(vec2):
        v2 = _as_data(vec2)
        return sum(x * y for x, y in zip(v1, v2))
    return _result_like(vec1, (x * vec2 for x in v1))


def v_dot(vec1: Vec, vec2: Vec | float) -> float | Vec:
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


def v_cross(vec1: Vec, vec2: Vec) -> Vec | float:
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


def v_rotated(vec: Vec, angle: float, axis: Vec | None = None) -> Vec:
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
        (
            v[i] * c + k_cross_v[i] * s + u[i] * k_dot_v * (1 - c)
            for i in range(3)
        ),
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


def v_line_through(point1: Vec, point2: Vec) -> list[float]:
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


def v_intersection(line1: Vec, line2: Vec) -> list[float]:
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


def v_linethrough(point1: Vec, point2: Vec) -> list[float]:
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


def v_from_points(start: PointType, end: PointType) -> Vec:
    """Returns the vector defined by the start and end points."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    return Vector(dx, dy)


def isarray(a) -> bool:
    """Check if object is array-like (has __getitem__)."""
    return hasattr(a, "__getitem__")


def distance(point1: Vec, point2: Vec) -> float:
    """Return Euclidean distance between two points/vectors."""
    p1 = _as_data(point1)
    p2 = _as_data(point2)
    return hypot(*(q - p for p, q in zip(p1, p2)))


def dot_product2(a: PointType, b: PointType, c: PointType) -> float:
    """Dot product of two vectors. BA and BC
    Args:
        a (PointType): First point, creating vector BA
        b (PointType): Second point, common point for both vectors
        c (PointType): Third point, creating vector BC

    Returns:
        float: The dot product of vectors BA and BC
    Note:
        The function calculates (a-b)·(c-b) which is the dot product of vectors BA and BC.
        This is useful for finding angles between segments that share a common point.
    """
    a_x, a_y = a[:2]
    b_x, b_y = b[:2]
    c_x, c_y = c[:2]
    b_a_x = a_x - b_x
    b_a_y = a_y - b_y
    b_c_x = c_x - b_x
    b_c_y = c_y - b_y
    return b_a_x * b_c_x + b_a_y * b_c_y


def cross_product2(a: PointType, b: PointType, c: PointType) -> float:
    """
    Return the cross product of two vectors: BA and BC.

    Args:
        a (PointType): First point, creating vector BA
        b (PointType): Second point, common point for both vectors
        c (PointType): Third point, creating vector BC

    Returns:
        float: The z-component of cross product between vectors BA and BC

    Note:
        This gives the signed area of the parallelogram formed by the vectors BA and BC.
        The sign indicates the orientation (positive for counter-clockwise, negative for clockwise).
        It is useful for determining the orientation of three points and calculating angles.

    vec1 = b - a
    vec2 = c - b
    """
    a_x, a_y = a[:2]
    b_x, b_y = b[:2]
    c_x, c_y = c[:2]
    b_a_x = a_x - b_x
    b_a_y = a_y - b_y
    b_c_x = c_x - b_x
    b_c_y = c_y - b_y
    return b_a_x * b_c_y - b_a_y * b_c_x


def unit_vector(line: LineType) -> VecType:
    """Return the unit vector of a line

    Args:
        line (LineType): Input line.

    Returns:
        VecType: Unit vector of the line.
    """
    norm_ = length(line)
    p1, p2 = line
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    return [(x2 - x1) / norm_, (y2 - y1) / norm_]


def unit_vector_(line: LineType) -> Sequence[VecType]:
    """Return the cartesian unit vector of a line
    with the given line's start and end points

    Args:
        line (LineType): Input line.

    Returns:
        Sequence[VecType]: Cartesian unit vector of the line.
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    dx = x2 - x1
    dy = y2 - y1
    norm_ = sqrt(dx**2 + dy**2)
    return [dx / norm_, dy / norm_]


def vec_along_line(line: LineType, magnitude: float) -> VecType:
    """Return a vector along a line with the given magnitude.

    Args:
        line (LineType): Input line.
        magnitude (float): Magnitude of the vector.

    Returns:
        VecType: Vector along the line with the given magnitude.
    """
    from .segments.line_utils import line_angle

    if line == axis_x:
        dx, dy = magnitude, 0
    elif line == axis_y:
        dx, dy = 0, magnitude
    else:
        # line is (p1, p2)
        theta = line_angle(*line)
        dx = magnitude * cos(theta)
        dy = magnitude * sin(theta)
    return dx, dy


def vec_dir_angle(vec: Sequence[float]) -> float:
    """Return the direction angle of a vector

    Args:
        vec (Sequence[float]): Input vector.

    Returns:
        float: Direction angle of the vector.
    """
    return atan2(vec[1], vec[0])


def cross_product_sense(a: PointType, b: PointType, c: PointType) -> int:
    """Return the cross product sense of vectors a and b.

    Args:
        a (PointType): First point.
        b (PointType): Second point.
        c (PointType): Third point.

    Returns:
        int: Cross product sense.
    """
    length_ = cross_product2(a, b, c)
    if length_ == 0:
        res = 1
    else:
        res = length_ / abs(length)

    return res


#      A
#      /
#     /
#   B/
#    \
#     \
#      \
#       C


def right_turn(p1, p2, p3):
    """Return True if p1, p2, p3 make a right turn.

    Args:
        p1 (PointType): First point.
        p2 (PointType): Second point.
        p3 (PointType): Third point.

    Returns:
        bool: True if the points make a right turn, False otherwise.
    """
    return cross(p1, p2, p3) < 0


def left_turn(p1, p2, p3):
    """Return True if p1, p2, p3 make a left turn.

    Args:
        p1 (PointType): First point.
        p2 (PointType): Second point.
        p3 (PointType): Third point.

    Returns:
        bool: True if the points make a left turn, False otherwise.
    """
    return cross(p1, p2, p3) > 0


def cross(p1, p2, p3):
    """Return the cross product of vectors p1p2 and p1p3.

    Args:
        p1 (PointType): First point.
        p2 (PointType): Second point.
        p3 (PointType): Third point.

    Returns:
        float: Cross product of the vectors.
    """
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return x1 * y2 - x2 * y1


def line_to_vector(line: LineType) -> VecType:
    """Return the vector representation of a line

    Args:
        line (LineType): Input line.

    Returns:
        VecType: Vector representation of the line.
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    dx = x2 - x1
    dy = y2 - y1
    return [dx, dy]


def line_vector(line: LineType) -> VecType:
    """Return the vector representation of a line.

    Args:
        line (LineType): Input line.

    Returns:
        VecType: Vector representation of the line.
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    return Vector(x2 - x1, y2 - y1)


def angled_vector(angle_: float) -> Sequence[float]:
    """
    Return a vector with the given angle

    Args:
        angle_ (float): Angle in radians.

    Returns:
        Sequence[float]: Vector with the given angle.
    """
    return [cos(angle_), sin(angle_)]


def norm(vec: VecType) -> float:
    """Return the norm (vector length) of a vector.

    Args:
        vec (VecType): Input vector.

    Returns:
        float: Norm of the vector.
    """
    return hypot(vec[0], vec[1])


def normalize(vec: VecType) -> VecType:
    """Return the normalized vector.

    Args:
        vec (VecType): Input vector.

    Returns:
        VecType: Normalized vector.
    """
    norm_ = norm(vec)
    return [vec[0] / norm_, vec[1] / norm_]


def perp_unit_vector(line: LineType) -> VecType:
    """Return the perpendicular unit vector to a line

    Args:
        line (LineType): Input line.

    Returns:
        VecType: Perpendicular unit vector.
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    dx = x2 - x1
    dy = y2 - y1
    norm_ = sqrt(dx**2 + dy**2)
    return [-dy / norm_, dx / norm_]


def point_to_line_vec(
    point: PointType, line: LineType, unit: bool = False
) -> VecType:
    """Return the perpendicular vector from a point to a line

    Args:
        point (PointType): Input point.
        line (LineType): Input line.
        unit (bool, optional): Whether to return a unit vector. Defaults to False.

    Returns:
        VecType: Perpendicular vector from the point to the line.
    """
    x0, y0 = point
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    dx = x2 - x1
    dy = y2 - y1
    norm_ = sqrt(dx**2 + dy**2)
    unit_vec = [-dy / norm_, dx / norm_]
    dist = (dx * (y1 - y0) - (x1 - x0) * dy) / sqrt(dx**2 + dy**2)
    if unit:
        if dist > 0:
            res = [unit_vec[0], unit_vec[1]]
        else:
            res = [-unit_vec[0], -unit_vec[1]]
    else:
        res = [unit_vec[0] * dist, unit_vec[1] * dist]

    return res


def surface_normal(p1: PointType, p2: PointType, p3: PointType) -> VecType:
    """
    Calculates the surface normal of a triangle given its vertices.

    Args:
        p1 (PointType): First vertex.
        p2 (PointType): Second vertex.
        p3 (PointType): Third vertex.

    Returns:
        VecType: Surface normal vector.
    """
    v1 = np.array(p1)
    v2 = np.array(p2)
    v3 = np.array(p3)
    # Create two vectors from the vertices
    u = v2 - v1
    v = v3 - v1

    # Calculate the cross product of the two vectors
    normal = np.cross(u, v)

    # Normalize the vector to get a unit normal vector
    normal = normal / np.linalg.norm(normal)

    return normal


def normal(point1, point2):
    """Return the normal vector of a line.

    Args:
        point1 (PointType): First point of the line.
        point2 (PointType): Second point of the line.

    Returns:
        VecType: Normal vector of the line.
    """
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]
    dx = x2 - x1
    dy = y2 - y1
    norm = sqrt(dx**2 + dy**2)
    return [-dy / norm, dx / norm]
