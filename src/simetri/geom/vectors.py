"""Vector objects and vector operations.

Any array or list can be passed to the vector operations. ``Vector`` is the
object form of the same operations; many methods return a new ``Vector`` and
can be chained. Inputs are not changed unless a docstring marks an argument
``(mutated)``.

A unit vector has length 1. The zero vector has no direction, so
``normalize``, ``v_normalize``, and ``Vector.normalize`` raise
``ZeroDivisionError`` instead of returning another zero vector.

Examples:
    >>> import simetri.graphics as sg
    >>> sg.normalize([0, 5])
    [0.0, 1.0]
    >>> v = sg.Vector(3, 4)
    >>> v.normalize()
    Vector(0.6, 0.8)
    >>> v
    Vector(3, 4)
    >>> sg.normalize([0, 0])
    Traceback (most recent call last):
        ...
    ZeroDivisionError: float division by zero
"""

from collections.abc import Sequence
from math import acos, atan2, cos, hypot, sin, sqrt
from numbers import Real

import numpy as np

from ..base.common import LineType, PointType, VecType, axis_x, axis_y
from ..helpers.validation import check_position
from ..config.settings import issue_warning


class Vector:
    """A 2D/3D vector with an object-oriented interface over sequence ops.

    Wraps the functional ``v_*`` helpers. Many methods return new ``Vector``
    instances and can be chained.

    Examples:
        >>> import simetri.graphics as sg
        >>> v = sg.Vector(3, 4)
        >>> v.mag()
        5.0
        >>> v.normalize()
        Vector(0.6, 0.8)
        >>> v
        Vector(3, 4)
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
        """Return a unit-length copy. Does not change ``self``.

        Uses :func:`normalize`. A zero vector has no direction.

        Returns:
            Vector: Unit vector in the same direction.

        Raises:
            ZeroDivisionError: If this vector has length 0.

        Examples:
            >>> import simetri.graphics as sg
            >>> v = sg.Vector(0, 5)
            >>> v.normalize()
            Vector(0.0, 1.0)
            >>> v
            Vector(0, 5)
            >>> sg.Vector(0, 0).normalize()
            Traceback (most recent call last):
                ...
            ZeroDivisionError: float division by zero
        """
        return Vector(normalize(self.data))

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


i_vec = Vector(1.0, 0.0)  # x direction unit vector
j_vec = Vector(0.0, 1.0)  # y direction unit vector


Vec = Sequence[float] | Vector


def _as_data(vec: Vec) -> Sequence[float]:
    """Return the underlying sequence for a Vector or sequence input."""
    return vec.data if isinstance(vec, Vector) else vec


def _result_like(vec: Vec, values: Sequence[float]) -> Vec:
    """Return values as Vector if vec is Vector, otherwise as a list."""
    materialized = [x for x in values]
    return Vector(materialized) if isinstance(vec, Vector) else materialized


def v_bisector(vec1: Vec, vec2: Vec) -> Vec:
    """Return the sum of the two unit vectors. Does not change the inputs.

    A zero vector has no direction, so this raises ``ZeroDivisionError``.

    Args:
        vec1 (Vec): First vector. Not mutated.
        vec2 (Vec): Second vector. Not mutated.

    Returns:
        Vector: ``normalize(vec1) + normalize(vec2)``.

    Raises:
        ZeroDivisionError: If either vector has length 0.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_bisector([1, 0], [0, 1])
        Vector(1.0, 1.0)
        >>> sg.v_bisector([0, 0], [1, 0])
        Traceback (most recent call last):
            ...
        ZeroDivisionError: float division by zero
    """
    return Vector(vec1).bisector(vec2)


def v_copy(vec: Vec) -> Vec:
    """Return a copy. Does not change ``vec``.

    Args:
        vec (Vec): Input vector. Not mutated.

    Returns:
        Vec: A list if ``vec`` is a sequence, otherwise a ``Vector``.

    Examples:
        >>> import simetri.graphics as sg
        >>> raw = [1, 2]
        >>> copied = sg.v_copy(raw)
        >>> copied
        [1, 2]
        >>> copied is raw
        False
        >>> sg.v_copy(sg.Vector(1, 2))
        Vector(1, 2)
    """
    return _result_like(vec, _as_data(vec))


def v_minus(vec: Vec) -> Vec:
    """Return the additive inverse. Does not change ``vec``.

    Args:
        vec (Vec): Input vector. Not mutated.

    Returns:
        Vec: Negated components. Type matches ``vec``.

    Examples:
        >>> import simetri.graphics as sg
        >>> raw = [1, -2]
        >>> sg.v_minus(raw)
        [-1, 2]
        >>> raw
        [1, -2]
        >>> sg.v_minus(sg.Vector(0, 0))
        Vector(0, 0)
    """
    return _result_like(vec, (-x for x in _as_data(vec)))


def v_neg(vec: Vec) -> Vec:
    """Alias for :func:`v_minus`. Does not change ``vec``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_neg([1, -2])
        [-1, 2]
    """
    return v_minus(vec)


def v_mul(vec1: Vec, vec2: Vec | float) -> float | Vec:
    """Return the dot product, or scale ``vec1`` by a scalar.

    Does not change either argument.

    Args:
        vec1 (Vec): First vector. Not mutated.
        vec2 (Vec | float): Second vector (dot) or scalar (scale). Not mutated.

    Returns:
        float | Vec: Dot product, or a scaled vector whose type matches ``vec1``.

    Examples:
        >>> import simetri.graphics as sg
        >>> raw = [1, 2]
        >>> sg.v_mul(raw, [3, 4])
        11
        >>> sg.v_mul(raw, 3)
        [3, 6]
        >>> raw
        [1, 2]
        >>> sg.v_mul(sg.Vector(1, 2), 3)
        Vector(3, 6)
    """
    v1 = _as_data(vec1)
    if isarray(vec2):
        v2 = _as_data(vec2)
        return sum(x * y for x, y in zip(v1, v2))
    return _result_like(vec1, (x * vec2 for x in v1))


def v_dot(vec1: Vec, vec2: Vec | float) -> float | Vec:
    """Alias for :func:`v_mul`. Does not change either argument.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_dot([1, 2], [3, 4])
        11
    """
    return v_mul(vec1, vec2)


def v_div(vec: Vec, c: float) -> Vec:
    """Divide a vector by a scalar. Does not change ``vec``.

    Args:
        vec (Vec): Input vector. Not mutated.
        c (float): Divisor.

    Returns:
        Vec: Scaled vector. Type matches ``vec``.

    Raises:
        ZeroDivisionError: If ``c`` is 0.

    Examples:
        >>> import simetri.graphics as sg
        >>> raw = [4, 2]
        >>> sg.v_div(raw, 2)
        [2.0, 1.0]
        >>> raw
        [4, 2]
        >>> sg.v_div([1, 0], 0)
        Traceback (most recent call last):
            ...
        ZeroDivisionError: division by zero
    """
    return _result_like(vec, (x / c for x in _as_data(vec)))


def v_sum(vec1: Vec, vec2: Vec) -> Vec:
    """Return the element-wise sum. Does not change either argument.

    Args:
        vec1 (Vec): First vector. Not mutated.
        vec2 (Vec): Second vector. Not mutated.

    Returns:
        Vec: Sum. Type matches ``vec1``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_sum([1, 2], [3, 4])
        [4, 6]
        >>> sg.v_sum(sg.Vector(1, 2), [3, 4])
        Vector(4, 6)
    """
    v1 = _as_data(vec1)
    v2 = _as_data(vec2)
    return _result_like(vec1, (x + y for x, y in zip(v1, v2)))


def v_diff(vec1: Vec, vec2: Vec) -> Vec:
    """Return ``vec1 - vec2``. Does not change either argument.

    Args:
        vec1 (Vec): First vector. Not mutated.
        vec2 (Vec): Second vector. Not mutated.

    Returns:
        Vec: Difference. Type matches ``vec1``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_diff([5, 3], [1, 1])
        [4, 2]
    """
    v1 = _as_data(vec1)
    v2 = _as_data(vec2)
    return _result_like(vec1, (x - y for x, y in zip(v1, v2)))


def v_equals(vec1: Vec, vec2: Vec) -> bool:
    """Return whether both vectors have the same components.

    Args:
        vec1 (Vec): First vector. Not mutated.
        vec2 (Vec): Second vector. Not mutated.

    Returns:
        bool: True when the component lists are equal.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_equals([1, 2], sg.Vector(1, 2))
        True
        >>> sg.v_equals([1, 2], [1, 3])
        False
    """
    return list(_as_data(vec1)) == list(_as_data(vec2))


def v_cross(vec1: Vec, vec2: Vec) -> Vec | float:
    """Return the 2D scalar cross product or a 3D cross-product vector.

    Does not change either argument. Mixed dimensions are an error.

    Args:
        vec1 (Vec): First vector. Not mutated.
        vec2 (Vec): Second vector. Not mutated.

    Returns:
        Vec | float: A float in 2D, or a vector whose type matches ``vec1`` in 3D.

    Raises:
        ValueError: If the vectors are not both 2D or both 3D.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_cross([1, 0], [0, 1])
        1
        >>> sg.v_cross([1, 0, 0], [0, 1, 0])
        [0, 0, 1]
        >>> sg.v_cross([1, 0], [0, 1, 0])
        Traceback (most recent call last):
            ...
        ValueError: Vectors must be both 2D or both 3D for cross product.
    """
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
    """Return Euclidean norm of the vector.

    Args:
        vec (Vec): Input vector. Not mutated.

    Returns:
        float: Euclidean length.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_length([3, 4])
        5.0
        >>> sg.v_length(sg.Vector(0, 0))
        0.0
    """
    return hypot(*_as_data(vec))


def v_normalize(vec: Vec) -> Vec:
    """Return a unit vector. Does not change ``vec``.

    Same rule as :func:`normalize`: a zero vector raises
    ``ZeroDivisionError``. A ``Vector`` input returns a ``Vector``; a
    sequence input returns a list.

    Args:
        vec (Vec): Input vector. Not mutated.

    Returns:
        Vec: Unit vector. Type matches ``vec``.

    Raises:
        ZeroDivisionError: If ``vec`` has length 0.

    Examples:
        >>> import simetri.graphics as sg
        >>> raw = [6, 8]
        >>> sg.v_normalize(raw)
        [0.6, 0.8]
        >>> raw
        [6, 8]
        >>> vec = sg.Vector(0, 5)
        >>> sg.v_normalize(vec)
        Vector(0.0, 1.0)
        >>> vec
        Vector(0, 5)
        >>> sg.v_normalize([0, 0])
        Traceback (most recent call last):
            ...
        ZeroDivisionError: float division by zero
    """
    return _result_like(vec, normalize(_as_data(vec)))


def v_angle_between(vec1: Vec, vec2: Vec) -> float:
    """Return the angle in radians between two vectors.

    Does not change either argument. A zero vector has no direction, so
    the result is ``0.0``.

    Args:
        vec1 (Vec): First vector. Not mutated.
        vec2 (Vec): Second vector. Not mutated.

    Returns:
        float: Angle in radians, in ``[0, pi]``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_angle_between([1, 0], [0, 1])
        1.5707963267948966
        >>> sg.v_angle_between([0, 0], [1, 0])
        0.0
    """
    ru = v_length(vec1)
    rv = v_length(vec2)
    if ru == 0 or rv == 0:
        return 0.0
    cos_val = v_mul(vec1, vec2) / (ru * rv)
    cos_val = max(min(cos_val, 1.0), -1.0)
    return acos(cos_val)


def v_arg(vec: Vec) -> float:
    """Return the polar argument of a 2D vector. Does not change ``vec``.

    Args:
        vec (Vec): Input vector. Not mutated.

    Returns:
        float: ``atan2(y, x)`` in radians.

    Raises:
        ValueError: If ``vec`` is not 2D.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_arg([1, 0])
        0.0
        >>> sg.v_arg([0, 0])
        0.0
        >>> sg.v_arg([1, 0, 0])
        Traceback (most recent call last):
            ...
        ValueError: v_arg is only defined for 2D vectors.
    """
    v = _as_data(vec)
    if len(v) != 2:
        raise ValueError("v_arg is only defined for 2D vectors.")
    return atan2(v[1], v[0])


def v_perp(vec: Vec) -> Vec:
    """Return a 2D vector rotated 90 degrees counterclockwise.

    Does not change ``vec``.

    Args:
        vec (Vec): Input vector. Not mutated.

    Returns:
        Vec: ``(-y, x)``. Type matches ``vec``.

    Raises:
        ValueError: If ``vec`` is not 2D.

    Examples:
        >>> import simetri.graphics as sg
        >>> raw = [1, 0]
        >>> sg.v_perp(raw)
        [0, 1]
        >>> raw
        [1, 0]
        >>> sg.v_perp([1, 0, 0])
        Traceback (most recent call last):
            ...
        ValueError: v_perp is only defined for 2D vectors.
    """
    v = _as_data(vec)
    if len(v) != 2:
        raise ValueError("v_perp is only defined for 2D vectors.")
    return _result_like(vec, (-v[1], v[0]))


def v_rotated(vec: Vec, angle: float, axis: Vec | None = None) -> Vec:
    """Return a rotated copy. Does not change ``vec`` or ``axis``.

    With no axis, ``vec`` must be 2D. A zero 3D axis returns a copy of
    ``vec``.

    Args:
        vec (Vec): Input vector. Not mutated.
        angle (float): Rotation in radians, counterclockwise.
        axis (Vec | None): Optional 3D axis. Not mutated.

    Returns:
        Vec: Rotated vector. Type matches ``vec``.

    Raises:
        ValueError: If the dimensions do not match the rotation.

    Examples:
        >>> import simetri.graphics as sg
        >>> raw = [1, 0]
        >>> sg.v_rotated(raw, 0)
        [1.0, 0.0]
        >>> raw
        [1, 0]
        >>> sg.v_rotated([1, 0, 0], 0)
        Traceback (most recent call last):
            ...
        ValueError: 2D rotation requires a 2D vector.
    """
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
    """Reflect ``vec2`` using coefficients ``f`` and direction ``vec1``.

    Does not change the inputs. If ``f[0] * vec1[0] + f[1] * vec1[1]``
    is 0, a copy of ``vec2`` is returned.

    Args:
        f (Vec): Coefficients ``[A, B, C]``. Not mutated.
        vec1 (Vec): Direction used to reflect. Not mutated.
        vec2 (Vec): Vector to reflect. Not mutated.

    Returns:
        Vec: Reflected vector. Type matches ``vec2``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_reflect([1, 0, 0], [1, 0], [1, 1])
        [-1.0, 1.0]
        >>> raw = [3, 4]
        >>> sg.v_reflect([0, 0, 0], [1, 0], raw)
        [3, 4]
        >>> raw
        [3, 4]
    """
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
    """Evaluate implicit line ``[A, B, C]`` at a 2D point.

    Does not change either argument. A point on the line gives 0.

    Args:
        line (Vec): Line coefficients. Not mutated.
        point (Vec): Point ``(x, y)``. Not mutated.

    Returns:
        float: ``A x + B y + C``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_evaluate([1, 0, -2], [2, 5])
        0
        >>> sg.v_evaluate([1, 0, -2], [0, 0])
        -2
    """
    ln = _as_data(line)
    p = _as_data(point)
    return ln[0] * p[0] + ln[1] * p[1] + ln[2]


def v_line_through(point1: Vec, point2: Vec) -> list[float]:
    """Return normalized line coefficients ``[A, B, C]`` through two points.

    Does not change either point. Coincident points return ``[0, 0, 0]``.

    Args:
        point1 (Vec): First point. Not mutated.
        point2 (Vec): Second point. Not mutated.

    Returns:
        list[float]: Line coefficients, or zeros if the points coincide.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_line_through([0, 0], [1, 0])
        [0.0, 1.0, 0.0]
        >>> sg.v_line_through([1, 1], [1, 1])
        [0.0, 0.0, 0.0]
    """
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
    """Return the intersection of two implicit lines ``[A, B, C]``.

    Does not change either argument. Parallel lines are an error.

    Args:
        line1 (Vec): First line. Not mutated.
        line2 (Vec): Second line. Not mutated.

    Returns:
        list[float]: Intersection point ``[x, y]``.

    Raises:
        ValueError: If the lines are parallel.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_intersection([1, 0, -1], [0, 1, -2])
        [1.0, 2.0]
        >>> sg.v_intersection([1, 0, 0], [2, 0, -1])
        Traceback (most recent call last):
            ...
        ValueError: Lines are parallel
    """
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
    """Return ``[A, B, C]`` for the line through two points.

    Does not change either point. Coincident points return ``[0, 0, 0]``.

    Args:
        point1 (Vec): First point. Not mutated.
        point2 (Vec): Second point. Not mutated.

    Returns:
        list[float]: Line coefficients.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_linethrough([0, 0], [0, 1])
        [-1.0, 0.0, -0.0]
        >>> sg.v_linethrough([2, 2], [2, 2])
        [0.0, 0.0, 0.0]
    """
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
    """Scale a vector by ``k``. Does not change ``vec``.

    Args:
        vec (Vec): Input vector. Not mutated.
        k (float): Scale factor.

    Returns:
        Vec: Scaled vector. Type matches ``vec``.

    Examples:
        >>> import simetri.graphics as sg
        >>> raw = [2, 3]
        >>> sg.v_scale(raw, 2)
        [4, 6]
        >>> raw
        [2, 3]
    """
    return v_mul(vec, k)


def v_string(vec: Vec) -> str:
    """Return a bracketed component string. Does not change ``vec``.

    Args:
        vec (Vec): Input vector. Not mutated.

    Returns:
        str: Components inside brackets.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_string([1, 2])
        '[ 1, 2 ]'
    """
    return "[ " + ", ".join(str(x) for x in _as_data(vec)) + " ]"


def v_dim(vec: Vec) -> int:
    """Return the number of components. Does not change ``vec``.

    Args:
        vec (Vec): Input vector. Not mutated.

    Returns:
        int: Component count.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_dim([1, 2, 3])
        3
        >>> sg.v_dim(sg.Vector(1, 2))
        2
    """
    return len(_as_data(vec))


def v_rotate(vec: Vec, angle: float) -> Vec:
    """Rotate a 2D vector by ``angle`` radians.

    A sequence is rotated in place. A ``Vector`` is not changed; a new
    ``Vector`` is returned.

    Args:
        vec (Vec): Input vector (mutated if it is a list or other
            mutable sequence). A ``Vector`` is not changed.
        angle (float): Rotation in radians, counterclockwise.

    Returns:
        Vec: The rotated vector. Same object when ``vec`` is a sequence.

    Examples:
        >>> import simetri.graphics as sg
        >>> raw = [1.0, 0.0]
        >>> sg.v_rotate(raw, sg.pi / 2)
        [6.123233995736766e-17, 1.0]
        >>> raw[1]
        1.0
        >>> vec = sg.Vector(1, 0)
        >>> sg.v_rotate(vec, sg.pi / 2)
        Vector(6.123233995736766e-17, 1.0)
        >>> vec
        Vector(1, 0)
    """
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
    """Return the point ``(1 - t) * vec1 + t * vec2``.

    Does not change either vector. ``t`` is not clamped.

    Args:
        vec1 (Vec): Start vector. Not mutated.
        vec2 (Vec): End vector. Not mutated.
        t (float): Blend parameter. ``0`` is ``vec1``, ``1`` is ``vec2``.

    Returns:
        Vec: Interpolated vector. Type matches ``vec1``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_interpolated([0, 0], [10, 0], 0.5)
        [5.0, 0.0]
        >>> sg.v_interpolated([0, 0], [10, 0], 0)
        [0, 0]
    """
    v1 = _as_data(vec1)
    v2 = _as_data(vec2)
    s = 1 - t
    return _result_like(vec1, (s * p1 + t * p2 for p1, p2 in zip(v1, v2)))


def v_from_points(start: PointType, end: PointType) -> Vec:
    """Return the vector from ``start`` to ``end``. Does not change either point.

    Args:
        start (PointType): Start point. Not mutated.
        end (PointType): End point. Not mutated.

    Returns:
        Vector: ``end - start`` in 2D.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.v_from_points((0, 0), (3, 4))
        Vector(3, 4)
        >>> sg.v_from_points((1, 1), (1, 1))
        Vector(0, 0)
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    return Vector(dx, dy)


def isarray(a) -> bool:
    """Return whether ``a`` supports indexing.

    Args:
        a: Object to test.

    Returns:
        bool: True when ``a`` has ``__getitem__``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.isarray([1, 2])
        True
        >>> sg.isarray(3)
        False
    """
    return hasattr(a, "__getitem__")


def distance(point1: Vec, point2: Vec) -> float:
    """Return the Euclidean distance between two points. Inputs are not changed.

    Args:
        point1 (Vec): First point. Not mutated.
        point2 (Vec): Second point. Not mutated.

    Returns:
        float: Distance. ``0.0`` if the points are the same.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.distance([0, 0], [3, 4])
        5.0
        >>> sg.distance([1, 1], [1, 1])
        0.0
    """
    p1 = _as_data(point1)
    p2 = _as_data(point2)
    return hypot(*(q - p for p, q in zip(p1, p2)))


def dot_product2(a: PointType, b: PointType, c: PointType) -> float:
    """Return ``(a - b) · (c - b)``. Does not change the points.

    Args:
        a (PointType): First point, forming vector ``BA``. Not mutated.
        b (PointType): Common point. Not mutated.
        c (PointType): Third point, forming vector ``BC``. Not mutated.

    Returns:
        float: Dot product of ``BA`` and ``BC``. ``0`` if they are perpendicular.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.dot_product2((2, 0), (0, 0), (0, 3))
        0
        >>> sg.dot_product2((2, 0), (0, 0), (1, 0))
        2
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
    """Return the z-component of ``(a - b) × (c - b)``.

    Does not change the points. Positive means ``c`` is to the left of
    the direction from ``b`` to ``a``.

    Args:
        a (PointType): First point, forming vector ``BA``. Not mutated.
        b (PointType): Common point. Not mutated.
        c (PointType): Third point, forming vector ``BC``. Not mutated.

    Returns:
        float: Signed cross product. ``0`` if the points are collinear.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.cross_product2((1, 0), (0, 0), (0, 1))
        1
        >>> sg.cross_product2((1, 0), (0, 0), (2, 0))
        0
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
    """Return a unit vector along a line. Does not change ``line``.

    The body calls ``length``, which is not defined in this module, so
    a call currently raises ``NameError``.

    Args:
        line (LineType): Input line. Not mutated.

    Returns:
        VecType: Intended unit vector from the first point toward the second.

    Raises:
        NameError: ``length`` is not defined here.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.unit_vector(((0, 0), (0, 5)))
        Traceback (most recent call last):
            ...
        NameError: name 'length' is not defined
    """
    norm_ = length(line)
    p1, p2 = line
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    return [(x2 - x1) / norm_, (y2 - y1) / norm_]


def unit_vector_(line: LineType) -> Sequence[VecType]:
    """Return a unit vector along a line. Does not change ``line``.

    A zero-length line has no direction.

    Args:
        line (LineType): Input line. Not mutated.

    Returns:
        Sequence[VecType]: Unit vector ``[dx / length, dy / length]``.

    Raises:
        ZeroDivisionError: If the line has length 0.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.unit_vector_(((0, 0), (0, 5)))
        [0.0, 1.0]
        >>> sg.unit_vector_(((1, 1), (1, 1)))
        Traceback (most recent call last):
            ...
        ZeroDivisionError: float division by zero
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    dx = x2 - x1
    dy = y2 - y1
    norm_ = sqrt(dx**2 + dy**2)
    return [dx / norm_, dy / norm_]


def vec_along_line(line: LineType, magnitude: float) -> VecType:
    """Return a vector of the given length along a line.

    Does not change ``line``. The axes ``sg.axis_x`` and ``sg.axis_y``
    are handled directly.

    Args:
        line (LineType): Input line. Not mutated.
        magnitude (float): Signed length of the result.

    Returns:
        VecType: ``(dx, dy)`` along the line.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.vec_along_line(sg.axis_x, 3)
        (3, 0)
        >>> sg.vec_along_line(sg.axis_y, -2)
        (0, -2)
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
    """Return the direction angle of a 2D vector. Does not change ``vec``.

    Args:
        vec (Sequence[float]): Input vector. Not mutated.

    Returns:
        float: ``atan2(y, x)`` in radians. ``0.0`` for the zero vector.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.vec_dir_angle([1, 0])
        0.0
        >>> sg.vec_dir_angle([0, 0])
        0.0
    """
    return atan2(vec[1], vec[0])


def cross_product_sense(a: PointType, b: PointType, c: PointType) -> int:
    """Return the sign of ``(a - b) × (c - b)``.

    Does not change the points. Collinear points return ``1``. A non-zero
    cross product currently raises ``NameError`` because the body calls
    ``abs(length)`` rather than ``abs`` of the computed value.

    Args:
        a (PointType): First point. Not mutated.
        b (PointType): Common point. Not mutated.
        c (PointType): Third point. Not mutated.

    Returns:
        int: ``1`` or ``-1``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.cross_product_sense((2, 0), (0, 0), (1, 0))
        1
        >>> sg.cross_product_sense((1, 0), (0, 0), (0, 1))
        Traceback (most recent call last):
            ...
        NameError: name 'length' is not defined
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
    """Return whether ``p1``, ``p2``, ``p3`` make a right turn.

    Does not change the points. Collinear points are not a right turn.

    Args:
        p1 (PointType): First point. Not mutated.
        p2 (PointType): Second point. Not mutated.
        p3 (PointType): Third point. Not mutated.

    Returns:
        bool: True for a clockwise turn.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.right_turn((0, 0), (1, 0), (1, -1))
        True
        >>> sg.right_turn((0, 0), (1, 0), (2, 0))
        False
    """
    return cross(p1, p2, p3) < 0


def left_turn(p1, p2, p3):
    """Return whether ``p1``, ``p2``, ``p3`` make a left turn.

    Does not change the points. Collinear points are not a left turn.

    Args:
        p1 (PointType): First point. Not mutated.
        p2 (PointType): Second point. Not mutated.
        p3 (PointType): Third point. Not mutated.

    Returns:
        bool: True for a counterclockwise turn.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.left_turn((0, 0), (1, 0), (0, 1))
        True
        >>> sg.left_turn((0, 0), (1, 0), (2, 0))
        False
    """
    return cross(p1, p2, p3) > 0


def cross(p1, p2, p3):
    """Return the z-component of ``p1p2 × p1p3``. Does not change the points.

    Args:
        p1 (PointType): Common start point. Not mutated.
        p2 (PointType): End of the first vector. Not mutated.
        p3 (PointType): End of the second vector. Not mutated.

    Returns:
        float: Signed cross product. ``0`` if the points are collinear.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.cross((0, 0), (1, 0), (0, 1))
        1
        >>> sg.cross((0, 0), (1, 0), (2, 0))
        0
    """
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return x1 * y2 - x2 * y1


def line_to_vector(line: LineType) -> VecType:
    """Return the 2D vector from the first point of a line to the second.

    Does not change ``line``.

    Args:
        line (LineType): Input line. Not mutated.

    Returns:
        VecType: ``[dx, dy]``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.line_to_vector(((0, 0), (3, 4)))
        [3, 4]
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    dx = x2 - x1
    dy = y2 - y1
    return [dx, dy]


def line_vector(line: LineType) -> VecType:
    """Return a ``Vector`` from the first point of a line to the second.

    Does not change ``line``.

    Args:
        line (LineType): Input line. Not mutated.

    Returns:
        Vector: ``end - start``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.line_vector(((0, 0), (3, 4)))
        Vector(3, 4)
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    return Vector(x2 - x1, y2 - y1)


def angled_vector(angle_: float) -> Sequence[float]:
    """Return a unit vector at the given angle. Does not change ``angle_``.

    Args:
        angle_ (float): Angle in radians.

    Returns:
        Sequence[float]: ``[cos(angle), sin(angle)]``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.angled_vector(0)
        [1.0, 0.0]
    """
    return [cos(angle_), sin(angle_)]


def norm(vec: VecType) -> float:
    """Return the 2D length of a vector. Does not change ``vec``.

    Uses the first two components only.

    Args:
        vec (VecType): Input vector. Not mutated.

    Returns:
        float: Euclidean length. ``0.0`` for the zero vector.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.norm([3, 4])
        5.0
        >>> sg.norm([0, 0])
        0.0
    """
    return hypot(vec[0], vec[1])


def normalize(vec: VecType) -> VecType:
    """Return a new 2D unit vector. Does not change ``vec``.

    Uses the first two components only. A zero vector has no direction.

    Args:
        vec (VecType): Input vector. Not mutated.

    Returns:
        VecType: New list ``[x / length, y / length]``.

    Raises:
        ZeroDivisionError: If the first two components are both 0.

    Examples:
        >>> import simetri.graphics as sg
        >>> raw = [3, 4]
        >>> sg.normalize(raw)
        [0.6, 0.8]
        >>> raw
        [3, 4]
        >>> sg.normalize([0, 5])
        [0.0, 1.0]
        >>> sg.normalize([0, 0])
        Traceback (most recent call last):
            ...
        ZeroDivisionError: float division by zero
    """
    norm_ = norm(vec)
    return [vec[0] / norm_, vec[1] / norm_]


def perp_unit_vector(line: LineType) -> VecType:
    """Return a unit vector perpendicular to a line. Does not change ``line``.

    A zero-length line has no direction.

    Args:
        line (LineType): Input line. Not mutated.

    Returns:
        VecType: Unit vector ``[-dy, dx] / length``.

    Raises:
        ZeroDivisionError: If the line has length 0.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.perp_unit_vector(((0, 0), (1, 0)))
        [0.0, 1.0]
        >>> sg.perp_unit_vector(((1, 1), (1, 1)))
        Traceback (most recent call last):
            ...
        ZeroDivisionError: float division by zero
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
    """Return the perpendicular from a point to a line.

    Does not change ``point`` or ``line``. A zero-length line has no
    direction. ``unit=False`` scales the perpendicular by the signed
    distance.

    Args:
        point (PointType): Input point. Not mutated.
        line (LineType): Input line. Not mutated.
        unit (bool): If True, return a unit perpendicular. Defaults to False.

    Returns:
        VecType: Perpendicular vector from the point toward the line.

    Raises:
        ZeroDivisionError: If the line has length 0.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.point_to_line_vec((0, 1), ((0, 0), (1, 0)))
        [-0.0, -1.0]
        >>> sg.point_to_line_vec((0, 1), ((0, 0), (1, 0)), unit=True)
        [-0.0, -1.0]
        >>> sg.point_to_line_vec((0, 1), ((1, 1), (1, 1)))
        Traceback (most recent call last):
            ...
        ZeroDivisionError: float division by zero
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
    """Return a unit normal of the triangle ``p1``, ``p2``, ``p3``.

    Does not change the vertices. Collinear vertices have no direction,
    so the result is a vector of NaNs.

    Args:
        p1 (PointType): First vertex. Not mutated.
        p2 (PointType): Second vertex. Not mutated.
        p3 (PointType): Third vertex. Not mutated.

    Returns:
        VecType: Unit surface normal.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.surface_normal((0, 0, 0), (1, 0, 0), (0, 1, 0))
        array([0., 0., 1.])
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
    """Return a unit normal of the segment from ``point1`` to ``point2``.

    Does not change either point. A zero-length segment has no direction.

    Args:
        point1 (PointType): First point. Not mutated.
        point2 (PointType): Second point. Not mutated.

    Returns:
        VecType: Unit vector ``[-dy, dx] / length``.

    Raises:
        ZeroDivisionError: If the two points are the same.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.normal((0, 0), (1, 0))
        [0.0, 1.0]
        >>> sg.normal((1, 1), (1, 1))
        Traceback (most recent call last):
            ...
        ZeroDivisionError: float division by zero
    """
    x1, y1 = point1[:2]
    x2, y2 = point2[:2]
    dx = x2 - x1
    dy = y2 - y1
    norm = sqrt(dx**2 + dy**2)
    return [-dy / norm, dx / norm]
