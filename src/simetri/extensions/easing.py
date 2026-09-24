"""Robert Penner easing functions for animation and interpolation.

Adapted from https://github.com/semitable/easing-functions. Each ease class
maps a progress value in ``[0, 1]`` (or a custom duration) to an eased value
between ``start`` and ``end``.

Examples:
    >>> e = QuadEaseInOut(start=0, end=100)
    >>> round(e(0.5), 2)
    50.0
"""

# Penner's easing functions
# from https://github.com/semitable/easing-functions

import math
from typing import Any

from numpy import array


class EasingBase:
    """Base class for Penner-style easing functions.

    Attributes:
        limit: Normalized input range used by ``ease``, typically ``(0, 1)``.
        start: Output value at progress 0.
        end: Output value at progress 1.
        duration: Progress scale; ``alpha`` is divided by this before easing.

    Examples:
        >>> round(LinearInOut(start=0, end=100)(0.5), 2)
        50.0
    """

    limit = (0, 1)

    def __init__(self, start: float = 0, end: float = 1, duration: float = 1) -> None:
        """Initialize the easing range.

        Args:
            start: Output value when progress is 0.
            end: Output value when progress is 1.
            duration: Divisor applied to normalized progress before ``func``.

        Examples:
            >>> e = LinearInOut(start=10, end=20)
            >>> round(e.ease(0.5), 2)
            15.0
        """
        self.start = start
        self.end = end
        self.duration = duration

    def func(self, t: float) -> float:
        """Map normalized time ``t`` in roughly ``[0, 1]`` to eased unit progress.

        Args:
            t: Normalized time.

        Returns:
            Eased unit value, typically in ``[0, 1]``.

        Raises:
            NotImplementedError: Subclasses must override this method.

        Examples:
            >>> EasingBase.func(EasingBase(), 0.5)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            NotImplementedError
        """
        raise NotImplementedError

    def ease(self, alpha: float) -> float:
        """Ease progress ``alpha`` into the configured ``start``/``end`` range.

        Args:
            alpha: Progress value (often in ``[0, 1]``).

        Returns:
            Interpolated value between ``start`` and ``end``.

        Examples:
            >>> round(LinearInOut(start=0, end=10).ease(0.5), 2)
            5.0
        """
        t = self.limit[0] * (1 - alpha) + self.limit[1] * alpha
        t /= self.duration
        a = self.func(t)
        return self.end * a + self.start * (1 - a)

    def __call__(self, alpha: float) -> float:
        """Call ``ease`` so the instance is usable as a function.

        Args:
            alpha: Progress value.

        Returns:
            Eased value between ``start`` and ``end``.

        Examples:
            >>> round(LinearInOut(start=0, end=10)(0.5), 2)
            5.0
        """
        return self.ease(alpha)


# Linear


class LinearInOut(EasingBase):
    """Linear (constant-speed) easing.

    Examples:
        >>> round(LinearInOut(start=0, end=100)(0.5), 4)
        50.0
    """

    def func(self, t: float) -> float:
        """Return ``t`` unchanged.

        Examples:
            >>> round(LinearInOut().func(0.5), 4)
            0.5
        """
        return t


# Quadratic easing functions


class QuadEaseInOut(EasingBase):
    """Quadratic ease-in then ease-out.

    Examples:
        >>> round(QuadEaseInOut(start=0, end=100)(0.5), 4)
        50.0
    """

    def func(self, t: float) -> float:
        """Apply quadratic ease-in-out to ``t``.

        Examples:
            >>> round(QuadEaseInOut().func(0.5), 6)
            0.5
        """
        if t < 0.5:
            return 2 * t * t
        return (-2 * t * t) + (4 * t) - 1


class QuadEaseIn(EasingBase):
    """Quadratic ease-in (accelerating from zero velocity).

    Examples:
        >>> round(QuadEaseIn(start=0, end=100)(0.5), 4)
        25.0
    """

    def func(self, t: float) -> float:
        """Apply quadratic ease-in to ``t``.

        Examples:
            >>> round(QuadEaseIn().func(0.5), 6)
            0.25
        """
        return t * t


class QuadEaseOut(EasingBase):
    """Quadratic ease-out (decelerating to zero velocity).

    Examples:
        >>> round(QuadEaseOut(start=0, end=100)(0.5), 4)
        75.0
    """

    def func(self, t: float) -> float:
        """Apply quadratic ease-out to ``t``.

        Examples:
            >>> round(QuadEaseOut().func(0.5), 6)
            0.75
        """
        return -(t * (t - 2))


# Cubic easing functions


class CubicEaseIn(EasingBase):
    """Cubic ease-in.

    Examples:
        >>> round(CubicEaseIn(start=0, end=100)(0.5), 4)
        12.5
    """

    def func(self, t: float) -> float:
        """Apply cubic ease-in to ``t``.

        Examples:
            >>> round(CubicEaseIn().func(0.5), 6)
            0.125
        """
        return t * t * t


class CubicEaseOut(EasingBase):
    """Cubic ease-out.

    Examples:
        >>> round(CubicEaseOut(start=0, end=100)(0.5), 4)
        87.5
    """

    def func(self, t: float) -> float:
        """Apply cubic ease-out to ``t``.

        Examples:
            >>> round(CubicEaseOut().func(0.5), 6)
            0.875
        """
        return (t - 1) * (t - 1) * (t - 1) + 1


class CubicEaseInOut(EasingBase):
    """Cubic ease-in then ease-out.

    Examples:
        >>> round(CubicEaseInOut(start=0, end=100)(0.5), 4)
        50.0
    """

    def func(self, t: float) -> float:
        """Apply cubic ease-in-out to ``t``.

        Examples:
            >>> round(CubicEaseInOut().func(0.5), 6)
            0.5
        """
        if t < 0.5:
            return 4 * t * t * t
        p = 2 * t - 2
        return 0.5 * p * p * p + 1


# Quartic easing functions


class QuarticEaseIn(EasingBase):
    """Quartic (t^4) ease-in.

    Examples:
        >>> round(QuarticEaseIn(start=0, end=100)(0.5), 4)
        6.25
    """

    def func(self, t: float) -> float:
        """Apply quartic ease-in to ``t``.

        Examples:
            >>> round(QuarticEaseIn().func(0.5), 6)
            0.0625
        """
        return t * t * t * t


class QuarticEaseOut(EasingBase):
    """Quartic ease-out.

    Examples:
        >>> round(QuarticEaseOut(start=0, end=100)(0.5), 4)
        93.75
    """

    def func(self, t: float) -> float:
        """Apply quartic ease-out to ``t``.

        Examples:
            >>> round(QuarticEaseOut().func(0.5), 6)
            0.9375
        """
        return (t - 1) * (t - 1) * (t - 1) * (1 - t) + 1


class QuarticEaseInOut(EasingBase):
    """Quartic ease-in then ease-out.

    Examples:
        >>> round(QuarticEaseInOut(start=0, end=100)(0.5), 4)
        50.0
    """

    def func(self, t: float) -> float:
        """Apply quartic ease-in-out to ``t``.

        Examples:
            >>> round(QuarticEaseInOut().func(0.5), 6)
            0.5
        """
        if t < 0.5:
            return 8 * t * t * t * t
        p = t - 1
        return -8 * p * p * p * p + 1


# Quintic easing functions


class QuinticEaseIn(EasingBase):
    """Quintic (t^5) ease-in.

    Examples:
        >>> round(QuinticEaseIn(start=0, end=100)(0.5), 4)
        3.125
    """

    def func(self, t: float) -> float:
        """Apply quintic ease-in to ``t``.

        Examples:
            >>> round(QuinticEaseIn().func(0.5), 6)
            0.03125
        """
        return t * t * t * t * t


class QuinticEaseOut(EasingBase):
    """Quintic ease-out.

    Examples:
        >>> round(QuinticEaseOut(start=0, end=100)(0.5), 4)
        96.875
    """

    def func(self, t: float) -> float:
        """Apply quintic ease-out to ``t``.

        Examples:
            >>> round(QuinticEaseOut().func(0.5), 6)
            0.96875
        """
        return (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1


class QuinticEaseInOut(EasingBase):
    """Quintic ease-in then ease-out.

    Examples:
        >>> round(QuinticEaseInOut(start=0, end=100)(0.5), 4)
        50.0
    """

    def func(self, t: float) -> float:
        """Apply quintic ease-in-out to ``t``.

        Examples:
            >>> round(QuinticEaseInOut().func(0.5), 6)
            0.5
        """
        if t < 0.5:
            return 16 * t * t * t * t * t
        p = (2 * t) - 2
        return 0.5 * p * p * p * p * p + 1


# Sine easing functions


class SineEaseIn(EasingBase):
    """Sinusoidal ease-in.

    Examples:
        >>> round(SineEaseIn(start=0, end=100)(0.5), 4)
        29.2893
    """

    def func(self, t: float) -> float:
        """Apply sine ease-in to ``t``.

        Examples:
            >>> round(SineEaseIn().func(0.5), 6)
            0.292893
        """
        return math.sin((t - 1) * math.pi / 2) + 1


class SineEaseOut(EasingBase):
    """Sinusoidal ease-out.

    Examples:
        >>> round(SineEaseOut(start=0, end=100)(0.5), 4)
        70.7107
    """

    def func(self, t: float) -> float:
        """Apply sine ease-out to ``t``.

        Examples:
            >>> round(SineEaseOut().func(0.5), 6)
            0.707107
        """
        return math.sin(t * math.pi / 2)


class SineEaseInOut(EasingBase):
    """Sinusoidal ease-in then ease-out.

    Examples:
        >>> round(SineEaseInOut(start=0, end=100)(0.5), 4)
        50.0
    """

    def func(self, t: float) -> float:
        """Apply sine ease-in-out to ``t``.

        Examples:
            >>> round(SineEaseInOut().func(0.5), 6)
            0.5
        """
        return 0.5 * (1 - math.cos(t * math.pi))


# Circular easing functions


class CircularEaseIn(EasingBase):
    """Circular ease-in.

    Examples:
        >>> round(CircularEaseIn(start=0, end=100)(0.5), 4)
        13.3975
    """

    def func(self, t: float) -> float:
        """Apply circular ease-in to ``t``.

        Examples:
            >>> round(CircularEaseIn().func(0.5), 6)
            0.133975
        """
        return 1 - math.sqrt(1 - (t * t))


class CircularEaseOut(EasingBase):
    """Circular ease-out.

    Examples:
        >>> round(CircularEaseOut(start=0, end=100)(0.5), 4)
        86.6025
    """

    def func(self, t: float) -> float:
        """Apply circular ease-out to ``t``.

        Examples:
            >>> round(CircularEaseOut().func(0.5), 6)
            0.866025
        """
        return math.sqrt((2 - t) * t)


class CircularEaseInOut(EasingBase):
    """Circular ease-in then ease-out.

    Examples:
        >>> round(CircularEaseInOut(start=0, end=100)(0.5), 4)
        50.0
    """

    def func(self, t: float) -> float:
        """Apply circular ease-in-out to ``t``.

        Examples:
            >>> round(CircularEaseInOut().func(0.5), 6)
            0.5
        """
        if t < 0.5:
            return 0.5 * (1 - math.sqrt(1 - 4 * (t * t)))
        return 0.5 * (math.sqrt(-((2 * t) - 3) * ((2 * t) - 1)) + 1)


# Exponential easing functions


class ExponentialEaseIn(EasingBase):
    """Exponential ease-in.

    Examples:
        >>> round(ExponentialEaseIn(start=0, end=100)(0.5), 4)
        3.125
    """

    def func(self, t: float) -> float:
        """Apply exponential ease-in to ``t``.

        Examples:
            >>> round(ExponentialEaseIn().func(0.5), 6)
            0.03125
        """
        if t == 0:
            return 0
        return math.pow(2, 10 * (t - 1))


class ExponentialEaseOut(EasingBase):
    """Exponential ease-out.

    Examples:
        >>> round(ExponentialEaseOut(start=0, end=100)(0.5), 4)
        96.875
    """

    def func(self, t: float) -> float:
        """Apply exponential ease-out to ``t``.

        Examples:
            >>> round(ExponentialEaseOut().func(0.5), 6)
            0.96875
        """
        if t == 1:
            return 1
        return 1 - math.pow(2, -10 * t)


class ExponentialEaseInOut(EasingBase):
    """Exponential ease-in then ease-out.

    Examples:
        >>> round(ExponentialEaseInOut(start=0, end=100)(0.5), 4)
        50.0
    """

    def func(self, t: float) -> float:
        """Apply exponential ease-in-out to ``t``.

        Examples:
            >>> round(ExponentialEaseInOut().func(0.5), 6)
            0.5
        """
        if t == 0 or t == 1:
            return t

        if t < 0.5:
            return 0.5 * math.pow(2, (20 * t) - 10)
        return -0.5 * math.pow(2, (-20 * t) + 10) + 1


# Elastic Easing Functions


class ElasticEaseIn(EasingBase):
    """Elastic ease-in (overshooting oscillation into place).

    Examples:
        >>> round(ElasticEaseIn(start=0, end=100)(0.5), 4)
        -2.2097
    """

    def func(self, t: float) -> float:
        """Apply elastic ease-in to ``t``.

        Examples:
            >>> round(ElasticEaseIn().func(0.5), 6)
            -0.022097
        """
        return math.sin(13 * math.pi / 2 * t) * math.pow(2, 10 * (t - 1))


class ElasticEaseOut(EasingBase):
    """Elastic ease-out.

    Examples:
        >>> round(ElasticEaseOut(start=0, end=100)(0.5), 4)
        102.2097
    """

    def func(self, t: float) -> float:
        """Apply elastic ease-out to ``t``.

        Examples:
            >>> round(ElasticEaseOut().func(0.5), 6)
            1.022097
        """
        return math.sin(-13 * math.pi / 2 * (t + 1)) * math.pow(2, -10 * t) + 1


class ElasticEaseInOut(EasingBase):
    """Elastic ease-in then ease-out.

    Examples:
        >>> round(ElasticEaseInOut(start=0, end=100)(0.5), 4)
        50.0
    """

    def func(self, t: float) -> float:
        """Apply elastic ease-in-out to ``t``.

        Examples:
            >>> round(ElasticEaseInOut().func(0.5), 6)
            0.5
        """
        if t < 0.5:
            return (
                0.5
                * math.sin(13 * math.pi / 2 * (2 * t))
                * math.pow(2, 10 * ((2 * t) - 1))
            )
        return 0.5 * (
            math.sin(-13 * math.pi / 2 * ((2 * t - 1) + 1))
            * math.pow(2, -10 * (2 * t - 1))
            + 2
        )


# Back Easing Functions


class BackEaseIn(EasingBase):
    """Back ease-in (slight overshoot backward before moving forward).

    Examples:
        >>> round(BackEaseIn(start=0, end=100)(0.5), 4)
        -37.5
    """

    def func(self, t: float) -> float:
        """Apply back ease-in to ``t``.

        Examples:
            >>> round(BackEaseIn().func(0.5), 6)
            -0.375
        """
        return t * t * t - t * math.sin(t * math.pi)


class BackEaseOut(EasingBase):
    """Back ease-out.

    Examples:
        >>> round(BackEaseOut(start=0, end=100)(0.5), 4)
        137.5
    """

    def func(self, t: float) -> float:
        """Apply back ease-out to ``t``.

        Examples:
            >>> round(BackEaseOut().func(0.5), 6)
            1.375
        """
        p = 1 - t
        return 1 - (p * p * p - p * math.sin(p * math.pi))


class BackEaseInOut(EasingBase):
    """Back ease-in then ease-out.

    Examples:
        >>> round(BackEaseInOut(start=0, end=100)(0.5), 4)
        50.0
    """

    def func(self, t: float) -> float:
        """Apply back ease-in-out to ``t``.

        Examples:
            >>> round(BackEaseInOut().func(0.5), 6)
            0.5
        """
        if t < 0.5:
            p = 2 * t
            return 0.5 * (p * p * p - p * math.sin(p * math.pi))

        p = 1 - (2 * t - 1)

        return 0.5 * (1 - (p * p * p - p * math.sin(p * math.pi))) + 0.5


# Bounce Easing Functions


class BounceEaseIn(EasingBase):
    """Bounce ease-in.

    Examples:
        >>> round(BounceEaseIn(start=0, end=100)(0.5), 4)
        28.125
    """

    def func(self, t: float) -> float:
        """Apply bounce ease-in to ``t``.

        Examples:
            >>> round(BounceEaseIn().func(0.5), 6)
            0.28125
        """
        return 1 - BounceEaseOut().func(1 - t)


class BounceEaseOut(EasingBase):
    """Bounce ease-out (piecewise parabolic bounce).

    Examples:
        >>> round(BounceEaseOut(start=0, end=100)(0.5), 4)
        71.875
    """

    def func(self, t: float) -> float:
        """Apply bounce ease-out to ``t``.

        Examples:
            >>> round(BounceEaseOut().func(0.5), 6)
            0.71875
        """
        if t < 4 / 11:
            return 121 * t * t / 16
        elif t < 8 / 11:
            return (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0
        elif t < 9 / 10:
            return (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0
        return (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0


class BounceEaseInOut(EasingBase):
    """Bounce ease-in then ease-out.

    Examples:
        >>> round(BounceEaseInOut(start=0, end=100)(0.5), 4)
        50.0
    """

    def func(self, t: float) -> float:
        """Apply bounce ease-in-out to ``t``.

        Examples:
            >>> round(BounceEaseInOut().func(0.5), 6)
            0.5
        """
        if t < 0.5:
            return 0.5 * BounceEaseIn().func(t * 2)
        return 0.5 * BounceEaseOut().func(t * 2 - 1) + 0.5


#####################################################################

q = QuadEaseInOut(1, 0, 1)

# for i in range(10):
#     print(q(i/10))


def cubicInterpolation(
    p0: Any, p1: Any, p2: Any, p3: Any, t: float
) -> Any:
    """Catmull-Rom style cubic interpolation between four control points.

    Args:
        p0: Point before the segment start.
        p1: Segment start point.
        p2: Segment end point.
        p3: Point after the segment end.
        t: Interpolation parameter in ``[0, 1]``.

    Returns:
        Interpolated point (scalar or array, matching the control points).

    Examples:
        >>> p0 = array([0.0, 0.0])
        >>> p1 = array([0.0, 0.0])
        >>> p2 = array([2.0, 0.0])
        >>> p3 = array([2.0, 0.0])
        >>> [float(x) for x in cubicInterpolation(p0, p1, p2, p3, 0.5)]
        [1.0, 0.0]
    """
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


p1 = array([0, 0])
p2 = array([1, 1])
p3 = array([2, 1])
p4 = array([3, 0])


# print(cubicInterpolation(p1, p2, p3, p4, .5))


def ease(
    alpha: float,
    duration: float = 10,
    minV: float = 0,
    maxV: float = 1,
) -> float:
    """Linearly map ``alpha`` into ``[minV, maxV]`` scaled by ``duration``.

    Note:
        Unlike the ``EasingBase`` subclasses, this helper does not apply a
        nonlinear easing curve; it only remaps the progress value.

    Args:
        alpha: Progress value.
        duration: Divisor applied after remapping.
        minV: Output contribution when ``alpha`` is 0.
        maxV: Output contribution when ``alpha`` is 1.

    Returns:
        Remapped progress ``(minV * (1 - alpha) + maxV * alpha) / duration``.

    Examples:
        >>> round(ease(0.5, duration=2), 4)
        0.25
        >>> round(ease(1.0, duration=10, minV=0, maxV=100), 2)
        10.0
    """
    t = minV * (1 - alpha) + maxV * alpha
    t /= duration
    return t
    # a = self.func(t)
    # return self.end * a + self.start * (1 - a)


# print(ease(1))
