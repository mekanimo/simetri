"""Robert Penner easing functions for animation and interpolation.

Adapted from https://github.com/semitable/easing-functions. Each ease class
maps a progress value in ``[0, 1]`` (or a custom duration) to an eased value
between ``start`` and ``end``.

Examples:
    >>> ease = QuadEaseInOut(start=0, end=100, duration=1)
    >>> ease(0.5)  # mid-progress eased value
"""

# Penner's easing functions
# from https://github.com/semitable/easing-functions

import math


class EasingBase:
    """Base class for Penner-style easing functions.

    Attributes:
        limit: Normalized input range used by ``ease``, typically ``(0, 1)``.
        start: Output value at progress 0.
        end: Output value at progress 1.
        duration: Progress scale; ``alpha`` is divided by this before easing.
    """

    limit = (0, 1)

    def __init__(self, start: float = 0, end: float = 1, duration: float = 1):
        """Initialize the easing range.

        Args:
            start: Output value when progress is 0.
            end: Output value when progress is 1.
            duration: Divisor applied to normalized progress before ``func``.
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
        """
        raise NotImplementedError

    def ease(self, alpha: float) -> float:
        """Ease progress ``alpha`` into the configured ``start``/``end`` range.

        Args:
            alpha: Progress value (often in ``[0, 1]``).

        Returns:
            Interpolated value between ``start`` and ``end``.
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
        """
        return self.ease(alpha)


# Linear


class LinearInOut(EasingBase):
    """Linear (constant-speed) easing."""

    def func(self, t: float) -> float:
        """Return ``t`` unchanged."""
        return t


# Quadratic easing functions


class QuadEaseInOut(EasingBase):
    """Quadratic ease-in then ease-out."""

    def func(self, t: float) -> float:
        """Apply quadratic ease-in-out to ``t``."""
        if t < 0.5:
            return 2 * t * t
        return (-2 * t * t) + (4 * t) - 1


class QuadEaseIn(EasingBase):
    """Quadratic ease-in (accelerating from zero velocity)."""

    def func(self, t: float) -> float:
        """Apply quadratic ease-in to ``t``."""
        return t * t


class QuadEaseOut(EasingBase):
    """Quadratic ease-out (decelerating to zero velocity)."""

    def func(self, t: float) -> float:
        """Apply quadratic ease-out to ``t``."""
        return -(t * (t - 2))


# Cubic easing functions


class CubicEaseIn(EasingBase):
    """Cubic ease-in."""

    def func(self, t: float) -> float:
        """Apply cubic ease-in to ``t``."""
        return t * t * t


class CubicEaseOut(EasingBase):
    """Cubic ease-out."""

    def func(self, t: float) -> float:
        """Apply cubic ease-out to ``t``."""
        return (t - 1) * (t - 1) * (t - 1) + 1


class CubicEaseInOut(EasingBase):
    """Cubic ease-in then ease-out."""

    def func(self, t: float) -> float:
        """Apply cubic ease-in-out to ``t``."""
        if t < 0.5:
            return 4 * t * t * t
        p = 2 * t - 2
        return 0.5 * p * p * p + 1


# Quartic easing functions


class QuarticEaseIn(EasingBase):
    """Quartic (t^4) ease-in."""

    def func(self, t: float) -> float:
        """Apply quartic ease-in to ``t``."""
        return t * t * t * t


class QuarticEaseOut(EasingBase):
    """Quartic ease-out."""

    def func(self, t: float) -> float:
        """Apply quartic ease-out to ``t``."""
        return (t - 1) * (t - 1) * (t - 1) * (1 - t) + 1


class QuarticEaseInOut(EasingBase):
    """Quartic ease-in then ease-out."""

    def func(self, t: float) -> float:
        """Apply quartic ease-in-out to ``t``."""
        if t < 0.5:
            return 8 * t * t * t * t
        p = t - 1
        return -8 * p * p * p * p + 1


# Quintic easing functions


class QuinticEaseIn(EasingBase):
    """Quintic (t^5) ease-in."""

    def func(self, t: float) -> float:
        """Apply quintic ease-in to ``t``."""
        return t * t * t * t * t


class QuinticEaseOut(EasingBase):
    """Quintic ease-out."""

    def func(self, t: float) -> float:
        """Apply quintic ease-out to ``t``."""
        return (t - 1) * (t - 1) * (t - 1) * (t - 1) * (t - 1) + 1


class QuinticEaseInOut(EasingBase):
    """Quintic ease-in then ease-out."""

    def func(self, t: float) -> float:
        """Apply quintic ease-in-out to ``t``."""
        if t < 0.5:
            return 16 * t * t * t * t * t
        p = (2 * t) - 2
        return 0.5 * p * p * p * p * p + 1


# Sine easing functions


class SineEaseIn(EasingBase):
    """Sinusoidal ease-in."""

    def func(self, t: float) -> float:
        """Apply sine ease-in to ``t``."""
        return math.sin((t - 1) * math.pi / 2) + 1


class SineEaseOut(EasingBase):
    """Sinusoidal ease-out."""

    def func(self, t: float) -> float:
        """Apply sine ease-out to ``t``."""
        return math.sin(t * math.pi / 2)


class SineEaseInOut(EasingBase):
    """Sinusoidal ease-in then ease-out."""

    def func(self, t: float) -> float:
        """Apply sine ease-in-out to ``t``."""
        return 0.5 * (1 - math.cos(t * math.pi))


# Circular easing functions


class CircularEaseIn(EasingBase):
    """Circular ease-in."""

    def func(self, t: float) -> float:
        """Apply circular ease-in to ``t``."""
        return 1 - math.sqrt(1 - (t * t))


class CircularEaseOut(EasingBase):
    """Circular ease-out."""

    def func(self, t: float) -> float:
        """Apply circular ease-out to ``t``."""
        return math.sqrt((2 - t) * t)


class CircularEaseInOut(EasingBase):
    """Circular ease-in then ease-out."""

    def func(self, t: float) -> float:
        """Apply circular ease-in-out to ``t``."""
        if t < 0.5:
            return 0.5 * (1 - math.sqrt(1 - 4 * (t * t)))
        return 0.5 * (math.sqrt(-((2 * t) - 3) * ((2 * t) - 1)) + 1)


# Exponential easing functions


class ExponentialEaseIn(EasingBase):
    """Exponential ease-in."""

    def func(self, t: float) -> float:
        """Apply exponential ease-in to ``t``."""
        if t == 0:
            return 0
        return math.pow(2, 10 * (t - 1))


class ExponentialEaseOut(EasingBase):
    """Exponential ease-out."""

    def func(self, t: float) -> float:
        """Apply exponential ease-out to ``t``."""
        if t == 1:
            return 1
        return 1 - math.pow(2, -10 * t)


class ExponentialEaseInOut(EasingBase):
    """Exponential ease-in then ease-out."""

    def func(self, t: float) -> float:
        """Apply exponential ease-in-out to ``t``."""
        if t == 0 or t == 1:
            return t

        if t < 0.5:
            return 0.5 * math.pow(2, (20 * t) - 10)
        return -0.5 * math.pow(2, (-20 * t) + 10) + 1


# Elastic Easing Functions


class ElasticEaseIn(EasingBase):
    """Elastic ease-in (overshooting oscillation into place)."""

    def func(self, t: float) -> float:
        """Apply elastic ease-in to ``t``."""
        return math.sin(13 * math.pi / 2 * t) * math.pow(2, 10 * (t - 1))


class ElasticEaseOut(EasingBase):
    """Elastic ease-out."""

    def func(self, t: float) -> float:
        """Apply elastic ease-out to ``t``."""
        return math.sin(-13 * math.pi / 2 * (t + 1)) * math.pow(2, -10 * t) + 1


class ElasticEaseInOut(EasingBase):
    """Elastic ease-in then ease-out."""

    def func(self, t: float) -> float:
        """Apply elastic ease-in-out to ``t``."""
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
    """Back ease-in (slight overshoot backward before moving forward)."""

    def func(self, t: float) -> float:
        """Apply back ease-in to ``t``."""
        return t * t * t - t * math.sin(t * math.pi)


class BackEaseOut(EasingBase):
    """Back ease-out."""

    def func(self, t: float) -> float:
        """Apply back ease-out to ``t``."""
        p = 1 - t
        return 1 - (p * p * p - p * math.sin(p * math.pi))


class BackEaseInOut(EasingBase):
    """Back ease-in then ease-out."""

    def func(self, t: float) -> float:
        """Apply back ease-in-out to ``t``."""
        if t < 0.5:
            p = 2 * t
            return 0.5 * (p * p * p - p * math.sin(p * math.pi))

        p = 1 - (2 * t - 1)

        return 0.5 * (1 - (p * p * p - p * math.sin(p * math.pi))) + 0.5


# Bounce Easing Functions


class BounceEaseIn(EasingBase):
    """Bounce ease-in."""

    def func(self, t: float) -> float:
        """Apply bounce ease-in to ``t``."""
        return 1 - BounceEaseOut().func(1 - t)


class BounceEaseOut(EasingBase):
    """Bounce ease-out (piecewise parabolic bounce)."""

    def func(self, t: float) -> float:
        """Apply bounce ease-out to ``t``."""
        if t < 4 / 11:
            return 121 * t * t / 16
        elif t < 8 / 11:
            return (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0
        elif t < 9 / 10:
            return (4356 / 361.0 * t * t) - (35442 / 1805.0 * t) + 16061 / 1805.0
        return (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0


class BounceEaseInOut(EasingBase):
    """Bounce ease-in then ease-out."""

    def func(self, t: float) -> float:
        """Apply bounce ease-in-out to ``t``."""
        if t < 0.5:
            return 0.5 * BounceEaseIn().func(t * 2)
        return 0.5 * BounceEaseOut().func(t * 2 - 1) + 0.5


#####################################################################

q = QuadEaseInOut(1, 0, 1)

# for i in range(10):
#     print(q(i/10))


def cubicInterpolation(p0, p1, p2, p3, t):
    """Catmull-Rom style cubic interpolation between four control points.

    Args:
        p0: Point before the segment start.
        p1: Segment start point.
        p2: Segment end point.
        p3: Point after the segment end.
        t: Interpolation parameter in ``[0, 1]``.

    Returns:
        Interpolated point (scalar or array, matching the control points).
    """
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )


from numpy import array

p1 = array([0, 0])
p2 = array([1, 1])
p3 = array([2, 1])
p4 = array([3, 0])


# print(cubicInterpolation(p1, p2, p3, p4, .5))


def ease(alpha: float, duration=10, minV=0, maxV=1) -> float:
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
    """
    t = minV * (1 - alpha) + maxV * alpha
    t /= duration
    return t
    # a = self.func(t)
    # return self.end * a + self.start * (1 - a)


# print(ease(1))
