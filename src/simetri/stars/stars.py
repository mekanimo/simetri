"""This module contains classes and functions for creating stars and rosettes."""

from math import pi, sin, cos
from typing import Union

from ..graphics.batch import Batch
from ..graphics.shape import Shape

from ..graphics.common import common_properties, axis_x, Line
from ..graphics.all_enums import Types
from ..helpers.geometry import intersect, distance


def rosette(
    n: int,
    kernel: Union[Shape, Batch],
    cyclic: bool = False,
    axis: Line = axis_x,
    merge: bool = True,
) -> Batch:
    """Returns a pattern with cyclic or dihedral symmetry with n petals."""
    if cyclic:
        petal = kernel
    else:
        petal = kernel.mirror(axis, reps=1)
        if merge:
            petal = petal.merge_shapes()
    petal = petal.rotate(2 * pi / n, reps=n - 1)

    return petal


class Star(Batch):
    """Star(n:int, inner_radius:float = None, circumradius:float = None, **kwargs)"""

    def __init__(
        self, n: int, inner_radius: float = None, circumradius: float = None, **kwargs
    ):
        if circumradius is not None and inner_radius is not None:
            raise ValueError(
                "Only one of circumradius or inner_radius can be specified."
            )

        self.n = n
        self.circumradius = circumradius
        self.inner_radius = inner_radius
        if circumradius is None and inner_radius is None:
            self.inner_radius = 50

        self.center = (0, 0)
        self.subtype = Types.STAR
        common_properties(self)
        self._initialize(n)
        super().__init__(**kwargs)

    def _initialize(self, n):
        if n < 7:
            raise ValueError("n must be greater than 6")
        if self.inner_radius is None:
            r = 50
        else:
            r = self.inner_radius  # start with a reasonable value
        alpha = (n - 2) * pi / n
        beta = (pi - alpha) / 2
        gamma = 2 * pi / n
        theta = 3 * pi / n
        t = r * sin(theta)
        x = t / cos(beta)

        x1, y1 = r * cos(theta), r * sin(theta)
        up1 = (x1, y1)
        up2_ = 100, r * sin(theta)
        line1 = Shape([up1, up2_])
        lp1 = x1, -y1
        lp2_ = 100, -r * sin(theta)
        line2 = Shape([lp1, lp2_])
        up1_ = intersect(line1, line2.copy().rotate(gamma))
        up2 = up1_[0] + r, y1
        up3 = up2[0] + x * sin(beta), 0

        self._kernel2 = Shape([up1, up2, up3])
        self._petal2 = self._kernel2.copy().mirror(axis_x, reps=1)
        self._level2 = self._petal2.rotate(gamma, reps=n - 1)
        self._r2 = distance((0, 0), up1)
        self._circum2 = up3[0]

        p1 = line1.copy().rotate(-gamma)[0]
        p2 = intersect(line2.copy().rotate(gamma), line1.copy().rotate(-gamma))

        self._kernel0 = Shape([p1, p2])
        self._petal0 = self._kernel0.copy().mirror(axis_x, reps=1)
        self._level0 = self._petal0.rotate(gamma, reps=n - 1)
        self._r0 = distance((0, 0), p1)
        self._circum0 = p2[0]

        line3 = Shape([up1, up1_])
        self._kernel1 = line3.rotate(-gamma / 2)
        self._petal1 = self._kernel1.copy().mirror(axis_x, reps=1)
        self._level1 = self._petal1.rotate(gamma, reps=n - 1)
        self._r1 = distance((0, 0), up1)
        self._circum1 = up1_[0]

    def _calc_kernel(self, segments, n):
        segments = segments.copy()
        segments.rotate(pi / n)
        segments = segments.mirror(axis_x, reps=1)
        p3 = intersect(segments[0].vertex_pairs[1], segments[1].vertex_pairs[1])
        p1, p2 = segments[0].vertex_pairs[0]
        kernel = Shape([p1, p2, p3])
        inner_radius = distance((0, 0), p1)
        circumradius = distance((0, 0), p3)

        return (kernel, inner_radius, circumradius)

    def _get_kernel(self, level):
        kernel = self._kernel2.copy()
        for _ in range(level - 2):
            kernel, inner_radius, circumradius = self._calc_kernel(kernel, self.n)
        return kernel, inner_radius, circumradius

    def _get_scale_factor(self, level, inner_radius=None, circumradius=None):
        if self.inner_radius is None:
            if level == 0:
                denom = self._circum0
            elif level == 1:
                denom = self._circum1
            elif level == 2:
                denom = self._circum2
            else:
                denom = circumradius
            scale_factor = self.circumradius / denom
        else:
            if level == 0:
                denom = self._r0
            elif level == 1:
                denom = self._r1
            elif level == 2:
                denom = self._r2
            else:
                denom = inner_radius
            scale_factor = self.inner_radius / denom

        return scale_factor

    def kernel(self, level: int) -> Shape:
        """Returns the kernel of the star at the specified level."""
        if level < 0 or not isinstance(level, int):
            raise ValueError("level must be a positive integer or zero.")
        if level == 0:
            scale_factor = self._get_scale_factor(0, self._r0, self._circum0)
            kernel = self._kernel0.copy().scale(scale_factor)
        elif level == 1:
            scale_factor = self._get_scale_factor(1, self._r1, self._circum1)
            kernel = self._kernel1.copy().scale(scale_factor)
        elif level == 2:
            scale_factor = self._get_scale_factor(2, self._r2, self._circum2)
            kernel = self._kernel2.copy().scale(scale_factor)
        else:
            kernel, inner_radius, circumradius = self._get_kernel(level)
            scale_factor = self._get_scale_factor(level, inner_radius, circumradius)
            kernel = kernel.scale(scale_factor)

        return kernel

    def petal(self, level: int) -> Shape:
        """Returns the petal of the star at the specified level."""
        if level < 0 or not isinstance(level, int):
            raise ValueError("level must be a positive integer or zero.")
        if level == 0:
            scale_factor = self._get_scale_factor(0, self._r0, self._circum0)
            petal = self._kernel0.copy().mirror(axis_x, reps=1).scale(scale_factor)
        elif level == 1:
            scale_factor = self._get_scale_factor(1, self._r1, self._circum1)
            petal = self._kernel1.copy().mirror(axis_x, reps=1).scale(scale_factor)
        elif level == 2:
            scale_factor = self._get_scale_factor(2, self._r2, self._circum2)
            petal = self._kernel2.copy().mirror(axis_x, reps=1).scale(scale_factor)
        else:
            kernel, inner_radius, circumradius = self._get_kernel(level)
            scale_factor = self._get_scale_factor(level, inner_radius, circumradius)
            petal = kernel.mirror(axis_x, reps=1).scale(scale_factor)

        return petal

    def level(self, n: int) -> Batch:
        """Returns the star at the specified level."""
        if n < 0:
            raise ValueError("level must be a positive integer or zero.")
        if n == 0:
            scale_factor = self._get_scale_factor(0)
            star = self._level0.copy().scale(scale_factor)
            star.subtype = Types.STAR
            star.circumradius = self._circum0 * scale_factor
            star.inner_radius = self._r0 * scale_factor
        elif n == 1:
            scale_factor = self._get_scale_factor(1)
            star = self._level1.copy().scale(scale_factor)
            star.subtype = Types.STAR
            star.circumradius = self._circum1 * scale_factor
            star.inner_radius = self._r0 * scale_factor
        elif n == 2:
            scale_factor = self._get_scale_factor(2)
            star = self._level2.copy().scale(scale_factor)
            star.subtype = Types.STAR
            star.circumradius = self._circum2 * scale_factor
            star.inner_radius = self._r0 * scale_factor
        else:
            kernel, inner_radius, circumradius = self._get_kernel(n)
            scale_factor = self._get_scale_factor(n, inner_radius, circumradius)
            petal = kernel.mirror(axis_x, reps=1)
            star = petal.rotate(2 * pi / self.n, reps=self.n - 1)
            scale_factor = self._get_scale_factor(n, inner_radius, circumradius)
            star = star.scale(scale_factor)
            star.subtype = Types.STAR
            star.circumradius = circumradius
            star.inner_radius = inner_radius

        return star
