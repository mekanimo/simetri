"""This module contains classes and functions for creating stars and rosettes."""

from math import pi, sin, cos, tan, sqrt, asin, acos, atan
from typing import Union

from ..graphics.batch import Batch
from ..graphics.shape import Shape

from ..graphics.common import common_properties, axis_x, Line
from ..graphics.all_enums import Types
from ..geometry.geometry import intersect, distance


def rosette(
    n: int,
    kernel: Union[Shape, Batch],
    cyclic: bool = False,
    axis: Line = axis_x,
    merge: bool = True,
) -> Batch:
    """Returns a pattern with cyclic or dihedral symmetry with n petals.

    Args:
        n (int): Number of petals.
        kernel (Union[Shape, Batch]): The base shape or batch to be used as a petal.
        cyclic (bool, optional): If True, creates a cyclic pattern. Defaults to False.
        axis (Line, optional): The axis for mirroring. Defaults to axis_x.
        merge (bool, optional): If True, merges shapes. Defaults to True.

    Returns:
        Batch: The resulting pattern with n petals.
    """
    if cyclic:
        petal = kernel
    else:
        petal = kernel.mirror(axis, reps=1)
        if merge:
            petal = petal.merge_shapes()
    star = petal.rotate(2 * pi / n, reps=n - 1)
    if merge:
        star = star.merge_shapes()
        if len(star) == 1:
            star = star[0]

    return star


class Star(Batch):
    """Represents a star shape with n points.

    Args:
        n (int): Number of points of the star.
        inner_radius (float, optional): Inner radius of the star. Defaults to None.
        circumradius (float, optional): Circumradius of the star. Defaults to None.
        **kwargs: Additional keyword arguments.
    """

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
            self.inner_radius = 54.12

        self.center = (0, 0)
        self.subtype = Types.STAR
        common_properties(self)
        self._initialize(n)
        super().__init__(**kwargs)

    def _initialize(self, n):
        """Initializes the star with n points.

        Args:
            n (int): Number of points of the star.

        Raises:
            ValueError: If n is less than 7.
        """
        if n < 6:
            raise ValueError("n must be greater than 5")
        if self.inner_radius is None:
            r = 50
        else:
            r = self.inner_radius  # start with a reasonable value
        # alpha = (n - 2) * pi / n
        # beta = (pi - alpha) / 2
        phi = pi / n
        gamma = 2 * phi
        theta = 3 * phi

        x1, y1 = r * cos(theta), r * sin(theta)
        up1 = (x1, y1)
        up2_ = 100, y1
        line1 = Shape([up1, up2_])
        lp1 = x1, -y1
        lp2_ = 100, -y1
        line2 = Shape([lp1, lp2_])
        up1_ = intersect(line1, line2.copy().rotate(gamma))
        ext = y1 / cos(phi)
        up2 = up1_[0] + ext, y1
        up3 = up2[0] + ext * sin(phi), 0

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
        """Calculates the kernel shape for the star.

        Args:
            segments (Shape): The segments to be used for calculation.
            n (int): Number of points of the star.

        Returns:
            tuple: A tuple containing the kernel shape, inner radius, and circumradius.
        """
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
        """Gets the kernel shape for the specified level.

        Args:
            level (int): The level of the star.

        Returns:
            tuple: A tuple containing the kernel shape, inner radius, and circumradius.
        """
        kernel = self._kernel2.copy()
        for _ in range(level - 2):
            kernel, inner_radius, circumradius = self._calc_kernel(kernel, self.n)
        return kernel, inner_radius, circumradius

    def _get_scale_factor(self, level, inner_radius=None, circumradius=None):
        """Calculates the scale factor for the specified level.

        Args:
            level (int): The level of the star.
            inner_radius (float, optional): Inner radius of the star. Defaults to None.
            circumradius (float, optional): Circumradius of the star. Defaults to None.

        Returns:
            float: The scale factor.
        """
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
        """Returns the kernel of the star at the specified level.

        Args:
            level (int): The level of the star.

        Returns:
            Shape: The kernel shape of the star.

        Raises:
            ValueError: If level is not a positive integer or zero.
        """
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
        """Returns the petal of the star at the specified level.

        Args:
            level (int): The level of the star.

        Returns:
            Shape: The petal shape of the star.

        Raises:
            ValueError: If level is not a positive integer or zero.
        """
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

        res = petal.merge_shapes()
        if len(res) == 1:
            res = res[0]

        return res

    def level(self, n: int) -> Batch:
        """Returns the star at the specified level.

        Args:
            n (int): The level of the star.

        Returns:
            Batch: The star shape at the specified level.

        Raises:
            ValueError: If level is not a positive integer or zero.
        """
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

    def find_trig_representation(target_value: float, tolerance: float = 1e-6) -> str:
        """Find trigonometric representations of a decimal value.

        Args:
            target_value (float): The decimal value to represent.
            tolerance (float): Tolerance for matching. Defaults to 1e-6.

        Returns:
            str: String representation of possible trigonometric forms.
        """
        representations = []

        # Check common angles in degrees and radians
        common_angles_deg = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        common_angles_rad = [deg * pi / 180 for deg in common_angles_deg]

        # Check basic trig functions
        for angle_deg, angle_rad in zip(common_angles_deg, common_angles_rad):
            if abs(sin(angle_rad) - target_value) < tolerance:
                representations.append(f"sin({angle_deg}°) = sin({angle_rad:.6f})")
            if abs(cos(angle_rad) - target_value) < tolerance:
                representations.append(f"cos({angle_deg}°) = cos({angle_rad:.6f})")
            if (
                angle_deg not in [90, 270]
                and abs(tan(angle_rad) - target_value) < tolerance
            ):
                representations.append(f"tan({angle_deg}°) = tan({angle_rad:.6f})")

        # Check combinations with sqrt, pi, etc.
        common_values = [
            pi / 6,
            pi / 4,
            pi / 3,
            pi / 2,
            2 * pi / 3,
            3 * pi / 4,
            5 * pi / 6,
            pi,
        ]
        for val in common_values:
            if abs(val - target_value) < tolerance:
                representations.append(
                    f"π/{6 * val / pi:.0f}" if val < pi else f"{val / pi:.3f}π"
                )

        # Check sqrt combinations
        for i in range(1, 10):
            if abs(sqrt(i) - target_value) < tolerance:
                representations.append(f"√{i}")

        # For 1.847759, check specific combinations
        if abs(target_value - 1.847759) < tolerance:
            # This is approximately tan(61.58°) or related to golden ratio calculations
            representations.append("≈ tan(61.58°)")
            representations.append(
                "≈ (1 + √5)/2 * cos(36°)"
            )  # Related to golden ratio and pentagon

        return (
            "; ".join(representations)
            if representations
            else f"No simple trig representation found for {target_value}"
        )
