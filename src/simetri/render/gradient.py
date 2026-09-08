"""Gradient stops and fill gradients for rendering backends.

Examples:
    >>> import simetri.graphics as sg
    >>> gradient = sg.Gradient(stops=((0, sg.gray), (1, sg.white)))
    >>> gradient.subtype.name
    'LINEAR'
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base.all_enums import GradientType, SvgUnits, Types
from ..coloring.colors import Color, gray, white
from ..config.settings import defaults
from ..helpers.validation import check_color, check_percent

__all__ = ["Gradient", "Stop", "normalize_stops"]


@dataclass(init=False)
class Stop:
    """A gradient stop with color and/or opacity data."""

    offset: float
    color: Color | None = None
    opacity: float | None = None

    def __init__(
        self,
        offset: float,
        color: Color | None = None,
        opacity: float | None = None,
    ):
        """Create a gradient stop.

        Args:
            offset: Stop position in ``[0, 1]``.
            color: Optional color at this stop.
            opacity: Optional opacity at this stop.

        Raises:
            ValueError: If validation of offset, color, or opacity fails.
        """
        if not check_percent(offset):
            raise ValueError("Stop offset must be between 0 and 1.0")
        if color is None and opacity is None:
            raise ValueError("Specify a color, opacity, or both.")
        if color is not None and not check_color(color):
            raise ValueError("Incorrect color value.")
        if opacity is not None and not check_percent(opacity):
            raise ValueError("Stop opacity must be between 0 and 1.0")
        self.offset = offset
        self.color = color
        self.opacity = opacity
        self.__post_init__()

    def __post_init__(self):
        self.type = Types.STOP
        self.subtype = Types.STOP


def _resolve_stops(stops):
    """Normalize stop sequences to a list of ``Stop`` instances."""
    if not isinstance(stops, (list, tuple)) or len(stops) < 2:
        raise ValueError("Invalid stop values.")
    if isinstance(stops[0], Stop):
        for stop in stops[1:]:
            if not isinstance(stop, Stop):
                raise TypeError("All stops must have the same type.")
        return stops

    stops_list = []
    for stop in stops:
        offset = stop[0]
        if not check_percent(offset):
            raise ValueError("Offset must be between 0 and 1")
        color = None
        opacity = None
        if isinstance(stop[1], float):
            opacity = stop[1]
            if not check_percent(opacity):
                raise ValueError("Offset must be between 0 and 1")
        elif isinstance(stop[1], Color):
            color = stop[1]
        if len(stop) > 2:
            color = stop[2]
            if not check_color(color):
                raise ValueError("Invalid color.")
        stops_list.append(Stop(offset, color, opacity))

    return stops_list


def normalize_stops(stops):
    """Return validated gradient stops as ``Stop`` instances.

    Args:
        stops: Stop objects or stop tuples.

    Returns:
        list[Stop] | tuple[Stop, ...]: Validated stops.
    """
    return _resolve_stops(stops)


@dataclass
class Gradient:
    """Linear or radial gradient for shape fills."""

    gradient_type: GradientType = GradientType.LINEAR
    stops: tuple = ((0, gray), (1, white))
    axis: tuple | None = ((0, 0), (1, 0))
    center: tuple[float, float] | None = None
    focal: tuple[float, float] | None = None
    radius: float | None = None
    units: SvgUnits = None
    spread_method: str | None = None
    transform: str | None = None
    subtype: Types = None

    def __post_init__(self):
        self.type = Types.GRADIENT

        if self.spread_method is None:
            self.spread_method = defaults["gradient_spread_method"]
        self.stops = _resolve_stops(self.stops)
        if self.gradient_type == GradientType.LINEAR:
            self.center = None
            self.focal = None
            self.radius = None
            self.subtype = Types.LINEAR
        else:
            self.axis = None
            if self.center is None:
                self.center = defaults["gradient_center"]
            if self.focal is None:
                self.focal = defaults["gradient_focal"]
            if self.radius is None:
                self.radius = defaults["gradient_radius"]
            if self.radius <= 0.0:
                raise ValueError("gradient radius must be positive.")
            self.subtype = Types.RADIAL