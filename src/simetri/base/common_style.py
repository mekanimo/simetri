"""Common owned-style attributes for drawable objects.

``CommonStyle`` is a mixin for classes that store stroke/fill/marker style
on ``self``. Used by ``Shape``, ``Path2D``, and ``Pattern``; other drawables
can opt in the same way.
"""

from __future__ import annotations

from typing import Any, Self

from simetri.base.all_enums import WarningType
from simetri.coloring.colors import Color
from simetri.config.settings import defaults, issue_warning

# Public style fields copied by ``copy_style`` (color/alpha handled separately).
STYLE_COPY_ATTRS: tuple[str, ...] = (
    "line_width",
    "fill",
    "stroke",
    "line_dash_array",
    "line_dash_phase",
    "line_cap",
    "line_join",
    "line_miter_limit",
    "smooth",
    "back_style",
    "draw_double",
    "draw_fillets",
    "double_distance",
    "double_color",
    "fill_mode",
    "fillet_radius",
    "gradient",
    "draw_markers",
    "marker_type",
    "marker_size",
    "marker_radius",
    "marker_alpha",
    "marker_color",
    "marker_shape",
    "markers_only",
)

# Constructor keys handled by ``_apply_color_alpha`` / ``_init_from_style_kwargs``.
COLOR_ALPHA_ATTRS: tuple[str, ...] = (
    "color",
    "alpha",
    "line_color",
    "fill_color",
    "line_alpha",
    "fill_alpha",
)


class CommonStyle:
    """Mixin: color/alpha properties and ``copy_style``.

    Subclasses must:

    - Include the six private color/alpha names if they use ``__slots__``.
    - Call ``_init_from_style_kwargs(kwargs)`` from ``__init__`` (or the
      lower-level helpers) after handling non-style constructor args.
    - Set ``_style_warning_stacklevel`` if ``__setattr__`` adds a frame
      (e.g. Path2D under ``Group`` uses ``4``; default is ``3``).

    ``color`` / ``alpha`` fan out to line and fill when set to a non-``None``
    value. Unset line/fill color and alpha resolve to ``defaults[...]``.
    """

    _style_warning_stacklevel: int = 3

    def _init_color_alpha_state(self) -> None:
        """Set raw color/alpha fields to unset (``None``)."""
        self._alpha = None
        self._color = None
        self._line_alpha = None
        self._fill_alpha = None
        self._line_color = None
        self._fill_color = None

    def _apply_color_alpha(
        self,
        *,
        color: Color | None = None,
        alpha: float | None = None,
        line_color: Color | None = None,
        fill_color: Color | None = None,
        line_alpha: float | None = None,
        fill_alpha: float | None = None,
    ) -> None:
        """Apply constructor color/alpha args in Shape order.

        ``color`` / ``alpha`` first (fan-out), then explicit line/fill overrides.
        """
        if color is not None:
            self.color = color
        if alpha is not None:
            self.alpha = alpha
        if line_color is not None:
            self.line_color = line_color
        if fill_color is not None:
            self.fill_color = fill_color
        if line_alpha is not None:
            self.line_alpha = line_alpha
        if fill_alpha is not None:
            self.fill_alpha = fill_alpha

    def _init_style_copy_attrs(self) -> None:
        """Set every ``STYLE_COPY_ATTRS`` field to ``None``."""
        for name in STYLE_COPY_ATTRS:
            setattr(self, name, None)

    def _init_from_style_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Initialize owned style from ``kwargs`` (mutated).

        Pops known color/alpha and ``STYLE_COPY_ATTRS`` keys from ``kwargs``.
        Leaves any non-style keys for the caller. Typical subclass ``__init__``:

            def __init__(self, points=None, **kwargs):
                ...  # geometry / identity setup ...
                self._init_from_style_kwargs(kwargs)
                if kwargs:
                    raise TypeError(
                        f"Unexpected keyword arguments: {sorted(kwargs)}"
                    )
        """
        self._init_style_copy_attrs()
        self._init_color_alpha_state()

        color = kwargs.pop("color") if "color" in kwargs else None
        alpha = kwargs.pop("alpha") if "alpha" in kwargs else None
        line_color = kwargs.pop("line_color") if "line_color" in kwargs else None
        fill_color = kwargs.pop("fill_color") if "fill_color" in kwargs else None
        line_alpha = kwargs.pop("line_alpha") if "line_alpha" in kwargs else None
        fill_alpha = kwargs.pop("fill_alpha") if "fill_alpha" in kwargs else None
        self._apply_color_alpha(
            color=color,
            alpha=alpha,
            line_color=line_color,
            fill_color=fill_color,
            line_alpha=line_alpha,
            fill_alpha=fill_alpha,
        )

        for name in STYLE_COPY_ATTRS:
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))

    @property
    def color(self) -> Color | None:
        """Convenience color shared by stroke and fill when set."""
        return self._color

    @color.setter
    def color(self, value: Color | None) -> None:
        self._color = value
        if value is not None:
            issue_warning(
                "Setting 'color' also sets 'line_color' and 'fill_color'.",
                warning_type=WarningType.STYLE,
                stacklevel=self._style_warning_stacklevel,
            )
            self._line_color = value
            self._fill_color = value

    @property
    def line_color(self) -> Color:
        """Stroke color. Unset values resolve to ``defaults['line_color']``."""
        if self._line_color is None:
            return defaults["line_color"]
        return self._line_color

    @line_color.setter
    def line_color(self, value: Color | None) -> None:
        self._line_color = value

    @property
    def fill_color(self) -> Color:
        """Fill color. Unset values resolve to ``defaults['fill_color']``."""
        if self._fill_color is None:
            return defaults["fill_color"]
        return self._fill_color

    @fill_color.setter
    def fill_color(self, value: Color | None) -> None:
        self._fill_color = value

    @property
    def alpha(self) -> float | None:
        """Convenience alpha shared by stroke and fill when set."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float | None) -> None:
        self._alpha = value
        if value is not None:
            issue_warning(
                "Setting 'alpha' also sets 'line_alpha' and 'fill_alpha'.",
                warning_type=WarningType.STYLE,
                stacklevel=self._style_warning_stacklevel,
            )
            self._line_alpha = value
            self._fill_alpha = value

    @property
    def line_alpha(self) -> float:
        """Stroke alpha. Unset values resolve to ``defaults['line_alpha']``."""
        if self._line_alpha is None:
            return defaults["line_alpha"]
        return self._line_alpha

    @line_alpha.setter
    def line_alpha(self, value: float | None) -> None:
        self._line_alpha = value

    @property
    def fill_alpha(self) -> float:
        """Fill alpha. Unset values resolve to ``defaults['fill_alpha']``."""
        if self._fill_alpha is None:
            return defaults["fill_alpha"]
        return self._fill_alpha

    @fill_alpha.setter
    def fill_alpha(self, value: float | None) -> None:
        self._fill_alpha = value

    def copy_style(self, other: Any) -> Self:
        """Copy style from ``other`` onto this object (mutated).

        Copies raw color/alpha privates (no setter fan-out), then each
        name in ``STYLE_COPY_ATTRS``. ``other`` must own the same fields.
        """
        self._alpha = other._alpha
        self._color = other._color
        self._line_alpha = other._line_alpha
        self._fill_alpha = other._fill_alpha
        self._line_color = other._line_color
        self._fill_color = other._fill_color

        for name in STYLE_COPY_ATTRS:
            setattr(self, name, getattr(other, name))

        return self
