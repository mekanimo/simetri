"""Times New Roman ASCII letter and digit outlines as ``Path2D``.

Only ``times_glyph`` is exported (``__all__``); everything else is private.
"""

from __future__ import annotations

from typing import Any

from ..base.all_enums import FillMode
from ..geom.nonlinear.path import Path2D
from ..render.render_svg.svg_utils import extract_glyph_path

__all__ = ("times_glyph",)

_DEFAULT_FONT = r"C:\Windows\Fonts\times.ttf"
_DEFAULT_SCALE = 0.1


class _TimesGlyphsAPI:
    """Single public entry point; ``__slots__`` avoids extra instance attributes."""

    __slots__ = ()

    @staticmethod
    def times_glyph(char: str, **kwargs: Any) -> Path2D:
        """Return a ``Path2D`` for an ASCII letter or digit (Times, scale 0.1).

        Multi-contour glyphs (holes, disjoint parts) stay one path and fill with
        non-zero winding unless ``fill_mode`` or ``even_odd`` overrides.

        Args:
            char: One character to look up in the font.
            **kwargs: Style fields applied on the path (for example
                ``fill_color``).

        Raises:
            ValueError: If ``char`` is not exactly one character, or the font
                has no glyph for it.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.times_glyph("A").subtype == sg.Types.PATH2D
            True
        """
        if len(char) != 1:
            raise ValueError(f"times_glyph expects one character, got {char!r}")
        _glyph_name, path = extract_glyph_path(
            _DEFAULT_FONT, char, scale=_DEFAULT_SCALE
        )
        path.fill_mode = kwargs.pop("fill_mode", FillMode.NONZERO)
        for key, value in kwargs.items():
            setattr(path, key, value)
        return path


times_glyph = _TimesGlyphsAPI.times_glyph
