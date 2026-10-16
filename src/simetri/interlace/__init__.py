"""Interlace (lace) patterns built from offset polylines.

Exports ``Lace``, ``Polyline``,
and ``ParallelPolyline``.

Examples:
    >>> from simetri.interlace import Polyline
    >>> len(Polyline([(0, 0), (40, 0)], closed=False).divisions)
    1
"""

from .lace import (
    Lace,
    ParallelPolyline,
    Polyline,
    shade_emboss,
    shade_end_quad,
    shade_triangle,
    emboss,
    shade_emboss2,
    emboss2_quads,
    get_frustum,
)
