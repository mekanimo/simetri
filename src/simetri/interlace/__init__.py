"""Interlace (lace) patterns built from offset polylines.

Exports ``Lace``, ``Polyline``,
and ``ParallelPolyline``.

Examples:
    >>> from simetri.config.settings import set_defaults
    >>> set_defaults()
    >>> from simetri.interlace import Polyline
    >>> len(Polyline([(0, 0), (10, 0)], closed=False).divisions)
    1
"""

from .lace import Lace, ParallelPolyline, Polyline
