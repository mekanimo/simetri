"""Compatibility shim: Shape moved to ``simetri.shapes.shape``."""

from simetri.shapes.shape import (  # noqa: F401
    Clipping,
    Shape,
    all_segments,
    clip,
    custom_attributes,
    get_loop,
    get_partition,
    polygon_diff,
    polygon_difference,
    polygon_intersection,
    polygon_xor,
    trim_margins,
)

__all__ = [
    "Clipping",
    "Shape",
    "all_segments",
    "clip",
    "custom_attributes",
    "get_loop",
    "get_partition",
    "polygon_diff",
    "polygon_difference",
    "polygon_intersection",
    "polygon_xor",
    "trim_margins",
]
