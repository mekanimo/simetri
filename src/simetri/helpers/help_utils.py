"""Interactive and topic-based help for Simetri (``sg.help``).

``sg.help(obj)`` accepts a string key or a callable/class/instance:

- String keys look up ``defaults_help[obj]`` (empty string if missing),
  except for reserved topic names described below.
- Classes (and instances of Simetri types) return the constructor
  signature, class docstring, and ``__init__`` docstring.
- Functions, methods, modules, and other objects return
  ``inspect.getdoc(obj)``.

``d_help_topic`` maps topic names to short descriptions and lists of
related ``sg.*`` names. ``sg.help('help')`` or ``sg.help(sg.help)``
summarizes how help works. ``sg.help('topics')`` lists available topics.

Examples:
    >>> import simetri.graphics as sg
    >>> 'distance' in sg.help('points')
    True
    >>> 'Shape' in sg.help('shapes')
    True
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence

from ..config.settings import defaults_help

# Topic name -> list of public ``sg.*`` names (and brief notes as plain lines).
d_help_topic: dict[str, list[str]] = {
    "points": [
        "sg.distance",
        "sg.distance2",
        "sg.midpoint",
        "sg.homogenize",
        "sg.cart_to_tri",
        "sg.tri_to_cart",
        "sg.lerp_point",
        "sg.offset_point",
        "sg.offset_point_from_start",
        "sg.close_points2",
        "sg.round_point",
        "sg.fix_degen_points",
        "sg.left",
        "sg.on_segment",
        "sg.point_on_line_segment",
        "sg.r_polar",
        "sg.connected_pairs",
        "sg.rel_coord",
        "sg.rel_polar",
        # Additional helpers live in simetri.geom.points.point_utils
        # (equal_points, project_point_on_line, remove_duplicate_points, …).
    ],
    "lines": [
        "sg.Line",
        "sg.Segment",
        "sg.line_shape",
        "sg.offset_line",
        "sg.extended_line",
        "sg.intersect",
        "sg.line_angle",
        "sg.angle_between_two_lines",
        "sg.angle_between_lines2",
        "sg.equal_edges",
        "sg.round_segment",
        "sg.stitch",
        "sg.clip_line_to_rect",
        "sg.line_by_point_angle_length",
        "sg.all_intersections",
        "sg.fillet_corners",
        "sg.axis",
    ],
    "segments": [],  # filled as alias of lines below
    "vertices": [
        "Shape.vertices / Shape.primary_points",
        "sg.round_point",
        "sg.fix_degen_points",
        "sg.close_points2",
        "sg.homogenize",
        "See also: sg.help('points'), sg.help('shapes')",
    ],
    "edges": [
        "sg.Edge",
        "sg.Segment",
        "sg.equal_edges",
        "sg.all_segments",
        "sg.offset_line",
        "sg.intersect",
        "See also: sg.help('lines')",
    ],
    "polygons": [
        "sg.Polygon",
        "sg.Node",
        "sg.Edge",
        "sg.Side",
        "sg.Polyline",
        "sg.polygon_area",
        "sg.offset_polygon",
        "sg.offset_polygon_shape",
        "sg.in_polygon",
        "sg.convex_hull",
        "sg.polygon_intersection",
        "sg.polygon_difference",
        "sg.polygon_xor",
        "sg.reg_poly_shape",
        "sg.reg_poly_points",
    ],
    "shapes": [
        "sg.Shape",
        "sg.Circle",
        "sg.Ellipse",
        "sg.Arc",
        "sg.Rectangle",
        "sg.Rectangle2",
        "sg.Line",
        "sg.Segment",
        "sg.Polyline",
        "sg.Dot",
        "sg.Dots",
        "sg.square",
        "sg.circle_shape",
        "sg.ellipse_shape",
        "sg.rect_shape",
        "sg.line_shape",
        "sg.arc_shape",
        "sg.star_shape",
        "sg.reg_poly_shape",
        "sg.reg_star_polygon",
        "sg.BoundingBox",
    ],
    "groups": [
        "sg.Group",
        "sg.Batch  (alias of Group)",
        "sg.Lace",
        "See Group methods: append, extend, translate, rotate, mirror, …",
    ],
}

d_help_topic["segments"] = list(d_help_topic["lines"])

_TOPIC_ALIASES = {
    "Shapes": "shapes",
    "Groups": "groups",
    "Points": "points",
    "Lines": "lines",
    "Segments": "segments",
    "Vertices": "vertices",
    "Edges": "edges",
    "Polygons": "polygons",
}

_HELP_ABOUT_HELP = """\
Simetri help (sg.help)
======================
sg.help(obj) documents a setting name, topic, class, function, or instance.

String keys
-----------
- defaults setting:  sg.help('line_width')  -> text from defaults_help
- topic:             sg.help('points')      -> related sg.* names
- topic list:        sg.help('topics')
- this summary:      sg.help('help')  or  sg.help(sg.help)

Topics
------
""" + ", ".join(sorted({*_TOPIC_ALIASES.values(), *d_help_topic})) + """

Callables / classes
-------------------
- sg.help(sg.Shape) or sg.help(sg.Shape(...))
  -> constructor signature, class docstring, __init__ docstring
- sg.help(sg.distance)
  -> function docstring

Missing defaults keys return an empty string.
"""


def _class_signature(cls: type) -> str:
    """Return ``ClassName(...)`` with constructor signature when available."""
    try:
        signature = inspect.signature(cls)
        return f"{cls.__name__}{signature}"
    except (TypeError, ValueError):
        return cls.__name__


def _class_help(cls: type) -> str:
    """Build help text for a class: signature, class doc, and ``__init__`` doc."""
    parts: list[str] = [_class_signature(cls)]

    class_doc = inspect.getdoc(cls)
    if class_doc:
        parts.append(class_doc)

    init = cls.__init__
    if init is not object.__init__:
        init_doc = inspect.getdoc(init)
        if init_doc:
            parts.append("__init__\n" + init_doc)

    return "\n\n".join(parts)


def _format_topic(topic: str, entries: Sequence[str]) -> str:
    """Format a topic heading and its ``sg.*`` entry list."""
    lines = [f"Topic: {topic}", ""]
    lines.extend(entries)
    lines.append("")
    lines.append("Use sg.help(name) on a callable, or sg.help('setting') for defaults.")
    return "\n".join(lines)


def _format_topics() -> str:
    """Return the sorted list of help topic names."""
    topics = sorted(
        set(d_help_topic) | set(_TOPIC_ALIASES.values()) | {"help", "topics"}
    )
    return "Available help topics:\n  " + "\n  ".join(topics)


def help(obj) -> str:
    """Return documentation text for ``obj``.

    For string keys, returns ``defaults_help[obj]`` when ``obj`` is a
    defaults setting name (empty string if missing). Reserved topic
    strings (``points``, ``lines``, ``topics``, ``help``, …) return
    topic listings from ``d_help_topic``. For classes (and instances of
    Simetri types), returns the constructor signature, class docstring,
    and ``__init__`` docstring. For functions, methods, modules, and
    other objects, returns ``inspect.getdoc(obj)``.

    Args:
        obj: Object to document, a defaults setting name, or a help topic.

    Returns:
        Documentation text, or an empty string if none is available.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.help('topics').splitlines()[0]
        'Available help topics:'
        >>> 'sg.distance' in sg.help('points')
        True
    """
    if obj is help or obj == "help":
        return _HELP_ABOUT_HELP

    if isinstance(obj, str):
        topic = _TOPIC_ALIASES[obj] if obj in _TOPIC_ALIASES else obj
        if topic == "topics":
            return _format_topics()
        if topic in d_help_topic:
            return _format_topic(topic, d_help_topic[topic])
        if obj in defaults_help:
            return defaults_help[obj]
        return ""

    if inspect.isclass(obj):
        return _class_help(obj)

    if not isinstance(obj, (bytes, int, float, bool, complex)):
        cls = type(obj)
        if (
            cls is not type
            and not inspect.isroutine(obj)
            and not inspect.ismodule(obj)
        ):
            mod = cls.__module__
            if mod != "builtins":
                return _class_help(cls)
            doc = inspect.getdoc(cls) or inspect.getdoc(obj)
            return doc if doc is not None else ""

    doc = inspect.getdoc(obj)
    return doc if doc is not None else ""
