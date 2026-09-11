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
from enum import Enum

from ..base.all_enums import WarningType
from ..coloring import colors
from ..coloring.colors import Color
from ..config.settings import VOID, defaults, defaults_help

_DISPLAY_MODULE_PREFIXES = (
    "collections.abc.",
    "simetri.base.all_enums.",
    "simetri.coloring.colors.",
    "typing.",
)
_DISPLAY_TYPE_ALIASES = {
    "simetri.group.batch.Group": "sg.Group",
    "simetri.render.canvas.Canvas": "sg.Canvas",
    "simetri.shapes.shape.Shape": "sg.Shape",
}
_MAX_INLINE_SIGNATURE_LENGTH = 88
_COLOR_NAME_BY_ID = {
    id(value): name
    for name, value in colors.__dict__.items()
    if isinstance(value, Color)
}

_WARNING_SUBGROUPS = {
    value: name
    for name, value in vars(WarningType).items()
    if isinstance(value, type) and issubclass(value, Enum)
}


def _warning_type_path(obj) -> str | None:
    """Return ``WarningType…`` path for a subgroup class or leaf member."""
    if obj is WarningType:
        return "WarningType"
    if inspect.isclass(obj) and obj in _WARNING_SUBGROUPS:
        return f"WarningType.{_WARNING_SUBGROUPS[obj]}"
    if isinstance(obj, Enum) and type(obj) in _WARNING_SUBGROUPS:
        group_name = _WARNING_SUBGROUPS[type(obj)]
        return f"WarningType.{group_name}.{obj.name}"
    return None


def _warning_type_help(obj) -> str:
    """Return help text for ``WarningType``, a subgroup, or a leaf."""
    if obj is WarningType:
        group_lines = [
            f"  WarningType.{name}" for name in sorted(_WARNING_SUBGROUPS.values())
        ]
        base = inspect.getdoc(WarningType) or ""
        return f"{base}\n\nSubgroups:\n" + "\n".join(group_lines)

    if inspect.isclass(obj) and obj in _WARNING_SUBGROUPS:
        base = inspect.getdoc(obj) or ""
        member_lines = []
        for member in obj:
            member_doc = inspect.getdoc(member) or ""
            member_lines.append(f"  {member.name}: {member_doc}")
        return f"{base}\n\nMembers:\n" + "\n".join(member_lines)

    doc = inspect.getdoc(obj)
    return doc if doc is not None else ""


# Topic name -> list of public ``sg.*`` names (and brief notes as plain lines).
d_help_topic: dict[str, list[str]] = {
    "angles": [
        "angle_abs_tol",
        "angle_rel_tol",
        "angle_tol",
        # marker_angle
        # marker_phase
        "pattern_angle",
        "sg.add_angles",
        "sg.angle",
        "sg.angle_between_lines3",
        "sg.angle_between_two_lines",
        "sg.angled_line",
        "sg.angled_vector",
        "sg.BoundingBox.angle_point",
        "sg.cartesian_to_polar",
        "sg.central_to_parametric_angle",
        "sg.close_angles",
        "sg.degrees",
        "sg.ellipse_point",
        "sg.get_quadrant",
        "sg.get_quadrant_from_deg_angle",
        "sg.Group.closest_angle_differences",
        "sg.inclination_angle",
        "sg.line_angle",
        "sg.line_by_point_angle_length",
        "sg.parametric_to_central_angle",
        "sg.polar_to_cartesian",
        "sg.polygon_internal_angles",
        "sg.positive_angle",
        "sg.r_polar",
        "sg.radians",
        "sg.rel_polar",
        "sg.triangle_angles_from_sides",
        "sg.turning_function",
        "sg.Vector.angle",
        "sg.Vector.angle_between",
        "sg.v_angle",
        "sg.v_angle_between",
    ],
    "boolean_ops": [
        "sg.polygon_difference",
        "sg.polygon_intersection",
        "sg.polygon_xor",
        "sg.clip",
        "See also: sg.help('clipping'), sg.help('polygons')",
    ],
    "canvas": [
        "Canvas.draw",
        "Canvas.save",
        "sg.Canvas",
        "sg.set_defaults",
        "sg.set_svg_defaults",
        "sg.set_tikz_defaults",
    ],
    "clipping": [
        "sg.Mask",
        "sg.clip",
        "sg.clip_line_to_rect",
        "sg.clip_mask",
        "See also: sg.help('boolean_ops')",
    ],
    "colors": [
        "sg.Color",
        "sg.LinearGradient",
        "sg.RadialGradient",
        "sg.black",
        "sg.blend",
        "sg.blue",
        "sg.random_color",
        "sg.red",
        "sg.white",
        "See also: sg.help('effects')",
    ],
    "edges": [
        "sg.Edge",
        "sg.Segment",
        "sg.all_segments",
        "sg.equal_edges",
        "sg.intersect",
        "sg.offset_line",
        "See also: sg.help('lines')",
    ],
    "effects": [
        "sg.Gradient",
        "sg.LinearGradient",
        "sg.Mask",
        "sg.RadialGradient",
        "sg.Stop",
        "See also: sg.help('colors')",
    ],
    "export": [
        "sg.Canvas",
        "Canvas.draw",
        "Canvas.save",
        "sg.get_svg_code",
        "sg.pdf_to_svg",
        "sg.set_svg_defaults",
        "sg.set_tikz_defaults",
    ],
    "grids": [
        "sg.Grid",
        "sg.axis",
        "See also: sg.help('patterns'), sg.help('canvas')",
    ],
    "groups": [
        "sg.Batch  (alias of Group)",
        "sg.Group",
        "sg.Lace",
        "See Group methods: append, extend, translate, rotate, mirror, …",
    ],
    "images": [
        "sg.Image",
        "sg.open_img",
    ],
    "lines": [
        "sg.Line",
        "sg.Segment",
        "sg.all_intersections",
        "sg.angle_between_lines3",
        "sg.angle_between_two_lines",
        "sg.axis",
        "sg.clip_line_to_rect",
        "sg.equal_edges",
        "sg.extended_line",
        "sg.fillet_corners",
        "sg.intersect",
        "sg.line_angle",
        "sg.line_by_point_angle_length",
        "sg.line_shape",
        "sg.offset_line",
        "sg.round_segment",
        "sg.stitch",
    ],
    "patterns": [
        "sg.Lace",
        "sg.Pattern",
        "sg.Star",
        "sg.frieze",
        "sg.reg_star_polygon",
        "sg.rosette",
        "See also: sg.help('grids')",
    ],
    "points": [
        "sg.cart_to_tri",
        "sg.close_points_square",
        "sg.connected_pairs",
        "sg.distance",
        "sg.distance_square",
        "sg.fix_degen_points",
        "sg.homogenize",
        "sg.left",
        "sg.lerp_point",
        "sg.midpoint",
        "sg.offset_point",
        "sg.offset_point_from_start",
        "sg.on_segment",
        "sg.point_on_line_segment",
        "sg.r_polar",
        "sg.rel_coord",
        "sg.rel_polar",
        "sg.round_point",
        "sg.tri_to_cart",
        # Additional helpers live in simetri.geom.points.point_utils
        # (equal_points, project_point_on_line, remove_duplicate_points, …).
    ],
    "polygons": [
        "sg.Edge",
        "sg.Node",
        "sg.Polygon",
        "sg.Polyline",
        "sg.Side",
        "sg.convex_hull",
        "sg.in_polygon",
        "sg.offset_polygon",
        "sg.offset_polygon_shape",
        "sg.polygon_area",
        "sg.polygon_difference",
        "sg.polygon_intersection",
        "sg.polygon_xor",
        "sg.reg_poly_points",
        "sg.reg_poly_shape",
    ],
    "segments": [],  # filled as alias of lines below
    "shapes": [
        "sg.Arc",
        "sg.BoundingBox",
        "sg.Circle",
        "sg.Dot",
        "sg.Dots",
        "sg.Ellipse",
        "sg.Line",
        "sg.Polyline",
        "sg.Rectangle",
        "sg.Rectangle2",
        "sg.Segment",
        "sg.Shape",
        "sg.arc_shape",
        "sg.circle_shape",
        "sg.ellipse_shape",
        "sg.line_shape",
        "sg.rect_shape",
        "sg.reg_poly_shape",
        "sg.reg_star_polygon",
        "sg.square",
        "sg.star_shape",
    ],
    "tags": [
        "sg.Arrow",
        "sg.ArrowHead",
        "sg.Tag",
        "sg.TagFrame",
    ],
    "text": [
        "sg.Tag",
        "sg.TagFrame",
        "sg.Arrow",
        "sg.ArrowHead",
        "sg.get_text_dimensions",
        "sg.get_text_size",
    ],
    "tolerances": [
        "sg.check_angle_tol",
        "sg.check_dist_tol",
        "sg.equal_angles",
        "sg.equal_points",
        "sg.distance",
        "See also: sg.help('angle_tol'), sg.help('area_tol'), sg.help('dist_tol')",
    ],
    "transforms": [
        "sg.glide_matrix",
        "sg.mirror",
        "sg.mirror_matrix",
        "sg.rotate",
        "sg.rotation_matrix",
        "sg.scale",
        "sg.scale_matrix",
        "sg.shear_matrix",
        "sg.translate",
        "sg.translation_matrix",
    ],
    "vertices": [
        "Shape.vertices / Shape.primary_points",
        "sg.close_points_square",
        "sg.fix_degen_points",
        "sg.homogenize",
        "sg.round_point",
        "See also: sg.help('points'), sg.help('shapes')",
    ],
}

d_help_topic["segments"] = list(d_help_topic["lines"])

_TOPIC_ALIASES = {
    "BooleanOps": "boolean_ops",
    "Canvas": "canvas",
    "Clipping": "clipping",
    "Colors": "colors",
    "Edges": "edges",
    "Effects": "effects",
    "Export": "export",
    "Grids": "grids",
    "Groups": "groups",
    "Images": "images",
    "Lines": "lines",
    "Patterns": "patterns",
    "Points": "points",
    "Polygons": "polygons",
    "Segments": "segments",
    "Shapes": "shapes",
    "Tags": "tags",
    "Text": "text",
    "Tolerances": "tolerances",
    "Transforms": "transforms",
    "Vertices": "vertices",
}

_HELP_ABOUT_HELP = (
    """\
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
"""
    + ", ".join(sorted({*_TOPIC_ALIASES.values(), *d_help_topic}))
    + """

Callables / classes
-------------------
- sg.help(sg.Shape) or sg.help(sg.Shape(...))
  -> constructor signature, class docstring, __init__ docstring
- sg.help(sg.distance)
  -> function docstring

Missing defaults keys return an empty string.
"""
)


def _class_signature(cls: type) -> str:
    """Return ``ClassName(...)`` with constructor signature when available."""
    try:
        signature = inspect.signature(cls)
        default_overrides = {}
        if (
            cls.__module__ == "simetri.render.canvas"
            and cls.__name__ == "Canvas"
        ):
            for parameter in signature.parameters.values():
                if (
                    parameter.default is None
                    and parameter.name in defaults.defaults
                    and defaults[parameter.name] is not VOID
                    and parameter.name != "page_size"
                ):
                    default_overrides[parameter.name] = defaults[parameter.name]
        return _format_signature(
            cls.__name__,
            signature,
            default_overrides=default_overrides,
        )
    except (TypeError, ValueError):
        return cls.__name__


def _format_annotation(annotation: object) -> str:
    """Return a compact display string for a signature annotation."""
    if annotation is inspect.Signature.empty:
        return ""

    if isinstance(annotation, str):
        text = annotation
    elif annotation is None:
        text = "None"
    else:
        text = repr(annotation)
        if text.startswith("<class '") and text.endswith("'>"):
            text = text[8:-2]

    for qualified_name, alias in _DISPLAY_TYPE_ALIASES.items():
        text = text.replace(qualified_name, alias)

    for prefix in _DISPLAY_MODULE_PREFIXES:
        text = text.replace(prefix, "")

    return text


def _format_default(value: object) -> str:
    """Return a compact display string for a default value."""
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"

    if isinstance(value, Color):
        color_id = id(value)
        if color_id in _COLOR_NAME_BY_ID:
            return f"sg.{_COLOR_NAME_BY_ID[color_id]}"

    return repr(value)


def _format_parameter(
    parameter: inspect.Parameter,
    default_override: object = inspect.Signature.empty,
) -> str:
    """Return one formatted parameter for display."""
    name = parameter.name
    if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
        name = "*" + name
    elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
        name = "**" + name

    text = name
    if parameter.annotation is not inspect.Signature.empty:
        text += f": {_format_annotation(parameter.annotation)}"
    if default_override is not inspect.Signature.empty:
        text += f" = {_format_default(default_override)}"
    elif parameter.default is not inspect.Signature.empty:
        text += f" = {_format_default(parameter.default)}"

    return text


def _format_signature(
    name: str,
    signature: inspect.Signature,
    default_overrides: dict[str, object] | None = None,
) -> str:
    """Return a readable display form for a callable signature."""
    parts: list[str] = []
    has_var_positional = False
    inserted_kw_separator = False
    if default_overrides is None:
        default_overrides = {}
    for parameter in signature.parameters.values():
        if (
            parameter.kind == inspect.Parameter.KEYWORD_ONLY
            and not has_var_positional
            and not inserted_kw_separator
        ):
            parts.append("*")
            inserted_kw_separator = True

        parts.append(
            _format_parameter(
                parameter,
                default_override=default_overrides.get(
                    parameter.name, inspect.Signature.empty
                ),
            )
        )
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True

    text = f"{name}({', '.join(parts)})"
    if signature.return_annotation is not inspect.Signature.empty:
        text += f" -> {_format_annotation(signature.return_annotation)}"

    if len(text) <= _MAX_INLINE_SIGNATURE_LENGTH:
        return text

    lines = [f"{name}("]
    for part in parts:
        lines.append(f"    {part},")

    closing = ")"
    if signature.return_annotation is not inspect.Signature.empty:
        closing += f" -> {_format_annotation(signature.return_annotation)}"
    lines.append(closing)

    return "\n".join(lines)


def _resolved_default_lines(signature: inspect.Signature) -> list[str]:
    """Return displayed resolved defaults for parameters whose default is None."""
    return [
        f"{parameter.name} = {_format_default(defaults[parameter.name])}"
        for parameter in signature.parameters.values()
        if parameter.default is None
        and parameter.name in defaults.defaults
        and defaults[parameter.name] is not VOID
    ]


def _canvas_draw_help(obj) -> str:
    """Return the help text for ``Canvas.draw``."""
    parts: list[str] = []

    signature = inspect.signature(obj)
    parts.append(_format_signature("Canvas.draw", signature))

    doc = inspect.getdoc(obj)
    if doc:
        parts.append(doc)

    parts.append(
        "How canvas.draw works\n"
        "style kwargs override the drawn object's corresponding style attributes\n"
        "for Shape and Group, pos / angle / scale are applied to the drawn snapshot\n"
        "pos, rotocenter, and about use canvas units (points); angles use radians\n"
        "canvas.translate/rotate/scale change the canvas coordinate system for later drawing"
    )

    return "\n\n".join(parts)


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

    if cls.__module__ == "simetri.shapes.shape" and cls.__name__ == "Shape":
        signature = inspect.signature(cls)
        resolved_default_lines = _resolved_default_lines(signature)
        if resolved_default_lines:
            parts.append(
                "Resolved defaults used when a constructor argument is None\n"
                + "\n".join(resolved_default_lines)
            )

    if cls.__module__ == "simetri.render.canvas" and cls.__name__ == "Canvas":
        signature = inspect.signature(cls)
        resolved_default_lines = [
            line
            for line in _resolved_default_lines(signature)
            if not line.startswith("page_size = ")
        ]
        canvas_default_lines = resolved_default_lines[:]
        canvas_default_lines.append(
            "page_size = calculated automatically when omitted"
        )
        canvas_default_lines.append(
            "canvas.draw(...) snapshots objects into sketch objects on the active page"
        )
        canvas_default_lines.append(
            "canvas.draw(..., style=value) overrides the drawn object's corresponding style attributes"
        )
        canvas_default_lines.append(
            "for Shape and Group, canvas.draw(..., pos=..., angle=...) moves or rotates the drawn snapshot"
        )
        canvas_default_lines.append(
            "canvas.translate/rotate/scale change the canvas coordinate system for later drawing"
        )
        canvas_default_lines.append("see also: sg.doc(sg.Canvas.draw)")
        parts.append(
            "Constructor behavior and resolved defaults\n"
            + "\n".join(canvas_default_lines)
        )

    if cls.__module__ == "simetri.group.batch" and cls.__name__ == "Group":
        parts.append(
            "Constructor behavior\n"
            "elements = [] when omitted\n"
            "modifiers = [] when omitted"
        )

    return "\n\n".join(parts)


def _format_topic(topic: str, entries: Sequence[str]) -> str:
    """Format a topic heading and its ``sg.*`` entry list."""
    lines = [f"Topic: {topic}", ""]
    lines.extend(entries)
    lines.append("")
    lines.append(
        "Use sg.help(name) on a callable, or sg.help('setting') for defaults."
    )
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

    # StrEnum members are also ``str``; resolve WarningType paths first.
    warning_path = _warning_type_path(obj)
    if warning_path is not None:
        return _warning_type_help(obj)

    if isinstance(obj, Enum):
        doc = inspect.getdoc(obj)
        return doc if doc is not None else ""

    if isinstance(obj, str):
        topic = obj
        if obj in _TOPIC_ALIASES:
            topic = _TOPIC_ALIASES[obj]
        if topic == "topics":
            return _format_topics()
        if topic in d_help_topic:
            return _format_topic(topic, d_help_topic[topic])
        if obj in defaults_help:
            return defaults_help[obj]
        return ""

    if inspect.isclass(obj):
        return _class_help(obj)

    if (
        getattr(obj, "__module__", None) == "simetri.render.canvas"
        and getattr(obj, "__qualname__", None) == "Canvas.draw"
    ):
        return _canvas_draw_help(obj)

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


def _doc_title(obj) -> str:
    """Return the display title used by ``sg.doc`` for ``obj``."""
    warning_path = _warning_type_path(obj)
    if warning_path is not None:
        return f"sg.{warning_path}"

    if isinstance(obj, str):
        return obj

    if inspect.ismodule(obj):
        return obj.__name__

    if inspect.isclass(obj):
        if obj.__module__.startswith("simetri."):
            return f"sg.{obj.__qualname__}"
        return obj.__qualname__

    if inspect.isroutine(obj):
        if obj.__module__.startswith("simetri."):
            return f"sg.{obj.__qualname__}"
        return obj.__qualname__

    if isinstance(obj, Enum):
        return f"{type(obj).__qualname__}.{obj.name}"

    if not isinstance(obj, (bytes, int, float, bool, complex)):
        cls = type(obj)
        if cls is not type and cls.__module__ != "builtins":
            return f"sg.{cls.__qualname__}"
        return cls.__qualname__

    return type(obj).__qualname__


def doc(obj) -> None:
    """Print documentation text for ``obj``.

    Args:
        obj: Object to document, a defaults setting name, or a help topic.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.doc('topics')
    """
    title = _doc_title(obj)
    text = help(obj)
    if text:
        print(f"{title}\n{'=' * len(title)}\n{text}")
    else:
        print(f"{title}\n{'=' * len(title)}")


# These should be in angle_doc
#  "angle_abs_tol",
# "angle_rel_tol",
# "angle_tol",
# # marker_angle
# # marker_phase
# "pattern_angle",
# "sg.angle",
# "sg.angle_between_lines3",
# "sg.angle_between_two_lines",
# "sg.angled_line",
# "sg.angled_vector",
# "sg.Arc.rot_angle",
# "sg.Arc.start_angle",
# "sg.Arc.span_angle",
# "sg.BoundingBox.angle_point",
# "sg.Canvas.angle",
# "sg.Canvas.orientation",
# "sg.cartesian_to_polar",
# "sg.central_to_parametric_angle",
# "sg.close_angles",
# "sg.degrees",
# "sg.ellipse_point",
# "sg.feSpotLight.limiting_cone_angle",
# "sg.get_quadrant",
# "sg.get_quadrant_from_deg_angle",
# "sg.Group.closest_angle_differences",
# "sg.inclination_angle",
# "sg.line_angle",
# "sg.line_by_point_angle_length",
# "sg.parametric_to_central_angle",
# "sg.Path.angle",
# "sg.polar_to_cartesian",
# "sg.polygon_internal_angles",
# "sg.positive_angle",
# "sg.r_polar",
# "sg.radians",
# "sg.rel_polar",
# "sg.Shape.angle",
# "sg.Shape.line_dash_phase",
# "sg.Shape.orientation",
# "sg.SineWave.phase_angle",
# "sg.SineWave.rot_angle",
# "sg.triangle_angles_from_sides",
# "sg.turning_function",
# "shade_axis_angle",
# "tile_angle",
# "turn_angle_digits",
# "sg.Turtle.angle",
# "sg.Vector.angle",
# "sg.Vector.angle_between",
# "sg.v_angle",
# "sg.v_angle_between",
