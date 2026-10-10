"""Interactive and topic-based help for Simetri (``sg.help``).

``sg.help(obj)`` accepts a string key or a callable/class/instance:

- String keys look up ``defaults_help[obj]`` (empty string if missing),
  except for reserved topic names described below. Unknown topic strings
  return similar topic names.
- Classes (and instances of Simetri types) return the constructor
  signature, class docstring, and ``__init__`` docstring.
- Simetri functions and methods return a signature (with resolved
  ``defaults`` where applicable), accepted ``**kwargs`` names when
  known, then the docstring.
- Other modules and objects return ``inspect.getdoc(obj)``.

``d_help_topic`` maps topic names to curated notes and highlight ``sg.*``
names. ``help_topics_generated.py`` (from
``python -m simetri.helpers.compile_help_topics``) adds every matching
public export per topic. ``help_topic_supplements.py`` adds semantic links
(dict / many-to-many graph / aliases). ``sg.help('help')`` loads the help-utilities
guide. ``sg.help(sg.help)`` summarizes how help lookup works.
``sg.help('topics')`` lists available topics.

Examples:
"""

from __future__ import annotations

import inspect
import re
import sys
import unicodedata
from collections.abc import Callable, Sequence
from enum import Enum
from functools import lru_cache
from pathlib import Path

from rapidfuzz.distance import DamerauLevenshtein

from ..base.all_enums import WarningType
from ..coloring import colors
from ..coloring.colors import Color
from ..config.settings import VOID, defaults, defaults_help, user_defaults
from ..config.user_config_help import (
    USER_CONFIG_TOPIC_PREEMPT,
    user_config_help,
    user_config_help_keys,
    user_config_sections,
)

try:
    from .help_topics_generated import (
        COMPILED_DOC_PLACEHOLDERS_BY_TOPIC,
        COMPILED_SG_BY_TOPIC,
        COMPILED_TOPIC_ALIASES,
    )
except ImportError:
    COMPILED_SG_BY_TOPIC: dict[str, tuple[str, ...]] = {}
    COMPILED_TOPIC_ALIASES: dict[str, str] = {}
    COMPILED_DOC_PLACEHOLDERS_BY_TOPIC: dict[str, tuple[str, ...]] = {}

from .help_doc_placeholders import format_doc_placeholder_for_help
from .help_topic_supplements import (
    RELATED_TOPICS,
    SG_NAME_TO_TOPICS,
    TOPIC_ALIASES as SUPPLEMENT_TOPIC_ALIASES,
    TOPIC_DOC_PLACEHOLDERS,
    TOPIC_SG_ENTRIES,
)

_TOPIC_GUIDES_DIR = Path(__file__).resolve().parents[1] / "topic_guides"

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


def _module_name(obj: object) -> str:
    """Return ``obj.__module__`` when it is a string, else ``""``."""
    module = getattr(obj, "__module__", None)
    return module if isinstance(module, str) else ""


def normalize(word: str) -> str:
    """Normalize a string for fuzzy help lookup (NFKC, casefold, strip).

    Examples:
        >>> from simetri.helpers.help_utils import normalize
        >>> normalize('  Shape  ')
        'shape'
    """
    return unicodedata.normalize("NFKC", word).casefold().strip()


def _user_config_help_for_string(query: str) -> str | None:
    """Resolve config help before topic aliases when appropriate."""
    if query in user_config_sections():
        if query in USER_CONFIG_TOPIC_PREEMPT:
            return user_config_help(query)
        if query in ("tex", "viewer"):
            return None
        if query not in _TOPIC_ALIASES and query not in all_help_topic_keys():
            if query not in _topic_guide_aliases():
                return user_config_help(query)
    if "." in query:
        section, _, key = query.partition(".")
        if key and section in user_config_sections():
            return user_config_help(query)
    if (
        query in _TOPIC_ALIASES
        or query in all_help_topic_keys()
        or query in _topic_guide_aliases()
        or query in _topic_help_hubs()
    ):
        return None
    return user_config_help(query)


def _resolve_help_suggestion_limit(limit: int | None) -> int:
    """Return the active similar-name suggestion limit."""
    if limit is None:
        return user_defaults["help_suggestion_limit"]
    return limit


def find_similar(
    query: str,
    words: Sequence[str],
    threshold: float = 0.75,
    limit: int | None = None,
) -> list[tuple[str, float]]:
    """Return words similar to ``query`` by Damerau–Levenshtein similarity.

    Args:
        query: Search string (normalized before comparison).
        words: Candidate names.
        threshold: Minimum normalized similarity in ``[0, 1]``. Defaults to 0.75.
        limit: Maximum matches; ``None`` uses ``defaults['help_suggestion_limit']``.

    Returns:
        list[tuple[str, float]]: ``(word, score)`` pairs, highest score first.

    Examples:
        >>> from simetri.helpers.help_utils import find_similar
        >>> find_similar('shpe', ['shape', 'group'], limit=2)[0][0]
        'shape'
    """
    limit = _resolve_help_suggestion_limit(limit)
    query_normalized = normalize(query)
    matches = []

    for word in words:
        score = DamerauLevenshtein.normalized_similarity(
            query_normalized, normalize(word)
        )

        if score >= threshold:
            matches.append((word, score))

    return sorted(matches, key=lambda item: item[1], reverse=True)[:limit]


def _warning_type_path(obj: object) -> str | None:
    """Return ``WarningType…`` path for a subgroup class or leaf member."""
    if obj is WarningType:
        return "WarningType"
    if inspect.isclass(obj) and obj in _WARNING_SUBGROUPS:
        return f"WarningType.{_WARNING_SUBGROUPS[obj]}"
    if isinstance(obj, Enum) and type(obj) in _WARNING_SUBGROUPS:
        group_name = _WARNING_SUBGROUPS[type(obj)]
        return f"WarningType.{group_name}.{obj.name}"
    return None


def _warning_type_help(obj: object) -> str:
    """Return help text for ``WarningType``, a subgroup, or a leaf."""
    if obj is WarningType:
        group_lines = [
            f"  WarningType.{name}"
            for name in sorted(_WARNING_SUBGROUPS.values())
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
        "abs_tol",
        "rel_tol",
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
    "bounding_box_doc": [
        "sg.BoundingBox",
        "sg.bounding_box",
        "Shape.b_box",
        "Group.b_box",
        "sg.Reference",
        "sg.DynRef",
        (
            "See also: sg.help('shapes_doc'), sg.help('groups_doc'), "
            "sg.help('transforms_doc'), sg.help('dynamic_references_doc')"
        ),
    ],
    "building_help": [
        "sg.help",
        "sg.doc",
        (
            "See also: sg.help('help_doc'), sg.help('testing_doc'), "
            "sg.help('topics')"
        ),
    ],
    "canvas": [
        "Canvas.draw",
        "Canvas.draw_lace",
        "Canvas.save",
        "sg.Canvas",
        "sg.set_defaults",
        "sg.set_svg_defaults",
        "sg.set_tikz_defaults",
        (
            "See also: sg.help('canvas_doc'), sg.help('user_settings'), "
            "sg.help('tex_compiler'), sg.help('viewer')"
        ),
    ],
    "canvas_context_managers": [
        "Canvas.style",
        "Canvas.push_style",
        "Canvas.pop_style",
        "Canvas.translate",
        "Canvas.rotate",
        "Canvas.scale",
        "Canvas.push_matrix",
        "Canvas.pop_matrix",
        "Canvas.reset_transform",
        "Canvas.reset_style",
        "Canvas.reset_line_style",
        "Canvas.reset_fill_style",
        "Shape.reset_style",
        "Shape.reset_line_style",
        "Shape.reset_fill_style",
        "sg.Style",
        "sg.user_styles",
        "sg.save_user_style",
        "See also: sg.help('canvas_doc'), sg.help('user_settings')",
    ],
    "canvas_doc": [
        "sg.Canvas",
        "Canvas.draw",
        "Canvas.draw_lace",
        "Canvas.save",
        "Canvas.display",
        "Canvas.insert_svg",
        "Canvas.insert_tex",
        "Canvas.reset",
        (
            "See also: sg.help('user_settings'), sg.help('script_sharing'), "
            "sg.help('tex_compiler'), "
            "sg.help('image_converters'), sg.help('viewer')"
        ),
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
    "composite_transformations": [
        "sg.Transform",
        "sg.Transformation",
        "sg.translation_matrix",
        "sg.rotation_matrix",
        "sg.mirror_matrix",
        "sg.glide_matrix",
        "sg.scale_matrix",
        "sg.shear_matrix",
        (
            "Shape.transform: a 3×3 matrix (product of helpers such as "
            "translation_matrix @ rotation_matrix), or a Transformation of "
            "Transform steps. dyn_ref=True rebuilds every step each "
            "repetition against the same KERNEL / PATTERN / ACTIVE."
        ),
        "See also: sg.help('dynamic_references_doc'), sg.help('transforms')",
    ],
    "dimensioning_doc": [
        "sg.AlignedDimension",
        "sg.Dimension",
        "sg.AnnotationArrow",
        "sg.RadialDimension",
        "sg.AngularDimension",
        "sg.defaults['dim_color']",
        "sg.defaults['dim_line_width']",
        "sg.defaults['gap']",
        "sg.defaults['overshoot']",
        "sg.defaults['text_offset']",
        "sg.defaults['landing_length']",
        "sg.defaults['n_dim_digits']",
        "sg.defaults['stub_length']",
        "sg.defaults['ext_angle']",
        "sg.defaults['gap_angle']",
        "See also: sg.help('canvas_doc'), sg.help('tags')",
    ],
    "drawing_laces_doc": [
        "sg.Lace",
        "sg.Canvas.draw_lace",
        "Canvas.draw_fragments",
        "Canvas.draw_plaits",
        "Canvas.draw_lace_with_fillets",
        "sg.FragmentColoring",
        "sg.PlaitStyle",
        "sg.Star",
        "sg.stars.Star",
        "sg.rosette",
        "sg.defaults['fragment_coloring']",
        "sg.defaults['lace_plait_style']",
        "sg.defaults['shade_plaits']",
        "sg.defaults['fillet_radii']",
        (
            "See also: sg.help('canvas_doc'), sg.help('groups_doc'), "
            "sg.help('shapes_doc'), sg.help('patterns')"
        ),
    ],
    "dynamic_references_doc": [
        "sg.DynRef",
        "sg.Reference",
        "sg.ReferenceTarget",
        "sg.resolve_dyn_ref",
        "sg.Transform",
        "sg.Transformation",
        (
            "Shape.translate: two lengths (dx, dy), or one vector "
            "translate((x, y)) / translate(edge). EDGE in a length slot "
            "is ‖edge‖; one EDGE argument is the edge vector."
        ),
        (
            "See also: sg.help('transforms'), "
            "sg.help('composite_transformations'), "
            "sg.doc(sg.Shape.translate)"
        ),
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
        "sg.extract_glyph_path",
        "sg.get_svg_code",
        "sg.pdf_to_svg",
        "sg.set_svg_defaults",
        "sg.set_tikz_defaults",
        (
            "See also: sg.help('image_converters'), sg.help('tex_compiler'), "
            "sg.help('viewer')"
        ),
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
    "groups_doc": [
        "sg.Group",
        "sg.Batch",
        "sg.Path2D",
        "sg.Lace",
        "sg.Pattern",
        "sg.Star",
        "sg.Dots",
        (
            "See also: sg.help('shapes_doc'), sg.help('canvas_doc'), "
            "sg.help('style_definitions'), sg.help('boolean_ops'), "
            "sg.help('tag_objects'), sg.help('drawing_laces_doc')"
        ),
    ],
    "help_doc": [
        "sg.help",
        "sg.doc",
        "sg.help('topics')",
        "sg.Canvas.draw",
        "Canvas.help_lines",
        "Canvas.draw_all_segments",
        (
            "See also: sg.help('canvas_doc'), sg.help('warnings'), "
            "sg.help('user_settings'), sg.help('shapes_doc')"
        ),
    ],
    "image_converters": [
        "sg.Canvas.save",
        "sg.Canvas.capture",
        "sg.Image",
        "sg.open_img",
        "sg.user_config_path",
        "sg.set_user_settings_path",
        "sg.apply_user_config",
        (
            "See also: sg.help('canvas_doc'), sg.help('images'), "
            "sg.help('user_settings'), sg.help('tex_compiler')"
        ),
    ],
    "images": [
        "sg.Image",
        "sg.open_img",
        "See also: sg.help('image_converters'), sg.help('canvas_doc')",
    ],
    "images_doc": [
        "sg.Image",
        "sg.open_img",
        "Canvas.draw_on_image",
        "Canvas.save_image",
        "See also: sg.help('image_converters'), sg.help('canvas_doc')",
    ],
    "latex_engine": [
        "sg.Canvas.save",
        "sg.Compiler",
        "sg.defaults['latex_compiler']",
        "sg.user_config_path",
        (
            "See also: sg.help('tex_compiler'), sg.help('user_settings'), "
            "sg.help('viewer')"
        ),
    ],
    "lattices_doc": [
        "sg.Lattice",
        "sg.Isometry",
        "sg.LatType",
        "sg.LatRef",
        "sg.lattice_p1",
        "sg.lattice_p2",
        "sg.lattice_pm",
        "sg.lattice_pg",
        "sg.lattice_cm",
        "sg.lattice_pmm",
        "sg.lattice_pmg",
        "sg.lattice_pgg",
        "sg.lattice_cmm",
        "sg.lattice_p4",
        "sg.lattice_p4m",
        "sg.lattice_p4g",
        "sg.lattice_p3",
        "sg.lattice_p3m1",
        "sg.lattice_p31m",
        "sg.lattice_p6",
        "sg.lattice_p6m",
        "sg.get_unit",
        "sg.draw_unit",
        (
            "See also: sg.help('shapes_doc'), sg.help('groups_doc'), "
            "sg.help('transforms_doc'), sg.help('patterns')"
        ),
    ],
    "merge_shapes_doc": [
        "sg.Group.merge_shapes",
        "sg.Group.merge_collinears",
        (
            "See also: sg.help('groups_doc'), sg.help('prune_shapes_doc'), "
            "sg.help('polygons')"
        ),
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
        "See also: sg.help('grids'), sg.help('lattices_doc'), "
        "sg.help('drawing_laces_doc')",
    ],
    "path_objects_doc": [
        "sg.Path2D",
        "sg.LinPath",
        "sg.Operation",
        "sg.PathOps",
        "sg.path_code",
        "sg.shape_to_path2d",
        "sg.svg_path_to_path2d",
        "sg.path2d_to_svg_path",
        "sg.path2d_svg",
        (
            "See also: sg.help('shapes_doc'), sg.help('groups_doc'), "
            "sg.help('canvas_doc'), sg.help('style_definitions')"
        ),
    ],
    "points": [
        "sg.cart_to_tri",
        "sg.close_points_square",
        "sg.connected_pairs",
        "sg.distance",
        "sg.distance_square",
        "sg.extend",
        "sg.fix_degen_points",
        "sg.homogenize",
        "sg.left",
        "sg.lerp_point",
        "sg.midpoint",
        "sg.offset_point",
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
        "sg.inflate",
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
        "sg.Square",
        "sg.arc_shape",
        "sg.circle_shape",
        "sg.ellipse_shape",
        "sg.inflate",
        "sg.line_shape",
        "sg.rect_shape",
        "sg.reg_poly_shape",
        "sg.reg_star_polygon",
        "sg.square",
        "sg.star_shape",
    ],
    "shapes_doc": [
        "sg.Shape",
        "sg.square",
        "sg.Square",
        "sg.Rectangle",
        "sg.Circle",
        "sg.Line",
        "sg.Segment",
        "sg.clip",
        "sg.polygon_difference",
        "sg.inflate",
        (
            "See also: sg.help('canvas_doc'), sg.help('style_definitions'), "
            "sg.help('groups_doc'), sg.help('boolean_ops'), "
            "sg.help('tag_objects')"
        ),
    ],
    "print_doc": [
        "sg.format_data",
        "sg.p_print",
        "sg.print_options",
        "sg.pretty_print_coords",
        "sg.register_format_handler",
        "sg.round_point",
        "sg.round_points",
        (
            "See also: sg.help('vertices'), sg.help('path_objects_doc')"
        ),
    ],
    "random_seeds": [
        "sg.random_angle",
        "sg.random_circle",
        "sg.random_circles",
        "sg.random_ellipse",
        "sg.random_ellipses",
        "sg.random_point",
        "sg.random_points",
        "sg.random_segment",
        "sg.random_segments",
        "sg.random_rectangle",
        "sg.random_rectangles",
        "sg.random_triangle",
        "sg.random_triangles",
        "sg.random_polygon",
        "sg.random_polygons",
        "sg.random_color",
        "sg.random_palette",
        "sg.random_swatch",
        "sg.random_characters",
        "See also: sg.help('script_sharing')",
    ],
    "script_sharing": [
        "sg.generate_shared_toml",
        "sg.save_as",
        "sg.use_script_header",
        "sg.use_settings",
        "sg.check_version",
        (
            "See also: sg.help('user_settings'), sg.help('random_seeds'), "
            "sg.help('tex_compiler')"
        ),
    ],
    "style_definitions": [
        "Canvas.style",
        "sg.Style",
        "Shape.style",
        "Shape.line_style",
        "Shape.fill_style",
        "Group.set_style",
        "sg.user_styles",
        "sg.save_user_style",
        "Shape.copy_style",
        "See also: sg.help('canvas_doc'), sg.help('user_settings')",
    ],
    "tag_objects": [
        "sg.Tag",
        "sg.TagFrame",
        "sg.Canvas.text",
        "sg.Canvas.draw",
        "sg.Anchor",
        "sg.Align",
        "sg.FrameShape",
        "sg.FontFamily",
        "sg.FontSize",
        (
            "See also: sg.help('canvas_doc'), sg.help('text'), "
            "sg.help('dimensioning_doc')"
        ),
    ],
    "tables": [
        "simetri.extensions.table.Table",
        "simetri.extensions.table.Range",
        "Table.display",
        "Table.build_rich_table",
        "Canvas.draw",
        (
            "See also: sg.help('tables_doc'), sg.help('text_doc'), "
            "sg.help('canvas_doc'), sg.help('tag_objects')"
        ),
    ],
    "tables_doc": [
        "simetri.extensions.table.Table",
        "simetri.extensions.table.Range",
        "Table.columns",
        "Table.rows",
        "Table.cells",
        "Table.range",
        "Range.width",
        "Range.height",
        "Range.size",
        "Range.set_format",
        "Canvas.draw",
        (
            "See also: sg.help('text_doc'), sg.help('canvas_doc'), "
            "sg.help('tag_objects'), sg.help('bounding_box_doc')"
        ),
    ],
    "text": [
        "sg.Tag",
        "sg.TagFrame",
        "sg.Arrow",
        "sg.ArrowHead",
        "sg.extract_glyph_path",
        "sg.get_text_dimensions",
        "sg.get_text_size",
        "See also: sg.help('tag_objects'), sg.help('canvas_doc')",
    ],
    "text_doc": [
        "sg.Canvas.text",
        "sg.Tag",
        "sg.TagFrame",
        "sg.Canvas.draw_latex",
        "sg.extract_glyph_path",
        "sg.get_text_dimensions",
        (
            "See also: sg.help('tag_objects'), sg.help('canvas_doc'), "
            "sg.help('images_doc')"
        ),
    ],
    "tex_compiler": [
        "sg.Canvas.save",
        "sg.user_config_path",
        "sg.set_user_settings_path",
        (
            "See also: sg.help('user_settings'), sg.help('image_converters'), "
            "sg.help('script_sharing'), sg.help('viewer')"
        ),
    ],
    "tolerances": [
        "abs_tol",
        "rel_tol",
        "sg.check_angle_tol",
        "sg.check_dist_tol",
        "sg.equal_angles",
        "sg.equal_points",
        "sg.distance",
        "sg.resolve_tol",
        "See also: sg.help('tolerances_doc'), sg.help('abs_tol'), sg.help('rel_tol')",
    ],
    "tolerances_doc": [
        "abs_tol",
        "rel_tol",
        "sg.resolve_tol",
        "sg.get_defaults",
        "sg.equal_points",
        "sg.equal_angles",
        "See also: sg.help('tolerances'), sg.help('user_settings')",
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
        "sg.Transform",
        "sg.Transformation",
        "sg.translate",
        "sg.translation_matrix",
        (
            "translate(dx, dy) is two lengths; translate((x, y)) or "
            "translate(edge) is one vector. Shape.transform accepts a "
            "matrix or a Transformation. See "
            "sg.help('dynamic_references_doc'), "
            "sg.help('composite_transformations'), "
            "and sg.doc(sg.Shape.translate)."
        ),
    ],
    "transforms_doc": [
        "sg.Transform",
        "sg.Transformation",
        "sg.DynRef",
        "sg.translate",
        "sg.rotate",
        "sg.mirror",
        "sg.glide",
        "sg.scale",
        "sg.shear",
        "sg.translation_matrix",
        "sg.rotation_matrix",
        (
            "See also: sg.help('dynamic_references_doc'), "
            "sg.help('composite_transformations'), "
            "sg.help('shapes_doc'), sg.help('groups_doc'), "
            "sg.help('canvas_context_managers')"
        ),
    ],
    "user_settings": [
        "sg.temp_defaults",
        "sg.user_defaults",
        "converters",
        "default_output_directory",
        "paths",
        "shell",
        "sg.user_config_path",
        "sg.set_user_settings_path",
        "sg.defaults",
        "sg.save_as",
        "sg.save_user_defaults",
        "sg.use_script_header",
        "sg.save_user_style",
        "sg.save_user_warning",
        "sg.set_defaults",
        "sg.set_svg_defaults",
        "sg.set_tikz_defaults",
        "sg.Canvas.save",
        (
            "See also: sg.help('warnings'), sg.help('canvas'), "
            "sg.help('image_converters'), sg.help('tex_compiler'), "
            "sg.help('viewer')"
        ),
    ],
    "vertices": [
        "Shape.vertices / Shape.primary_points",
        "sg.close_points_square",
        "sg.fix_degen_points",
        "sg.homogenize",
        "sg.round_point",
        "See also: sg.help('points'), sg.help('shapes')",
    ],
    "viewer": [
        "sg.Canvas.save",
        "sg.user_config_path",
        "sg.set_user_settings_path",
        "sg.defaults['show_browser']",
        (
            "See also: sg.help('user_settings'), sg.help('export'), "
            "sg.help('script_sharing')"
        ),
    ],
    "warnings": [
        "sg.WarningType",
        "sg.set_warning_on",
        "sg.set_warning_off",
        "sg.set_all_warnings_on",
        "sg.set_all_warnings_off",
        "sg.save_user_warning",
        "sg.pause_warning",
        "sg.resume_warning",
        "sg.pause_warnings",
        "sg.resume_warnings",
        "See also: sg.help('user_settings')",
    ],
}

d_help_topic["segments"] = list(d_help_topic["lines"])

_TOPIC_ALIASES = {
    "AlignedDimension": "dimensioning_doc",
    "AnnotationArrow": "dimensioning_doc",
    "BoundingBox": "bounding_box_doc",
    "BoundingBoxes": "bounding_box_doc",
    "bounding-box": "bounding_box_doc",
    "bounding_box": "bounding_box_doc",
    "bounding-boxes": "bounding_box_doc",
    "bounding_boxes": "bounding_box_doc",
    "bbox": "bounding_box_doc",
    "b_box": "bounding_box_doc",
    "BooleanOps": "boolean_ops",
    "building-help": "building_help",
    "BuildingHelp": "building_help",
    "Canvas": "canvas_doc",
    "canvas": "canvas_doc",
    "canvas-context-managers": "canvas_context_managers",
    "canvas_context_managers": "canvas_context_managers",
    "context-managers": "canvas_context_managers",
    "context_managers": "canvas_context_managers",
    "ContextManagers": "canvas_context_managers",
    "Clipping": "clipping",
    "Colors": "colors",
    "CompositeTransformations": "composite_transformations",
    "composite_transformations_with_dynamic_references": "composite_transformations",
    "Dimension": "dimensioning_doc",
    "dimensioning": "dimensioning_doc",
    "Dimensioning": "dimensioning_doc",
    "dimensions": "dimensioning_doc",
    "draw_lace": "drawing_laces_doc",
    "drawing-laces": "drawing_laces_doc",
    "drawing-laces-doc": "drawing_laces_doc",
    "drawing_laces": "drawing_laces_doc",
    "drawing_laces_doc": "drawing_laces_doc",
    "dynamic_references": "dynamic_references_doc",
    "DynamicReferences": "dynamic_references_doc",
    "Edges": "edges",
    "Effects": "effects",
    "extract_glyph_path": "text_doc",
    "Export": "export",
    "external-converters": "image_converters",
    "ExternalConverters": "image_converters",
    "external_converters": "image_converters",
    "converters": "image_converters",
    "image-converters": "image_converters",
    "ImageConverters": "image_converters",
    "tex": "tex_compiler",
    "tex-compiler": "tex_compiler",
    "TexCompiler": "tex_compiler",
    "latex": "tex_compiler",
    "LaTeX": "tex_compiler",
    "Grids": "grids",
    "help": "help_doc",
    "Help": "help_doc",
    "help-doc": "help_doc",
    "Group": "groups_doc",
    "groups": "groups_doc",
    "Groups": "groups_doc",
    "groups-doc": "groups_doc",
    "Batch": "groups_doc",
    "Image": "images_doc",
    "image": "images_doc",
    "images": "images_doc",
    "Images": "images_doc",
    "images-doc": "images_doc",
    "Lines": "lines",
    "latex-engine": "latex_engine",
    "LatexEngine": "latex_engine",
    "Lace": "drawing_laces_doc",
    "lace": "drawing_laces_doc",
    "laces": "drawing_laces_doc",
    "Lattice": "lattices_doc",
    "lattice": "lattices_doc",
    "lattices": "lattices_doc",
    "lattices-doc": "lattices_doc",
    "lattice_p1": "lattices_doc",
    "Isometry": "lattices_doc",
    "LatType": "lattices_doc",
    "LatRef": "lattices_doc",
    "LinPath": "path_objects_doc",
    "Path2D": "path_objects_doc",
    "path": "path_objects_doc",
    "paths": "path_objects_doc",
    "path_code": "path_objects_doc",
    "shape_to_path2d": "path_objects_doc",
    "path_objects": "path_objects_doc",
    "path_objects_doc": "path_objects_doc",
    "path-objects": "path_objects_doc",
    "path-objects-doc": "path_objects_doc",
    "svg_path_to_path2d": "path_objects_doc",
    "Patterns": "patterns",
    "Points": "points",
    "Polygons": "polygons",
    "pop_matrix": "canvas_context_managers",
    "pop_style": "canvas_context_managers",
    "push_matrix": "canvas_context_managers",
    "push_style": "canvas_context_managers",
    "random-seeds": "random_seeds",
    "RandomSeeds": "random_seeds",
    "Segments": "segments",
    "Shape": "shapes_doc",
    "shapes": "shapes_doc",
    "shapes-doc": "shapes_doc",
    "Shapes": "shapes_doc",
    "script-sharing": "script_sharing",
    "ScriptSharing": "script_sharing",
    "save_user_defaults": "user_settings",
    "save_user_style": "style_definitions",
    "save_user_warning": "warnings",
    "Style": "style_definitions",
    "style": "style_definitions",
    "style-definitions": "style_definitions",
    "styles": "style_definitions",
    "user_styles": "style_definitions",
    "Tag": "tag_objects",
    "tag-objects": "tag_objects",
    "TagFrame": "tag_objects",
    "Tags": "tag_objects",
    "tags": "tag_objects",
    "Table": "tables_doc",
    "table": "tables_doc",
    "tables": "tables_doc",
    "tables-doc": "tables_doc",
    "Text": "text_doc",
    "text": "text_doc",
    "text-doc": "text_doc",
    "Tolerances": "tolerances",
    "Transforms": "transforms_doc",
    "transforms": "transforms_doc",
    "transformations": "transforms_doc",
    "transforms-doc": "transforms_doc",
    "user-settings": "user_settings",
    "UserSettings": "user_settings",
    "viewer": "viewer",
    "preview": "viewer",
    "formatting": "print_doc",
    "p_print": "print_doc",
    "print": "print_doc",
    "print-doc": "print_doc",
    "Print": "print_doc",
    "open_saved": "viewer",
    "open-saved": "viewer",
    "Vertices": "vertices",
    "Warnings": "warnings",
    "WarningType": "warnings",
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
- this guide:        sg.help('help')  or  sg.help('help_doc')
- short lookup rules: sg.help(sg.help)

Topics
------
"""
    + ", ".join(sorted({*_TOPIC_ALIASES.values(), *d_help_topic}))
    + """

Callables / classes
-------------------
- sg.help(sg.Shape) or sg.help(sg.Shape(...))
  -> constructor signature, class docstring, __init__ docstring
- sg.help(sg.distance) or sg.help(sg.Canvas.draw)
  -> callable signature, accepted ``**kwargs`` when known, docstring,
  and any extra notes

Missing defaults keys return an empty string.
Unknown names list similar topics, settings, and public ``sg`` names.
``sg.help(obj, exact=False)`` keeps an exact match and also lists similar names.
``exclude='triangle'`` or ``exclude=('triangle', 'rectangle')`` omits
names and topics that contain any of those strings.
"""
)

_SIGNATURE_DEFAULT_SKIP_NAMES = frozenset(
    {"page_size", "plait_style", "swatch"}
)


def _default_overrides_for_signature(
    signature: inspect.Signature,
    *,
    skip_names: frozenset[str] = _SIGNATURE_DEFAULT_SKIP_NAMES,
) -> dict[str, object]:
    """Return ``defaults`` values to display for ``None``-default parameters."""
    overrides: dict[str, object] = {}
    for parameter in signature.parameters.values():
        if parameter.default is not None:
            continue
        name = parameter.name
        if name in skip_names:
            continue
        if name not in defaults.defaults:
            continue
        default_value = user_defaults[name]
        if default_value is VOID:
            continue
        overrides[name] = default_value
    return overrides


def _class_signature(cls: type) -> str:
    """Return ``ClassName(...)`` with constructor signature when available."""
    try:
        signature = inspect.signature(cls)
        return _format_signature(
            cls.__name__,
            signature,
            default_overrides=_default_overrides_for_signature(signature),
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
    lines.extend(f"    {part}," for part in parts)

    closing = ")"
    if signature.return_annotation is not inspect.Signature.empty:
        closing += f" -> {_format_annotation(signature.return_annotation)}"
    lines.append(closing)

    return "\n".join(lines)


def _signature_accepts_kwargs(signature: inspect.Signature) -> bool:
    """Return True when ``signature`` includes ``**kwargs``."""
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _explicit_parameter_names(signature: inspect.Signature) -> frozenset[str]:
    """Return named parameters, excluding ``self``, ``cls``, and ``*`` forms."""
    names: set[str] = set()
    for parameter in signature.parameters.values():
        if parameter.name in ("self", "cls"):
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue
        names.add(parameter.name)
    return frozenset(names)


def _format_kwarg_default_line(name: str) -> str:
    """Return one accepted-kwarg line, with a default value when available."""
    if name not in defaults.defaults:
        return name
    value = user_defaults[name]
    if value is VOID:
        return name
    return f"{name} = {_format_default(value)}"


def _format_accepted_kwargs_section(
    heading: str,
    key_names: Sequence[str],
) -> str:
    """Format accepted ``**kwargs`` names and their defaults."""
    if not key_names:
        return ""
    lines = [heading]
    lines.extend(f"  {_format_kwarg_default_line(name)}" for name in key_names)
    return "\n".join(lines)


def _canvas_init_kwargs_keys(signature: inspect.Signature) -> list[str]:
    """Return ``Canvas`` constructor kwargs from ``canvas_args``."""
    from ..render.style_map import canvas_args

    explicit_names = _explicit_parameter_names(signature)
    return sorted(
        name for name in canvas_args if name not in explicit_names
    )


def _canvas_draw_kwargs_keys(signature: inspect.Signature) -> list[str]:
    """Return ``Canvas.draw`` kwargs from ``get_draw_valid_kwargs()``."""
    from ..render.style_map import get_draw_valid_kwargs

    explicit_names = _explicit_parameter_names(signature)
    return sorted(
        name
        for name in get_draw_valid_kwargs()
        if name not in explicit_names and not name.startswith("_")
    )


_CLASS_KWARGS_KEY_SOURCES: dict[
    tuple[str, str], Callable[[inspect.Signature], Sequence[str]]
] = {
    ("simetri.render.canvas", "Canvas"): _canvas_init_kwargs_keys,
}

_CALLABLE_KWARGS_KEY_SOURCES: dict[
    str, Callable[[inspect.Signature], Sequence[str]]
] = {
    "Canvas.draw": _canvas_draw_kwargs_keys,
    "Canvas.draw_lace": _canvas_draw_kwargs_keys,
}


def _append_accepted_kwargs_help(
    parts: list[str],
    signature: inspect.Signature,
    key_source: Callable[[inspect.Signature], Sequence[str]],
) -> None:
    """Append accepted ``**kwargs`` documentation when the signature has them."""
    if not _signature_accepts_kwargs(signature):
        return
    key_names = key_source(signature)
    section = _format_accepted_kwargs_section(
        "Accepted **kwargs (defaults from settings)",
        key_names,
    )
    if section:
        parts.append(section)


_CANVAS_DRAW_HELP_NOTES = (
    "How canvas.draw works\n"
    "style kwargs override the drawn object's corresponding style attributes\n"
    "for Shape and Group, pos / angle / scale are applied to the drawn snapshot\n"
    "pos, rotocenter, and about use canvas units (points); angles use radians\n"
    "canvas.translate/rotate/scale change the canvas coordinate system for later drawing"
)

_CANVAS_DRAW_LACE_HELP_NOTES = (
    "Lace-specific options belong on canvas.draw_lace, not canvas.draw.\n"
    "None on a named argument means use the matching defaults entry.\n"
    "fragment_coloring uses FragmentColoring.AREA or FragmentColoring.RADIUS.\n"
    "plait_style uses PlaitStyle.EMBOSS1, EMBOSS2, DIAMOND, INNERLINES, or "
    "DOUBLE_LINES; INNERLOOPS raises until implemented.\n"
    "Generic shape styles in **kwargs are forwarded to fragment and plait draws."
)

_CALLABLE_HELP_NOTES: dict[str, str] = {
    "Canvas.draw": _CANVAS_DRAW_HELP_NOTES,
    "Canvas.draw_lace": _CANVAS_DRAW_LACE_HELP_NOTES,
}


def _callable_signature(obj: Callable[..., object]) -> inspect.Signature:
    """Return a display signature, omitting ``self`` / ``cls`` when present."""
    signature = inspect.signature(obj)
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name in ("self", "cls"):
        parameters = parameters[1:]
    return signature.replace(parameters=parameters)


def _callable_help(obj: Callable[..., object]) -> str:
    """Build help text for a function or method: signature, doc, and notes."""
    parts: list[str] = []
    signature = _callable_signature(obj)
    parts.append(
        _format_signature(
            obj.__qualname__,
            signature,
            default_overrides=_default_overrides_for_signature(signature),
        )
    )

    if obj.__qualname__ in _CALLABLE_KWARGS_KEY_SOURCES:
        _append_accepted_kwargs_help(
            parts,
            signature,
            _CALLABLE_KWARGS_KEY_SOURCES[obj.__qualname__],
        )

    doc = inspect.getdoc(obj)
    if doc:
        parts.append(doc)

    if obj.__qualname__ in _CALLABLE_HELP_NOTES:
        parts.append(_CALLABLE_HELP_NOTES[obj.__qualname__])

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

    class_key = (cls.__module__, cls.__name__)
    if class_key in _CLASS_KWARGS_KEY_SOURCES:
        _append_accepted_kwargs_help(
            parts,
            inspect.signature(cls),
            _CLASS_KWARGS_KEY_SOURCES[class_key],
        )

    if cls.__module__ == "simetri.render.canvas" and cls.__name__ == "Canvas":
        canvas_default_lines = [
            "page_size = calculated automatically when omitted",
            "canvas.draw(...) snapshots objects into sketch objects on the active page",
            "canvas.draw(..., style=value) overrides the drawn object's corresponding style attributes",
            "for Shape and Group, canvas.draw(..., pos=..., angle=...) moves or rotates the drawn snapshot",
            "canvas.translate/rotate/scale change the canvas coordinate system for later drawing",
            "see also: sg.doc(sg.Canvas.draw)",
        ]
        parts.append(
            "Constructor behavior\n" + "\n".join(canvas_default_lines)
        )

    if cls.__module__ == "simetri.group.batch" and cls.__name__ == "Group":
        parts.append(
            "Constructor behavior\n"
            "elements = [] when omitted\n"
            "modifiers = [] when omitted"
        )

    return "\n\n".join(parts)


def _sg_help_entry(name: str) -> str:
    """Normalize a public export name to ``sg.*`` listing form."""
    stripped = name.removeprefix("sg.")
    return f"sg.{stripped}"


def _doc_placeholders_for_topic(topic: str) -> tuple[str, ...]:
    """Return merged ``{{doc:…}}`` placeholders for ``topic``."""
    linked: list[str] = list(TOPIC_DOC_PLACEHOLDERS.get(topic, ()))
    linked.extend(COMPILED_DOC_PLACEHOLDERS_BY_TOPIC.get(topic, ()))
    return tuple(sorted(set(linked), key=str.casefold))


def _supplement_sg_entries_for_topic(topic: str) -> tuple[str, ...]:
    """Return manual ``sg.*`` lines linked to ``topic`` (dict + graph)."""
    linked: list[str] = list(TOPIC_SG_ENTRIES.get(topic, ()))
    for export_name, topics in SG_NAME_TO_TOPICS.items():
        if topic in topics:
            linked.append(_sg_help_entry(export_name))
    deduped = sorted(set(linked), key=str.casefold)
    return tuple(deduped)


def _topic_stem(topic: str) -> str:
    """Guide topic key without a trailing ``_doc`` suffix when present."""
    if topic.endswith("_doc"):
        return topic[: -len("_doc")]
    return topic


def _export_belongs_to_topic(topic: str, export_line: str) -> bool:
    """Return True when ``export_line`` is a primary match for ``topic``."""
    from .compile_help_topics import topic_query_tokens
    from .help_visibility import help_line_leaf_name

    leaf = help_line_leaf_name(export_line)
    stem = _topic_stem(topic)
    if leaf == stem:
        return True
    stem_tokens = topic_query_tokens(stem)
    export_tokens = _help_name_tokens(leaf)
    shared = stem_tokens & export_tokens
    noise = {"doc", "shape", "shapes"}
    if not (shared - noise):
        return False
    stem_compact = stem.replace("_", "")
    leaf_compact = leaf.replace("_", "")
    return stem_compact in leaf_compact or leaf_compact in stem_compact


def _topic_sg_index_lines(topic: str) -> list[str]:
    """Return ``sg.*`` lines for a topic guide footer (token match at runtime)."""
    from .compile_help_topics import _sg_names_for_topic
    from .help_visibility import HelpVisibilityContext, is_help_name_visible

    manual = [
        line
        for line in d_help_topic.get(topic, ())
        if line.startswith("sg.")
    ]
    token_matched = _sg_names_for_topic(topic, _public_sg_names())
    supplements = list(_supplement_sg_entries_for_topic(topic))
    seen: set[str] = set()
    lines: list[str] = []
    for entry in manual + list(token_matched) + supplements:
        if not entry.startswith("sg.") or entry in seen:
            continue
        if not is_help_name_visible(
            entry, context=HelpVisibilityContext.STRING_LOOKUP
        ):
            continue
        if not _export_belongs_to_topic(topic, entry):
            continue
        lines.append(entry)
        seen.add(entry)
    return sorted(lines, key=str.casefold)


def _merge_topic_entries(topic: str) -> list[str]:
    """Merge curated, compiled, and supplement ``sg.*`` topic lines."""
    manual = list(d_help_topic.get(topic, ()))
    compiled = COMPILED_SG_BY_TOPIC.get(topic, ())
    supplements = _supplement_sg_entries_for_topic(topic)
    see_also = [
        line
        for line in manual
        if line.startswith("See also:") or line.startswith("(")
    ]
    manual_body = [line for line in manual if line not in see_also]
    seen = {line for line in manual_body if line.startswith("sg.")}
    from .help_visibility import (
        INCLUDE_COMPILED_IN_TOPIC_BROWSE,
        is_help_name_visible,
        HelpVisibilityContext,
    )

    merged = list(manual_body)
    if INCLUDE_COMPILED_IN_TOPIC_BROWSE:
        for entry in compiled:
            if entry not in seen and is_help_name_visible(
                entry,
                context=HelpVisibilityContext.COMPILED_TOPIC,
            ):
                merged.append(entry)
                seen.add(entry)
    for entry in supplements:
        if entry not in seen:
            merged.append(entry)
            seen.add(entry)
    doc_refs = _doc_placeholders_for_topic(topic)
    if doc_refs:
        merged.append("Documentation (mkdocs placeholders):")
        merged.extend(format_doc_placeholder_for_help(ref) for ref in doc_refs)
    related = RELATED_TOPICS.get(topic, ())
    if related:
        hints = ", ".join(f"sg.help({related_topic!r})" for related_topic in related)
        merged.append(f"Related topics: {hints}")
    merged.extend(see_also)
    return merged


def _queries_for_topic(topic: str) -> set[str]:
    """Return the topic key plus aliases that resolve to it."""
    queries = {topic}
    for alias, dest in _TOPIC_ALIASES.items():
        if dest == topic:
            queries.add(alias)
    for alias, dest in COMPILED_TOPIC_ALIASES.items():
        if dest == topic:
            queries.add(alias)
    for alias, dest in SUPPLEMENT_TOPIC_ALIASES.items():
        if dest == topic:
            queries.add(alias)
    return queries


def _matching_names_for_topic(
    topic: str, entries: Sequence[str], exclude: tuple[str, ...] | None = None
) -> list[str]:
    """Return extra leaf-substring matches not already listed on the topic."""
    listed = "\n".join(entries)
    seen: set[str] = set()
    extras: list[str] = []
    for query in sorted(_queries_for_topic(topic), key=str.casefold):
        for name in _similar_help_names(query, exclude=exclude):
            key = name.casefold()
            if key in seen or name in listed:
                continue
            seen.add(key)
            extras.append(name)
    return extras


def _format_topic(
    topic: str, entries: Sequence[str], exclude: tuple[str, ...] | None = None
) -> str:
    """Format a topic heading and its ``sg.*`` entry list."""
    lines = [f"Topic: {topic}", ""]
    for entry in entries:
        if _help_name_excluded(entry, exclude):
            continue
        lines.append(entry)
    extras = _matching_names_for_topic(topic, entries, exclude=exclude)
    if extras:
        lines.append("")
        lines.append("Matching names:")
        lines.extend(f"  {name}" for name in extras)
    lines.append("")
    lines.append(
        "Use sg.help(name) on a callable, or sg.help('setting') for defaults."
    )
    return "\n".join(lines)


def _parse_qmd_help_aliases(path: Path) -> tuple[str, ...]:
    """Return ``help_aliases`` from YAML front matter when present."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    if not text.startswith("---"):
        return ()
    end = text.find("\n---", 3)
    if end == -1:
        return ()
    aliases: list[str] = []
    in_aliases = False
    for line in text[3:end].splitlines():
        if re.match(r"^help_aliases:\s*$", line):
            in_aliases = True
            continue
        inline = re.match(r"^help_aliases:\s*\[(.*)\]\s*$", line)
        if inline:
            inner = inline.group(1)
            for part in inner.split(","):
                part = part.strip().strip("'\"")
                if part:
                    aliases.append(part)
            return tuple(aliases)
        if in_aliases:
            item = re.match(r"^\s+-\s+(.+)$", line)
            if item:
                aliases.append(item.group(1).strip().strip("'\""))
                continue
            if line and not line.startswith(" "):
                break
    return tuple(aliases)


def _parse_qmd_title(path: Path) -> str:
    """Return YAML ``title`` from a topic guide front matter."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return path.stem
    if not text.startswith("---"):
        return path.stem
    end = text.find("\n---", 3)
    if end == -1:
        return path.stem
    for line in text[3:end].splitlines():
        match = re.match(r'^title:\s*"(.*)"\s*$', line)
        if match:
            return match.group(1)
        match = re.match(r"^title:\s*(.+)\s*$", line)
        if match:
            return match.group(1).strip().strip("'\"")
    return path.stem


@lru_cache(maxsize=1)
def _topic_guide_paths() -> dict[str, Path]:
    """Map topic key (``.qmd`` stem) to path for every topic guide file."""
    if not _TOPIC_GUIDES_DIR.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(_TOPIC_GUIDES_DIR.glob("*.qmd"))
    }


@lru_cache(maxsize=1)
def all_help_topic_keys() -> frozenset[str]:
    """Curated ``d_help_topic`` keys plus every ``topic_guides/*.qmd`` stem."""
    return frozenset(d_help_topic) | frozenset(_topic_guide_paths())


@lru_cache(maxsize=1)
def _topic_guide_aliases() -> dict[str, str]:
    """Map alternate help query strings to a topic guide stem."""
    paths = _topic_guide_paths()
    topic_keys = set(paths)
    public_exports = set(_public_sg_names())
    aliases: dict[str, str] = {}

    def register(alias: str, topic: str) -> None:
        if not alias or alias in topic_keys:
            return
        if alias in public_exports:
            return
        if alias not in aliases:
            aliases[alias] = topic

    for stem, path in paths.items():
        if stem.endswith("_doc"):
            register(stem[: -len("_doc")], stem)
    return aliases


@lru_cache(maxsize=1)
def _topic_help_hubs() -> dict[str, str]:
    """Map short ``help_aliases`` keys to topic guide stems (hub, not redirect)."""
    hubs: dict[str, str] = {}
    for stem, path in _topic_guide_paths().items():
        for alias in _parse_qmd_help_aliases(path):
            if alias not in hubs:
                hubs[alias] = stem
    return hubs


def _primary_export_leaves_for_topic(topic: str) -> list[str]:
    """Export leaf names that are the primary API for a topic guide stem."""
    from .help_visibility import help_line_leaf_name

    stem = _topic_stem(topic)
    leaves: list[str] = []
    for entry in _topic_sg_index_lines(topic):
        leaf = help_line_leaf_name(entry)
        if leaf == stem:
            leaves.append(leaf)
    return leaves


def _group_method_hub_line(stem: str) -> str | None:
    """Hub line when ``stem`` names a public method on ``sg.Group``."""
    import simetri.graphics as sg

    attr = getattr(sg.Group, stem, None)
    if attr is None:
        return None
    if isinstance(attr, classmethod):
        func = attr.__func__
    else:
        func = attr
    if not inspect.isroutine(func):
        return None
    qualified = f"Group.{stem}"
    return (
        f"- sg.help({qualified!r})  — Group method — sg.{qualified}"
    )


def _format_help_hub(hub_key: str, exclude: tuple[str, ...] | None = None) -> str:
    """Format a short help hub as line items (topic guide + related API)."""
    topic = _topic_help_hubs()[hub_key]
    path = _topic_guide_paths()[topic]
    title = _parse_qmd_title(path)
    stem = _topic_stem(topic)
    lines = [
        f"Help: {hub_key}",
        "",
    ]
    topic_line = f"- sg.help({topic!r})  — topic guide — {title}"
    if not _help_name_excluded(topic, exclude) and not _help_name_excluded(
        title, exclude
    ):
        lines.append(topic_line)
    group_line = _group_method_hub_line(stem)
    if group_line is not None and not _help_name_excluded(group_line, exclude):
        lines.append(group_line)
    for leaf in _primary_export_leaves_for_topic(topic):
        if _help_name_excluded(leaf, exclude):
            continue
        lines.append(f"- sg.help({leaf!r})  — function — sg.{leaf}")
    lines.append("")
    lines.append("Use sg.doc(...) with the same query to print.")
    return "\n".join(lines)


def _resolve_help_topic_key(query: str) -> str:
    """Map a help query string to a canonical topic key."""
    if query in _TOPIC_ALIASES:
        return _TOPIC_ALIASES[query]
    if query in COMPILED_TOPIC_ALIASES:
        return COMPILED_TOPIC_ALIASES[query]
    if query in SUPPLEMENT_TOPIC_ALIASES:
        return SUPPLEMENT_TOPIC_ALIASES[query]
    guide_aliases = _topic_guide_aliases()
    if query in guide_aliases:
        return guide_aliases[query]
    return query


def _topic_guide_text(topic: str) -> str | None:
    """Return topic-guide text when ``topic_guides/{topic}.qmd`` exists."""
    path = _topic_guide_paths().get(topic)
    if path is None:
        path = _TOPIC_GUIDES_DIR / f"{topic}.qmd"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _format_topics(exclude: tuple[str, ...] | None = None) -> str:
    """Return the sorted list of help topic names."""
    topics = sorted(
        set(all_help_topic_keys())
        | set(_TOPIC_ALIASES.values())
        | set(_topic_guide_aliases())
        | set(_topic_help_hubs())
        | {"help", "topics"}
    )
    if exclude is not None:
        topics = [
            topic for topic in topics if not _help_name_excluded(topic, exclude)
        ]
    return "Available help topics:\n  " + "\n  ".join(topics)


def _is_named_help_object(obj: object) -> bool:
    """Return True for public objects worth resolving by string name."""
    if inspect.isclass(obj) or inspect.isroutine(obj) or inspect.ismodule(obj):
        return True
    if isinstance(obj, Enum):
        return True
    return type(obj).__module__.startswith("simetri.")


def _register_named_help_object(
    mapping: dict[str, object], name: str, obj: object
) -> None:
    """Register both ``name`` and ``sg.name`` for string lookup."""
    from .help_visibility import filter_named_help_registration

    if not filter_named_help_registration(name, obj):
        return
    if name not in mapping:
        mapping[name] = obj
    sg_name = f"sg.{name}"
    if sg_name not in mapping:
        mapping[sg_name] = obj


def _register_class_members(
    mapping: dict[str, object], class_name: str, cls: type, depth: int = 1
) -> None:
    """Register public members for one exported class."""
    for member_name, member in inspect.getmembers(cls):
        if member_name.startswith("_"):
            continue
        qualified_name = f"{class_name}.{member_name}"
        _register_named_help_object(mapping, qualified_name, member)
        if depth > 1 and inspect.isclass(member):
            _register_class_members(
                mapping, qualified_name, member, depth=depth - 1
            )


@lru_cache(maxsize=1)
def _public_sg_names() -> list[str]:
    """Return public top-level names exported on ``simetri.graphics``."""
    module = sys.modules.get("simetri.graphics")
    if module is None:
        return []

    names = []
    for name, obj in vars(module).items():
        if name.startswith("_") or not _is_named_help_object(obj):
            continue
        names.append(name)

    return sorted(set(names))


def _similar_sg_attribute_names(
    query: str, limit: int | None = None
) -> list[str]:
    """Return similar top-level ``sg`` attribute names."""
    limit = _resolve_help_suggestion_limit(limit)
    names = _public_sg_names()
    suggestions: list[str] = []
    seen: set[str] = set()
    query_tokens = tuple(sorted(_help_name_tokens(query)))
    normalized_query_tokens = tuple(normalize(token) for token in query_tokens)

    def add(name: str) -> None:
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            suggestions.append(name)

    leaf_matches = [
        name for name in names if _query_in_help_leaf(query, name)
    ]
    for name in sorted(leaf_matches, key=lambda item: (len(item), item)):
        add(name)
    if suggestions:
        return suggestions

    direct_matches = find_similar(query, names, limit=limit)
    for name, _score in direct_matches:
        add(name)
    if suggestions:
        return suggestions[:limit]

    exact_token_matches = [
        name
        for name in names
        if set(query_tokens).intersection(_help_name_tokens(name))
    ]

    for name in sorted(exact_token_matches, key=lambda item: (len(item), item)):
        add(name)
        if len(suggestions) >= limit:
            return suggestions

    scored_matches = []
    for name in names:
        best_score = 0.0
        for token in _help_name_tokens(name):
            normalized_token = normalize(token)
            for normalized_query_token in normalized_query_tokens:
                score = DamerauLevenshtein.normalized_similarity(
                    normalized_query_token, normalized_token
                )
                best_score = max(best_score, score)
        if best_score >= 0.75:
            scored_matches.append((name, best_score))

    for name, _score in sorted(
        scored_matches,
        key=lambda item: (-item[1], len(item[0]), item[0]),
    ):
        add(name)
        if len(suggestions) >= limit:
            return suggestions

    return suggestions


@lru_cache(maxsize=1)
def _named_help_objects() -> dict[str, object]:
    """Return public ``sg`` names that can be resolved from strings."""
    module = sys.modules.get("simetri.graphics")
    if module is None:
        return {}

    mapping: dict[str, object] = {}
    for name, obj in vars(module).items():
        if name.startswith("_") or not _is_named_help_object(obj):
            continue
        _register_named_help_object(mapping, name, obj)
        if inspect.isclass(obj) and obj.__module__.startswith("simetri."):
            depth = 2 if name == "WarningType" else 1
            _register_class_members(mapping, name, obj, depth=depth)

    return mapping


@lru_cache(maxsize=1)
def _help_lookup_names() -> list[str]:
    """Return names that can be resolved or suggested by ``sg.help``."""
    names = set(all_help_topic_keys())
    names.update(_TOPIC_ALIASES)
    names.update(_topic_guide_aliases())
    names.update(_topic_help_hubs())
    names.update(defaults.defaults)
    names.update(defaults_help)
    names.update(user_config_help_keys())
    names.update(_named_help_objects())
    names.add("help")
    names.add("topics")
    return sorted(names)


def _canonical_help_name(name: str) -> str:
    """Prefer ``sg.`` display for public object names when available."""
    if name.startswith("sg."):
        return name

    named_objects = _named_help_objects()
    sg_name = f"sg.{name}"
    if name in named_objects and sg_name in named_objects:
        return sg_name

    return name


def _help_name_excluded(name: str, exclude: tuple[str, ...] | None) -> bool:
    """Return True if ``name`` contains any validated exclude substring."""
    if exclude is None:
        return False
    folded = normalize(name)
    for needle in exclude:
        if needle in folded:
            return True
    return False


@lru_cache(maxsize=1)
def _canonical_help_names() -> list[str]:
    """Return canonicalized, de-duplicated help names for suggestions."""
    return sorted({_canonical_help_name(name) for name in _help_lookup_names()})


def _help_name_tokens(name: str) -> set[str]:
    """Return searchable tokens derived from one help name."""
    name = name.removeprefix("sg.")

    tokens = {name}
    leaf_name = name.rsplit(".", maxsplit=1)[-1]
    if not leaf_name.isupper():
        tokens.add(leaf_name)
        for part in leaf_name.replace("-", "_").split("_"):
            if part:
                tokens.add(part)

    return tokens


def _help_name_leaf(name: str) -> str:
    """Return the symbol leaf (``split_segment`` from ``sg.split_segment``)."""
    return name.removeprefix("sg.").rsplit(".", maxsplit=1)[-1]


def _query_in_help_leaf(query: str, name: str) -> bool:
    """Return True if the query is a substring of the help name's leaf."""
    query_normalized = normalize(query.removeprefix("sg."))
    if not query_normalized:
        return False
    return query_normalized in normalize(_help_name_leaf(name))


@lru_cache(maxsize=1)
def _help_token_to_names() -> dict[str, set[str]]:
    """Map searchable tokens back to their original help names."""
    token_to_names: dict[str, set[str]] = {}
    for canonical in _canonical_help_names():
        for token in _help_name_tokens(canonical):
            if token not in token_to_names:
                token_to_names[token] = set()
            token_to_names[token].add(canonical)

    return token_to_names


def _similar_help_names(
    query: str, limit: int | None = None, exclude: tuple[str, ...] | None = None
) -> list[str]:
    """Return similar help names: leaf substring first, then fuzzy tokens."""
    limit = _resolve_help_suggestion_limit(limit)
    lookup_names = _canonical_help_names()
    suggestions: list[str] = []
    seen: set[str] = set()
    query_tokens = tuple(sorted(_help_name_tokens(query)))
    normalized_query_tokens = tuple(normalize(token) for token in query_tokens)

    def sort_key(name: str) -> tuple[int, int, str]:
        return (0 if name.startswith("sg.") else 1, len(name), name)

    def add(name: str) -> None:
        if _help_name_excluded(name, exclude):
            return
        canonical = _canonical_help_name(name)
        key = canonical.casefold()
        if key not in seen:
            seen.add(key)
            suggestions.append(canonical)

    leaf_matches = [
        name for name in lookup_names if _query_in_help_leaf(query, name)
    ]
    for name in sorted(leaf_matches, key=sort_key):
        add(name)
    if suggestions:
        return suggestions

    direct_matches = find_similar(query, lookup_names, limit=limit)
    for name, _score in direct_matches:
        add(name)
    if suggestions:
        return suggestions[:limit]

    exact_token_matches = [
        name
        for name in lookup_names
        if set(query_tokens).intersection(_help_name_tokens(name))
    ]

    for name in sorted(exact_token_matches, key=sort_key):
        add(name)
        if len(suggestions) >= limit:
            return suggestions

    scored_matches = []
    for name in lookup_names:
        best_score = 0.0
        for token in _help_name_tokens(name):
            normalized_token = normalize(token)
            for normalized_query_token in normalized_query_tokens:
                score = DamerauLevenshtein.normalized_similarity(
                    normalized_query_token, normalized_token
                )
                best_score = max(best_score, score)
        if best_score >= 0.75:
            scored_matches.append((name, best_score))

    for name, _score in sorted(
        scored_matches,
        key=lambda item: (-item[1], *sort_key(item[0])),
    ):
        add(name)
        if len(suggestions) >= limit:
            return suggestions

    return suggestions


def _unknown_topic_help(query: str, exclude: tuple[str, ...] | None = None) -> str:
    """Return similar help names when ``query`` is not an exact match."""
    matches = _similar_help_names(query, exclude=exclude)
    if not matches:
        return (
            f"No help entry named {query!r}.\n"
            "Use sg.help('topics') for topics, or pass an sg object directly."
        )
    lines = [f"No help entry named {query!r}. Similar names:"]
    lines.extend(f"  {name}" for name in matches)
    return "\n".join(lines)


_DOCSTRING_SECTION_HEADERS = frozenset(
    {
        "Args",
        "Arguments",
        "Attributes",
        "Examples",
        "Note",
        "Notes",
        "Raises",
        "Returns",
        "See Also",
        "Yields",
    }
)


def _simetri_data_descriptor_owner(obj: object) -> tuple[type, str] | None:
    """Return ``(class, name)`` for Simetri ``__slots__`` / data descriptors."""
    if inspect.ismethoddescriptor(obj):
        return None
    if not inspect.isdatadescriptor(obj):
        return None
    name = getattr(obj, "__name__", None)
    objclass = getattr(obj, "__objclass__", None)
    if not isinstance(name, str) or not inspect.isclass(objclass):
        return None
    if not objclass.__module__.startswith("simetri."):
        return None
    return objclass, name


def _init_parameter_help(cls: type, param_name: str) -> str | None:
    """Return the ``Args`` bullet for ``param_name`` from ``cls.__init__``."""
    init = cls.__init__
    if init is object.__init__:
        return None
    doc = inspect.getdoc(init)
    if not doc:
        return None
    in_args = False
    param_pattern = re.compile(
        rf"^{re.escape(param_name)}(\s*:|/|\s)",
    )
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:"):
            in_args = True
            continue
        if not in_args:
            continue
        if stripped.endswith(":"):
            header = stripped[:-1].split()[0]
            if header in _DOCSTRING_SECTION_HEADERS and header not in (
                "Args",
                "Arguments",
            ):
                break
        if not (line.startswith(" ") or line.startswith("\t")):
            continue
        if param_pattern.match(stripped):
            return stripped
    return None


def _help_for_simetri_data_descriptor(obj: object) -> str | None:
    """Help text for class data descriptors (e.g. ``sg.Shape.fill``).

    Returns ``None`` when ``obj`` is not a handled descriptor.
    """
    owner = _simetri_data_descriptor_owner(obj)
    if owner is None:
        return None
    cls, name = owner
    return _init_parameter_help(cls, name) or ""


def _help_text(obj: object, exclude: tuple[str, ...] | None = None) -> str:
    """Return the exact-match help string for ``obj``."""
    if obj is help:
        return _HELP_ABOUT_HELP

    # StrEnum members are also ``str``; resolve WarningType paths first.
    warning_path = _warning_type_path(obj)
    if warning_path is not None:
        return _warning_type_help(obj)

    if isinstance(obj, Enum):
        doc = inspect.getdoc(obj)
        return doc if doc is not None else ""

    descriptor_help = _help_for_simetri_data_descriptor(obj)
    if descriptor_help is not None:
        return descriptor_help

    if isinstance(obj, str):
        config_text = _user_config_help_for_string(obj)
        if config_text is not None:
            return config_text
        if obj in _topic_help_hubs():
            return _format_help_hub(obj, exclude=exclude)
        if obj == "topics":
            return _format_topics(exclude=exclude)
        named_obj = _named_help_objects().get(obj)
        if named_obj is None:
            named_obj = _named_help_objects().get(f"sg.{obj}")
        if named_obj is not None:
            return _help_text(named_obj)
        topic = _resolve_help_topic_key(obj)
        if topic in all_help_topic_keys():
            guide = _topic_guide_text(topic)
            if guide is not None:
                return guide
            return _format_topic(
                topic, _merge_topic_entries(topic), exclude=exclude
            )
        if obj in defaults:
            if obj in defaults_help:
                return defaults_help[obj]
            return ""
        return _unknown_topic_help(obj, exclude=exclude)

    if inspect.isclass(obj):
        return _class_help(obj)

    if inspect.isroutine(obj):
        if _module_name(obj).startswith("simetri."):
            return _callable_help(obj)
        doc = inspect.getdoc(obj)
        return doc if doc is not None else ""

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


def _doc_title(obj: object) -> str:
    """Return the display title used by ``sg.doc`` for ``obj``."""
    warning_path = _warning_type_path(obj)
    if warning_path is not None:
        return f"sg.{warning_path}"

    if isinstance(obj, Enum):
        enum_type = type(obj)
        if enum_type.__module__.startswith("simetri."):
            return f"sg.{enum_type.__qualname__}.{obj.name}"
        return f"{enum_type.__qualname__}.{obj.name}"

    if isinstance(obj, str):
        return obj

    if inspect.ismodule(obj):
        return obj.__name__

    if inspect.isclass(obj):
        if _module_name(obj).startswith("simetri."):
            return f"sg.{obj.__qualname__}"
        return obj.__qualname__

    if inspect.isroutine(obj):
        if _module_name(obj).startswith("simetri."):
            return f"sg.{obj.__qualname__}"
        return obj.__qualname__

    descriptor_owner = _simetri_data_descriptor_owner(obj)
    if descriptor_owner is not None:
        cls, name = descriptor_owner
        return f"sg.{cls.__qualname__}.{name}"

    if not isinstance(obj, (bytes, int, float, bool, complex)):
        cls = type(obj)
        if cls is not type and cls.__module__ != "builtins":
            return f"sg.{cls.__qualname__}"
        return cls.__qualname__

    return type(obj).__qualname__


def _with_similar_help_names(
    obj: object, text: str, exclude: tuple[str, ...] | None = None
) -> str:
    """Append similar help names after an exact-match help string."""
    if text.startswith("No help entry named "):
        return text
    query = obj if isinstance(obj, str) else _doc_title(obj)
    query_canonical = _canonical_help_name(query).casefold()
    others = [
        name
        for name in _similar_help_names(query, exclude=exclude)
        if _canonical_help_name(name).casefold() != query_canonical
    ]
    if not others:
        return text
    lines = [text.rstrip(), "", "Similar names:"]
    lines.extend(f"  {name}" for name in others)
    return "\n".join(lines)


def _validated_exclude(
    exclude: str | Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Return normalized exclude needles, or raise if ``exclude`` is unusable."""
    if exclude is None:
        return None
    if isinstance(exclude, str):
        items: tuple[str, ...] = (exclude,)
    else:
        try:
            items = tuple(exclude)
        except TypeError:
            raise TypeError(
                "exclude must be a string or a sequence of strings, "
                f"got {type(exclude).__name__}"
            ) from None
    needles: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise TypeError(
                "exclude must be a string or a sequence of strings, "
                f"got {type(item).__name__}"
            )
        needle = normalize(item)
        if not needle:
            raise ValueError("exclude must be a non-empty string")
        needles.append(needle)
    if not needles:
        raise ValueError("exclude must be a non-empty string")
    return tuple(needles)


def help(
    obj: object,
    exact: bool = True,
    exclude: str | Sequence[str] | None = None,
) -> str:
    """Return documentation text for ``obj``.

    For string keys, returns ``defaults_help[obj]`` when ``obj`` is a
    defaults setting name (empty string if missing), or resolves public
    ``sg`` names such as ``Canvas.draw`` and ``Shape.translate``.
    Reserved topic
    strings (``points``, ``lines``, ``topics``, ``help``, …) return
    topic listings from ``d_help_topic``.     For classes (and instances of
    Simetri types), returns the constructor signature, class docstring,
    and ``__init__`` docstring.     For Simetri functions and methods,
    returns a signature with resolved ``defaults``, accepted ``**kwargs``
    when known, then the docstring.
    For other modules and objects, returns ``inspect.getdoc(obj)``.

    Args:
        obj: Object to document, a defaults setting name, or a help topic.
        exact: If True, return only the exact match. If False, also list
            similar names. Defaults to True.
        exclude: If set, omit listed names and topics that contain this
            string, or any string in a sequence of strings. Defaults to None.

    Returns:
        Documentation text, similar help names when the string is not a
        known topic, setting, or public ``sg`` name, or an empty string
        if none is available.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.help('topics').splitlines()[0]
        'Available help topics:'
        >>> 'sg.distance' in sg.help('points')
        True
        >>> 'shapes' in sg.help('shapess')
        True
        >>> 'Similar names:' in sg.help('angle')
        False
        >>> 'sg.line_angle' in sg.help('angle', exact=False)
        True
        >>> listed = [
        ...     line.strip()
        ...     for line in sg.help('angle', exact=False, exclude='triangle').splitlines()
        ...     if line.startswith('  sg.')
        ... ]
        >>> [name for name in listed if 'triangle' in name.casefold()]
        []
        >>> listed = [
        ...     line.strip()
        ...     for line in sg.help(
        ...         'angle', exact=False, exclude=('triangle', 'rectangle')
        ...     ).splitlines()
        ...     if line.startswith('  sg.')
        ... ]
        >>> [
        ...     name
        ...     for name in listed
        ...     if 'triangle' in name.casefold() or 'rectangle' in name.casefold()
        ... ]
        []
    """
    needles = _validated_exclude(exclude)
    text = _help_text(obj, exclude=needles)
    if exact:
        return text
    return _with_similar_help_names(obj, text, exclude=needles)


def doc(
    obj: object,
    exact: bool = True,
    exclude: str | Sequence[str] | None = None,
) -> None:
    """Print documentation text for ``obj``.

    Args:
        obj: Object to document, a defaults setting name, or a help topic.
        exact: If True, print only the exact match. If False, also print
            similar names. Defaults to True.
        exclude: If set, omit listed names and topics that contain this
            string, or any string in a sequence of strings. Defaults to None.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.doc('topics')  # doctest: +SKIP
        >>> 'sg.line_angle' in sg.help('angle', exact=False)
        True
        >>> listed = [
        ...     line.strip()
        ...     for line in sg.help(
        ...         'angle', exact=False, exclude=('triangle', 'rectangle')
        ...     ).splitlines()
        ...     if line.startswith('  sg.')
        ... ]
        >>> [
        ...     name
        ...     for name in listed
        ...     if 'triangle' in name.casefold() or 'rectangle' in name.casefold()
        ... ]
        []
"""
    title = _doc_title(obj)
    text = help(obj, exact=exact, exclude=exclude)
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
