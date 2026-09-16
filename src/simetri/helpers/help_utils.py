"""Interactive and topic-based help for Simetri (``sg.help``).

``sg.help(obj)`` accepts a string key or a callable/class/instance:

- String keys look up ``defaults_help[obj]`` (empty string if missing),
  except for reserved topic names described below. Unknown topic strings
  return similar topic names.
- Classes (and instances of Simetri types) return the constructor
  signature, class docstring, and ``__init__`` docstring.
- Functions, methods, modules, and other objects return
  ``inspect.getdoc(obj)``.

``d_help_topic`` maps topic names to short descriptions and lists of
related ``sg.*`` names. ``sg.help('help')`` loads the help-utilities
guide. ``sg.help(sg.help)`` summarizes how help lookup works.
``sg.help('topics')`` lists available topics.

Examples:
    >>> import simetri.graphics as sg
    >>> 'distance' in sg.help('points')
    True
    >>> 'Shape' in sg.help('shapes')
    True
"""

from __future__ import annotations

import inspect
import sys
import unicodedata
from collections.abc import Sequence
from enum import Enum
from functools import lru_cache
from pathlib import Path

from rapidfuzz.distance import DamerauLevenshtein

from ..base.all_enums import WarningType
from ..coloring import colors
from ..coloring.colors import Color
from ..config.settings import VOID, defaults, defaults_help

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


def normalize(word):
    return unicodedata.normalize("NFKC", word).casefold().strip()


def _resolve_help_suggestion_limit(limit: int | None) -> int:
    """Return the active similar-name suggestion limit."""
    if limit is None:
        return defaults["help_suggestion_limit"]
    return limit


def find_similar(query, words, threshold=0.75, limit: int | None = None):
    """Example:
    words = ["hello", "help", "yellow", "hero", "world"]
    print(find_similar("hlelo", words))
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
    "canvas": [
        "Canvas.draw",
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
        "sg.Dimension",
        "sg.AnnotationArrow",
        "sg.RadialDimension",
        "sg.AngularDimension",
        "sg.Anchor",
        "sg.defaults['gap']",
        "sg.defaults['overshoot']",
        "sg.defaults['text_offset']",
        "sg.defaults['aligned_text']",
        "sg.defaults['landing_length']",
        "sg.defaults['rev_arrow_length']",
        "See also: sg.help('canvas_doc'), sg.help('tags')",
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
            "sg.help('tag_objects')"
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
        "See also: sg.help('grids'), sg.help('lattices_doc')",
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
        "sg.user_config_path",
        "sg.set_user_settings_path",
        "sg.defaults",
        "sg.save_user_defaults",
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
    "Lattice": "lattices_doc",
    "lattice": "lattices_doc",
    "lattices": "lattices_doc",
    "lattices-doc": "lattices_doc",
    "lattice_p1": "lattices_doc",
    "Isometry": "lattices_doc",
    "LatType": "lattices_doc",
    "LatRef": "lattices_doc",
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
- sg.help(sg.distance)
  -> function docstring

Missing defaults keys return an empty string.
Unknown names list similar topics, settings, and public ``sg`` names.
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


def _topic_guide_text(topic: str) -> str | None:
    """Return topic-guide text when ``topic_guides/{topic}.qmd`` exists."""
    path = _TOPIC_GUIDES_DIR / f"{topic}.qmd"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _format_topics() -> str:
    """Return the sorted list of help topic names."""
    topics = sorted(
        set(d_help_topic) | set(_TOPIC_ALIASES.values()) | {"help", "topics"}
    )
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

    direct_matches = find_similar(query, names, limit=limit)
    for name, _score in direct_matches:
        add(name)
    if suggestions:
        return suggestions[:limit]

    exact_token_matches = []
    for name in names:
        if set(query_tokens).intersection(_help_name_tokens(name)):
            exact_token_matches.append(name)

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
    names = set(d_help_topic)
    names.update(_TOPIC_ALIASES)
    names.update(defaults.defaults)
    names.update(defaults_help)
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


def _similar_help_names(query: str, limit: int | None = None) -> list[str]:
    """Return similar help names, including matches via token pieces."""
    limit = _resolve_help_suggestion_limit(limit)
    lookup_names = _canonical_help_names()
    suggestions: list[str] = []
    seen: set[str] = set()
    query_tokens = tuple(sorted(_help_name_tokens(query)))
    normalized_query_tokens = tuple(normalize(token) for token in query_tokens)

    def sort_key(name: str) -> tuple[int, int, str]:
        return (0 if name.startswith("sg.") else 1, len(name), name)

    def add(name: str) -> None:
        canonical = _canonical_help_name(name)
        key = canonical.casefold()
        if key not in seen:
            seen.add(key)
            suggestions.append(canonical)

    direct_matches = find_similar(query, lookup_names, limit=limit)
    for name, _score in direct_matches:
        add(name)
    if suggestions:
        return suggestions[:limit]

    exact_token_matches = []
    for name in lookup_names:
        if set(query_tokens).intersection(_help_name_tokens(name)):
            exact_token_matches.append(name)

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


def _unknown_topic_help(query: str) -> str:
    """Return similar help names when ``query`` is not an exact match."""
    matches = _similar_help_names(query)
    if not matches:
        return (
            f"No help entry named {query!r}.\n"
            "Use sg.help('topics') for topics, or pass an sg object directly."
        )
    lines = [f"No help entry named {query!r}. Similar names:"]
    for name in matches:
        lines.append(f"  {name}")
    return "\n".join(lines)


def help(obj) -> str:
    """Return documentation text for ``obj``.

    For string keys, returns ``defaults_help[obj]`` when ``obj`` is a
    defaults setting name (empty string if missing), or resolves public
    ``sg`` names such as ``Canvas.draw`` and ``Shape.translate``.
    Reserved topic
    strings (``points``, ``lines``, ``topics``, ``help``, …) return
    topic listings from ``d_help_topic``. For classes (and instances of
    Simetri types), returns the constructor signature, class docstring,
    and ``__init__`` docstring. For functions, methods, modules, and
    other objects, returns ``inspect.getdoc(obj)``.

    Args:
        obj: Object to document, a defaults setting name, or a help topic.

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
    """
    if obj is help:
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
            guide = _topic_guide_text(topic)
            if guide is not None:
                return guide
            return _format_topic(topic, d_help_topic[topic])
        if obj in defaults.defaults:
            if obj in defaults_help:
                return defaults_help[obj]
            return ""
        named_obj = _named_help_objects().get(obj)
        if named_obj is not None:
            return help(named_obj)
        return _unknown_topic_help(obj)

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
        if obj.__module__.startswith("simetri."):
            return f"sg.{obj.__qualname__}"
        return obj.__qualname__

    if inspect.isroutine(obj):
        if obj.__module__.startswith("simetri."):
            return f"sg.{obj.__qualname__}"
        return obj.__qualname__

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
