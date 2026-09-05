"""Shared save-time pre-render helpers.

The renderers now emit per-sketch styles directly.
Only real nested sketch containers such as clipping, masking,
and composites control nesting during save.
"""

from __future__ import annotations

from simetri.base.all_enums import BackStyle, FrameShape, MarkerType, Types


def collect_tikz_preamble_requirements_for_sketch(
    sketch, tikz_libraries, tikz_packages
):
    """Collect required TikZ libraries and TeX packages for a sketch."""
    sketch_dict = sketch.__dict__

    if sketch.subtype == Types.PATH_SKETCH:
        if "svg.path" not in tikz_libraries:
            tikz_libraries.append("svg.path")
    if "library" in sketch_dict and sketch.library == "fadings":
        if "fadings" not in tikz_libraries:
            tikz_libraries.append("fadings")
    if "_mask_stops" in sketch_dict and sketch._mask_stops is not None:
        if "fadings" not in tikz_libraries:
            tikz_libraries.append("fadings")
    if "draw_frame" in sketch_dict and sketch.draw_frame:
        if (
            "frame_shape" in sketch_dict
            and sketch.frame_shape != FrameShape.RECTANGLE
        ):
            if "shapes.geometric" not in tikz_libraries:
                tikz_libraries.append("shapes.geometric")
    if "draw_markers" in sketch_dict and sketch.draw_markers:
        if "patterns" not in tikz_libraries:
            tikz_libraries.append("patterns")
            tikz_libraries.append("patterns.meta")
            tikz_libraries.append("backgrounds")
            tikz_libraries.append("shadings")
            tikz_libraries.append("plotmarks")
    if "line_dash_array" in sketch_dict and sketch.line_dash_array:
        if "patterns" not in tikz_libraries:
            tikz_libraries.append("patterns")
    if sketch.subtype == Types.TAG_SKETCH:
        if "fontspec" not in tikz_packages:
            tikz_packages.append("fontspec")
    elif (
        "marker_type" in sketch_dict
        and sketch.marker_type == MarkerType.INDICES
    ):
        if "fontspec" not in tikz_packages:
            tikz_packages.append("fontspec")
    if "back_style" in sketch_dict:
        if sketch.back_style == BackStyle.COLOR:
            if "xcolor" not in tikz_packages:
                tikz_packages.append("xcolor")
        if sketch.back_style == BackStyle.SHADING:
            if "shadings" not in tikz_libraries:
                tikz_libraries.append("shadings")
        if sketch.back_style == BackStyle.PATTERN:
            if "patterns" not in tikz_libraries:
                tikz_libraries.append("patterns")
                tikz_libraries.append("patterns.meta")


def collect_tikz_preamble_requirements(canvas):
    """Collect required TikZ libraries and TeX packages for a canvas."""
    tikz_libraries = []
    tikz_packages = ["tikz", "pgf"]

    for page in canvas.pages:
        sketches_to_inspect = list(page.sketches)
        while sketches_to_inspect:
            sketch = sketches_to_inspect.pop()
            collect_tikz_preamble_requirements_for_sketch(
                sketch, tikz_libraries, tikz_packages
            )
            if sketch.subtype in (Types.CLIPPED_SKETCH, Types.MASKED_SKETCH):
                for sketch_list in sketch.sketches:
                    sketches_to_inspect.extend(sketch_list)

    return tikz_libraries, tikz_packages


def render_svg_scope_loop(
    sketches,
    ind,
    render_sketch_code,
):
    """Render SVG sketches directly without style scope groups."""
    code = []
    for sketch in sketches:
        sketch_code = render_sketch_code(sketch, ind, [])
        code.append(sketch_code)
    return "\n".join(code)


def render_tikz_scope_loop(
    sketches,
    ind,
    render_sketch_code,
    collect_requirements,
):
    """Render TikZ sketches directly without style scope groups."""
    code = []
    for sketch in sketches:
        if collect_requirements is not None:
            collect_requirements(sketch)
        sketch_code, ind = render_sketch_code(sketch, ind, [])
        code.append(sketch_code)
    return "".join(code), ind
