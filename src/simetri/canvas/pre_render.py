"""Shared save-time pre-render analysis helpers.

Scoping Algorithm Summary:
1. Validate scoping invariants before grouping:
        - non-scopable style keys cannot appear in scope keys or suppressed keys.
2. Build automatic scope groups for eligible sketches:
        - resolve style values,
        - build style signatures,
        - group sketches by matching signatures,
        - keep only groups with more than one sketch.
3. Build manual+automatic scope open/close maps:
        - merge adjacent scope groups with identical style data,
        - map first sketch ids to scope opens,
        - map last sketch ids to scope closes.
4. Build neutral style-pass mappings:
        - styles dictionary ({style_id: style_obj}),
        - style-to-sketch dictionary ({style_id: [sketch_id, ...]}),
        - reverse sketch-to-style ids mapping.
5. Validate coverage:
        - every sketch in the render pass must map to at least one style id.
6. Render loops consume these maps:
        - open scopes, render sketch with suppressed scoped keys,
        - close scopes,
        - use pass mappings for consistency checks.
"""

from __future__ import annotations

import copy

from simetri.graphics.all_enums import (
    BackStyle,
    FrameShape,
    MarkerType,
    Types,
    FillMode,
    LineCap,
    LineJoin,
)
from simetri.graphics.sketch import ScopeGroup
from simetri.graphics.common import d_id_obj
from simetri.colors.colors import black, white


style_properties = [
    "draw_double",
    "double_distance",
    "double_color",
    "closed",
    "fill",
    "fill_alpha",
    "fill_color",
    "fill_mode",
    "line_alpha",
    "line_cap",
    "line_color",
    "line_dash_array",
    "line_dash_phase",
    "line_join",
    "line_miter_limit",
    "line_width",
    "draw_fillets",
    "fillet_radius",
    "stroke",
]

default_values = {
    "draw_double": False,
    "draw_fillets": False,
    "double_color": white,
    "double_distance": 2,
    "fill": True,
    "fill_color": black,
    "fill_alpha": 1,
    "fill_mode": FillMode.EVENODD,
    "line_alpha": 1,
    "line_cap": LineCap.BUTT,
    "line_color": black,
    "line_join": LineJoin.MITER,
    "line_miter_limit": 10,
    "line_dash_phase": 0,
    "line_width": 1,
    "stroke": True,
}


def set_styles(sketches):
    d_styles = {}
    d_sketch_style = {}
    signature_to_style_id = {}
    style_index = 1

    for sketch in sketches:
        sketch_dict = sketch.__dict__
        style = {}
        signature = []

        for prop in style_properties:
            if prop in sketch_dict:
                value = sketch_dict[prop]
            else:
                value = None

            if prop in default_values:
                default_value = default_values[prop]
                if value is None or value == default_value:
                    continue
            elif value is None:
                continue

            style[prop] = value
            signature.append((prop, repr(value)))

        if not signature:
            continue

        style_signature = tuple(signature)
        if style_signature not in signature_to_style_id:
            style_id = f"style{style_index}"
            style_index += 1
            signature_to_style_id[style_signature] = style_id
            d_styles[style_id] = style

        d_sketch_style[sketch.id] = signature_to_style_id[style_signature]

    return d_styles, d_sketch_style


# The scoping algorithm has been a big failure. It created very fragile states that failed with any change or additions. Now I am changing the way common styles are going to be handled. No more scoping/grouping for styles.  In pre_render.py set_styles creates dictionaries of all unique styles and maps them to the sketches. The following section shows how to use these styles without scopes. Some operations may still need scopes (probably masking but not sure). If there is any of thoses scopes then we need to remove those sketches prior to calling the set_styles function. This probably includes the composite sketches too but not sure.

# If the line_width is 1 and line_color and/or fill_color is black we don't need to use this values since they are the defaults for both svg and Tikz.

# \tikzset{
#   style1/.style={
#     draw=red!70!black,
#     fill=orange!30,
#     line width=2pt
#   },
#   style2/.style={
#     draw=blue!70!black,
#     fill=cyan!20,
#     line width=2pt
#   }
# }

# \draw[style1] (0,0) rectangle (2,1);
# \draw[style2] (3,0) circle [radius=0.5];

# <svg width="500" height="120" viewBox="0 0 500 120"
#      xmlns="http://www.w3.org/2000/svg">

#   <style>
#     .style1 {
#       stroke: #991b1b;
#       stroke-width: 2;
#       fill: #fdba74;
#     }

#     .style2 {
#       stroke: #1d4ed8;
#       stroke-width: 2;
#       fill: #bfdbfe;
#     }
#   </style>

#   <rect class="style1" x="20" y="20" width="100" height="60" />
#   <circle class="style2" cx="200" cy="50" r="30" />
#   <polygon class="style1" points="300,80 340,20 380,80" />
# </svg>


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
            if sketch.subtype in [Types.CLIPPED_SKETCH, Types.MASKED_SKETCH]:
                for sketch_list in sketch.sketches:
                    sketches_to_inspect.extend(sketch_list)

    return tikz_libraries, tikz_packages
