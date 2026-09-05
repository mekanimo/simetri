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

from simetri.base.all_enums import BackStyle, FrameShape, MarkerType, Types
from simetri.render.sketch import ScopeGroup
from simetri.base.common import d_id_obj
from .style_passes import (
    NON_SCOPABLE_SCOPE_KEYS,
    build_sketch_style_ids,
    build_style_sketch_dict,
    build_styles_dict,
    create_style_signature,
    resolve_style_value,
    validate_style_sketch_coverage,
)


def validate_scope_style_keys(scope_style_keys):
    """Validate scope keys against non-scopable style keys.

    Raises:
            ValueError: If scope_style_keys contains non-scopable keys.
    """
    overlap = set(scope_style_keys) & NON_SCOPABLE_SCOPE_KEYS
    if overlap:
        raise ValueError(
            "scope_style_keys cannot include non-scopable keys: "
            f"{sorted(overlap)}"
        )


def validate_suppressed_style_keys(suppressed_style_keys):
    """Validate suppressed keys against non-scopable style keys.

    Raises:
            ValueError: If suppressed_style_keys contains non-scopable keys.
    """
    overlap = set(suppressed_style_keys) & NON_SCOPABLE_SCOPE_KEYS
    if overlap:
        raise ValueError(
            "suppressed style keys cannot include non-scopable keys: "
            f"{sorted(overlap)}"
        )


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


def _is_meaningful_scope(sketch_style_data):
    for style_key in sketch_style_data:
        if style_key not in ("fill", "stroke"):
            return True
    return False


def prepare_auto_scope_groups(
    sketches,
    scope_style_keys,
    excluded_subtypes,
    collect_requirements=None,
    is_eligible=None,
):
    """Prepare automatic scope-group metadata for a sketch sequence."""
    validate_scope_style_keys(scope_style_keys)
    auto_scope_group_by_sketch_id = {}
    auto_scope_style_keys_by_sketch_id = {}
    style_groups = {}
    style_group_sketch_ids = {}

    for sketch in sketches:
        sketch_id = sketch.id
        if collect_requirements is not None:
            collect_requirements(sketch)

        sketch_dict = sketch.__dict__
        eligible = sketch.subtype not in excluded_subtypes
        if eligible and is_eligible is not None:
            eligible = is_eligible(sketch)

        if eligible:
            sketch_style_data = {}
            for style_key in scope_style_keys:
                sketch_style_data[style_key] = resolve_style_value(
                    sketch_dict, style_key
                )

            if _is_meaningful_scope(sketch_style_data):
                signature = create_style_signature(
                    sketch_style_data, scope_style_keys
                )
                if signature not in style_groups:
                    style_groups[signature] = ScopeGroup(
                        label="",
                        subtype=Types.SCOPE_GROUP,
                        sketch_list=[],
                        style_data=sketch_style_data,
                    )
                    style_group_sketch_ids[signature] = []
                style_groups[signature].sketch_list.append(sketch)
                style_group_sketch_ids[signature].append(sketch_id)

    for signature, scope_group in style_groups.items():
        grouped_sketch_ids = style_group_sketch_ids[signature]
        if len(grouped_sketch_ids) > 1:
            scope_keys = list(scope_group.style_data.keys())
            for sketch_id in grouped_sketch_ids:
                auto_scope_group_by_sketch_id[sketch_id] = scope_group
                auto_scope_style_keys_by_sketch_id[sketch_id] = scope_keys

    return auto_scope_group_by_sketch_id, auto_scope_style_keys_by_sketch_id


def build_scope_group_maps(sketches, scope_groups):
    """Build first-sketch open and last-sketch close maps for scope groups."""
    scope_opens = {}
    scope_closes = {}

    merged_scope_groups = []
    for scope_group in scope_groups:
        if not merged_scope_groups:
            merged_scope_groups.append(scope_group)
            continue
        previous_group = merged_scope_groups[-1]
        if (
            previous_group.subtype == Types.SCOPE_GROUP
            and scope_group.subtype == Types.SCOPE_GROUP
            and previous_group.style_data == scope_group.style_data
            and previous_group.sketch_list
            and scope_group.sketch_list
            and previous_group.sketch_list[-1].id + 1
            == scope_group.sketch_list[0].id
        ):
            previous_group.sketch_list.extend(scope_group.sketch_list)
        else:
            merged_scope_groups.append(scope_group)

    for scope_group in merged_scope_groups:
        if scope_group.sketch_list:
            first_sketch_id = scope_group.sketch_list[0].id
            last_sketch_id = scope_group.sketch_list[-1].id
            if first_sketch_id not in scope_opens:
                scope_opens[first_sketch_id] = []
            scope_opens[first_sketch_id].append(scope_group)
            if last_sketch_id not in scope_closes:
                scope_closes[last_sketch_id] = []
            scope_closes[last_sketch_id].append(scope_group)

    return scope_opens, scope_closes


def _scope_sketch_from_source(source_sketch):
    """Create a shallow sketch copy for scope-style evaluation."""
    return copy.copy(source_sketch)


def _collect_scope_style_keys_by_sketch_id(scope_groups):
    """Map sketch ids to the style keys provided by explicit scope groups."""
    scope_style_keys_by_sketch_id = {}
    for scope_group in scope_groups:
        if scope_group.subtype != Types.SCOPE_GROUP:
            continue
        style_keys = list(scope_group.style_data.keys())
        for sketch in scope_group.sketch_list:
            sketch_id = sketch.id
            if sketch_id not in scope_style_keys_by_sketch_id:
                scope_style_keys_by_sketch_id[sketch_id] = []
            for style_key in style_keys:
                if style_key not in scope_style_keys_by_sketch_id[sketch_id]:
                    scope_style_keys_by_sketch_id[sketch_id].append(style_key)
    return scope_style_keys_by_sketch_id


def render_svg_scope_loop(
    sketches,
    ind,
    scope_groups,
    scope_style_keys,
    excluded_subtypes,
    render_sketch_code,
    format_manual_scope_open,
    format_auto_scope_open,
    is_scope_eligible,
):
    """Render SVG sketches with a single scope/group orchestration loop."""
    style_domain_key_sets = {"scope": scope_style_keys}
    styles_dict = build_styles_dict(sketches, style_domain_key_sets)
    style_sketch_dict = build_style_sketch_dict(
        sketches, style_domain_key_sets, styles_dict
    )
    sketch_style_ids = build_sketch_style_ids(style_sketch_dict)
    validate_style_sketch_coverage(sketches, sketch_style_ids)

    auto_scope_group_by_sketch_id, auto_scope_style_keys_by_sketch_id = (
        prepare_auto_scope_groups(
            sketches,
            scope_style_keys,
            excluded_subtypes,
            is_eligible=is_scope_eligible,
        )
    )
    manual_scope_style_keys_by_sketch_id = (
        _collect_scope_style_keys_by_sketch_id(scope_groups)
    )

    code = []
    scope_opens, scope_closes = build_scope_group_maps(sketches, scope_groups)
    for sketch_index, sketch in enumerate(sketches):
        sketch_id = sketch.id
        if sketch_id not in sketch_style_ids:
            raise ValueError(
                f"Sketch {sketch_id} missing style mapping during SVG render loop"
            )
        if sketch_id in scope_opens:
            for scope_group in scope_opens[sketch_id]:
                scope_source_sketch = scope_group.sketch_list[0]
                scope_sketch = _scope_sketch_from_source(scope_source_sketch)
                scope_open_tag = format_manual_scope_open(
                    scope_group, scope_sketch
                )
                if scope_open_tag:
                    code.append(scope_open_tag)

        current_auto_scope_group = None
        if sketch_id in auto_scope_group_by_sketch_id:
            current_auto_scope_group = auto_scope_group_by_sketch_id[sketch_id]
        previous_auto_scope_group = None
        if sketch_index > 0:
            previous_sketch = sketches[sketch_index - 1]
            previous_sketch_id = previous_sketch.id
            if previous_sketch_id in auto_scope_group_by_sketch_id:
                previous_auto_scope_group = auto_scope_group_by_sketch_id[
                    previous_sketch_id
                ]
        if (
            current_auto_scope_group is not None
            and current_auto_scope_group is not previous_auto_scope_group
        ):
            auto_scope_sketch = _scope_sketch_from_source(sketch)
            auto_scope_open_tag = format_auto_scope_open(
                current_auto_scope_group, auto_scope_sketch
            )
            if auto_scope_open_tag:
                code.append(auto_scope_open_tag)

        suppressed_style_keys = []
        if sketch_id in manual_scope_style_keys_by_sketch_id:
            for style_key in manual_scope_style_keys_by_sketch_id[sketch_id]:
                if style_key not in suppressed_style_keys:
                    suppressed_style_keys.append(style_key)
        if sketch_id in auto_scope_style_keys_by_sketch_id:
            for style_key in auto_scope_style_keys_by_sketch_id[sketch_id]:
                if style_key not in suppressed_style_keys:
                    suppressed_style_keys.append(style_key)

        validate_suppressed_style_keys(suppressed_style_keys)

        sketch_code = render_sketch_code(sketch, ind, suppressed_style_keys)
        code.append(sketch_code)

        next_auto_scope_group = None
        if sketch_index + 1 < len(sketches):
            next_sketch = sketches[sketch_index + 1]
            next_sketch_id = next_sketch.id
            if next_sketch_id in auto_scope_group_by_sketch_id:
                next_auto_scope_group = auto_scope_group_by_sketch_id[
                    next_sketch_id
                ]
        if (
            current_auto_scope_group is not None
            and current_auto_scope_group is not next_auto_scope_group
        ):
            code.append("</g>")

        if sketch_id in scope_closes:
            for _ in scope_closes[sketch_id]:
                code.append("</g>")

    return "\n".join(code)


def render_tikz_scope_loop(
    sketches,
    ind,
    scope_groups,
    scope_style_keys,
    excluded_subtypes,
    render_sketch_code,
    format_manual_scope_open,
    format_auto_scope_open,
    is_scope_eligible,
    collect_requirements,
):
    """Render TikZ sketches with a single scope/group orchestration loop."""
    style_domain_key_sets = {"scope": scope_style_keys}
    styles_dict = build_styles_dict(sketches, style_domain_key_sets)
    style_sketch_dict = build_style_sketch_dict(
        sketches, style_domain_key_sets, styles_dict
    )
    sketch_style_ids = build_sketch_style_ids(style_sketch_dict)
    validate_style_sketch_coverage(sketches, sketch_style_ids)

    auto_scope_group_by_sketch_id, auto_scope_style_keys_by_sketch_id = (
        prepare_auto_scope_groups(
            sketches,
            scope_style_keys,
            excluded_subtypes,
            collect_requirements=collect_requirements,
            is_eligible=is_scope_eligible,
        )
    )
    manual_scope_style_keys_by_sketch_id = (
        _collect_scope_style_keys_by_sketch_id(scope_groups)
    )

    code = []
    scope_opens, scope_closes = build_scope_group_maps(sketches, scope_groups)
    for sketch_index, sketch in enumerate(sketches):
        sketch_id = sketch.id
        if sketch_id not in sketch_style_ids:
            raise ValueError(
                f"Sketch {sketch_id} missing style mapping during TikZ render loop"
            )
        if sketch_id in scope_opens:
            for scope_group in scope_opens[sketch_id]:
                scope_source_sketch = scope_group.sketch_list[0]
                scope_sketch = _scope_sketch_from_source(scope_source_sketch)
                scope_open_code = format_manual_scope_open(
                    scope_group, scope_sketch
                )
                if scope_open_code:
                    code.append(scope_open_code)

        current_auto_scope_group = None
        if sketch_id in auto_scope_group_by_sketch_id:
            current_auto_scope_group = auto_scope_group_by_sketch_id[sketch_id]
        previous_auto_scope_group = None
        if sketch_index > 0:
            previous_sketch = sketches[sketch_index - 1]
            previous_sketch_id = previous_sketch.id
            if previous_sketch_id in auto_scope_group_by_sketch_id:
                previous_auto_scope_group = auto_scope_group_by_sketch_id[
                    previous_sketch_id
                ]
        if (
            current_auto_scope_group is not None
            and current_auto_scope_group is not previous_auto_scope_group
        ):
            auto_scope_sketch = _scope_sketch_from_source(sketch)
            auto_scope_open_code = format_auto_scope_open(
                current_auto_scope_group, auto_scope_sketch
            )
            if auto_scope_open_code:
                code.append(auto_scope_open_code)

        suppressed_style_keys = []
        if sketch_id in manual_scope_style_keys_by_sketch_id:
            for style_key in manual_scope_style_keys_by_sketch_id[sketch_id]:
                if style_key not in suppressed_style_keys:
                    suppressed_style_keys.append(style_key)
        if sketch_id in auto_scope_style_keys_by_sketch_id:
            for style_key in auto_scope_style_keys_by_sketch_id[sketch_id]:
                if style_key not in suppressed_style_keys:
                    suppressed_style_keys.append(style_key)

        validate_suppressed_style_keys(suppressed_style_keys)

        sketch_code, ind = render_sketch_code(
            sketch, ind, suppressed_style_keys
        )
        code.append(sketch_code)

        next_auto_scope_group = None
        if sketch_index + 1 < len(sketches):
            next_sketch = sketches[sketch_index + 1]
            next_sketch_id = next_sketch.id
            if next_sketch_id in auto_scope_group_by_sketch_id:
                next_auto_scope_group = auto_scope_group_by_sketch_id[
                    next_sketch_id
                ]
        if (
            current_auto_scope_group is not None
            and current_auto_scope_group is not next_auto_scope_group
        ):
            code.append("\\end{scope}")

        if sketch_id in scope_closes:
            for _ in scope_closes[sketch_id]:
                code.append("\\end{scope}")

    return "".join(code), ind
