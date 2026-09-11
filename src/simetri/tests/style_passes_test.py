import simetri.graphics as sg
from simetri.render.pre_render import (
    validate_scope_style_keys,
    validate_suppressed_style_keys,
)
from simetri.render.style_passes import (
    NON_SCOPABLE_SCOPE_KEYS,
    build_sketch_style_ids,
    build_style_sketch_dict,
    build_styles_dict,
    validate_style_sketch_coverage,
)

canvas = sg.Canvas()
shape_1 = sg.Circle(20, (0, 0))
shape_2 = sg.Circle(20, (50, 0))
shape_2.line_color = sg.blue
canvas.draw(shape_1)
canvas.draw(shape_2)

sketches = canvas.active_page.sketches

style_domain_key_sets = {
    "scope": [
        "back_style",
        "blend_mode",
        "fill",
        "fill_alpha",
        "fill_color",
        "fillet_radius",
        "line_alpha",
        "line_cap",
        "line_color",
        "line_dash_array",
        "line_join",
        "line_miter_limit",
        "line_width",
        "smooth",
        "stroke",
    ]
}

styles_dict = build_styles_dict(sketches, style_domain_key_sets)
style_sketch_dict = build_style_sketch_dict(sketches, style_domain_key_sets, styles_dict)
sketch_style_ids = build_sketch_style_ids(style_sketch_dict)

assert styles_dict
assert style_sketch_dict
assert sketch_style_ids
validate_style_sketch_coverage(sketches, sketch_style_ids)

for sketch in sketches:
    assert sketch.id in sketch_style_ids

bad_scope_keys = ["line_color", "draw_double"]
raised = False
try:
    validate_scope_style_keys(bad_scope_keys)
except ValueError:
    raised = True
assert raised

bad_suppressed_keys = ["line_width", "double_distance"]
raised = False
try:
    validate_suppressed_style_keys(bad_suppressed_keys)
except ValueError:
    raised = True
assert raised

for key in ["draw_double", "double_color", "double_distance"]:
    assert key in NON_SCOPABLE_SCOPE_KEYS

print("style_passes_test: ok")
