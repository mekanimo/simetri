"""Vertex coordinate label drawing and overlap resolution."""

import simetri.graphics as sg
from simetri.helpers.illustration import (
    _coord_label_vertex_indices,
    _index_label_vertex_indices,
    _label_axis_overlaps,
    estimate_index_label_bbox,
    estimate_vertex_coord_label_bbox,
    prepare_shape_index_labels,
    prepare_shape_vertex_coord_labels,
)

hex = sg.reg_poly_shape(6, 200)
F = sg.letter_F()

index_offset = 4
vertex_offset = 12
canvas = sg.Canvas()
canvas.draw(
    F,
    vertices=True,
    indices=True,
    fill=True,
    index_offset=index_offset,
    vertex_offset=vertex_offset,
    vertex_font_size=6,
    index_font_size=6,
    vertex_on_hull=True,
)
canvas.save("c:/tmp/vertices_test.svg", overwrite=True)


class _Sketch:
    vertices = F.vertices
    indices = True
    show_vertex_coords = True
    vertex_on_hull = True
    index_offset = index_offset
    vertex_offset = vertex_offset
    vertex_font_size = 6
    index_font_size = 6


sketch = _Sketch()

index_draw = prepare_shape_index_labels(sketch)
assert index_draw is not None
index_positions, index_labels = index_draw

vertex_draw = prepare_shape_vertex_coord_labels(sketch)
assert vertex_draw is not None
coord_positions, coord_labels = vertex_draw

all_positions = []
all_sizes = []
for label, pos in zip(index_labels, index_positions):
    all_positions.append(pos)
    all_sizes.append(estimate_index_label_bbox(label, sketch.index_font_size))
for text, pos in zip(coord_labels, coord_positions):
    all_positions.append(pos)
    all_sizes.append(
        estimate_vertex_coord_label_bbox(text, sketch.vertex_font_size)
    )

for i in range(len(all_positions)):
    for j in range(i + 1, len(all_positions)):
        oh, ov = _label_axis_overlaps(
            all_positions[i], all_sizes[i], all_positions[j], all_sizes[j]
        )
        assert not (oh > 0 and ov > 0), (
            f"labels {i}/{j} overlap: h={oh:.1f} v={ov:.1f}"
        )

busy = sg.Shape([(0, 0), (100, 0), (100, 100), (0, 100), (50, 50)], closed=True)


class _BusySketch:
    vertices = busy.vertices
    show_vertex_coords = True
    indices = True
    index_offset = index_offset
    vertex_offset = vertex_offset
    vertex_font_size = 6
    index_font_size = 6


sketch_all = _BusySketch()
sketch_all.vertex_on_hull = False
assert len(_coord_label_vertex_indices(sketch_all, len(busy.vertices))) == 5
assert len(_index_label_vertex_indices(sketch_all, len(busy.vertices))) == 5

sketch_hull = _BusySketch()
sketch_hull.vertex_on_hull = True
assert len(_coord_label_vertex_indices(sketch_hull, len(busy.vertices))) == 4
assert len(_index_label_vertex_indices(sketch_hull, len(busy.vertices))) == 5

print(f"indices at index_offset={index_offset} pt from each vertex")
