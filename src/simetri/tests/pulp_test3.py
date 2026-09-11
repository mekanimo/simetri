import simetri.graphics as sg
from simetri.helpers.illustration import (
    estimate_vertex_coord_label_bbox,
    format_vertex_coord,
    vert_label_layout,
)

Box = sg.Box


def get_vertex_label_bboxes(
    shape, vertex_offset=10, vertex_font_size=6
) -> list[Box]:
    """Return raw vertex coordinate label bboxes at ``vertex_offset``.

    Returns a list of ``Box`` objects (one per vertex). Each box uses center
    ``(x, y)`` plus ``width`` and ``height``. No overlap resolution.
    """
    layout = vert_label_layout(shape, vertex_offset)
    vertices = shape.vertices
    return [
        Box(
            *item["position"],
            *estimate_vertex_coord_label_bbox(
                format_vertex_coord(*vertices[i]), vertex_font_size
            ),
        )
        for i, item in enumerate(layout)
    ]


F = sg.letter_F()
boxes = get_vertex_label_bboxes(F, vertex_offset=4, vertex_font_size=6)

print("Initial centers:")
for i, box in enumerate(boxes):
    print(f"  {i}: (x={box.x}, y={box.y}), w={box.width}, h={box.height}")


# final_centers = push_boxes_apart(boxes, min_sep=0.0, bigM=1000)
final_centers = sg.push_boxes_apart(
    boxes,
    min_sep=3,  # required gap between rectangles
    bigM=1000,
)

canvas = sg.Canvas()

for box in boxes:
    canvas.rectangle(
        center=(box.x, box.y), width=box.width, height=box.height, fill=False
    )

canvas.translate(0, -200)


print("\nFinal centers (non-overlapping, minimal L1 push):")
for i, (xf, yf) in enumerate(final_centers):
    dx = xf - boxes[i].x
    dy = yf - boxes[i].y
    print(f"  {i}: (x={xf:.3f}, y={yf:.3f}), moved (dx={dx:.3f}, dy={dy:.3f})")
    box = boxes[i]
    canvas.rectangle(
        center=(xf, yf), width=box.width, height=box.height, fill=False
    )

canvas.save("c:/tmp/overlapping_labels_test.svg", overwrite=True)
# if __name__ == "__main__":
#     boxes = get_vertex_label_bboxes(F, vertex_offset=4, vertex_font_size=6)
#     for i, box in enumerate(boxes):
#         print(f"v{i}: center=({box.x:.2f}, {box.y:.2f}) w={box.width:.2f} h={box.height:.2f}")
