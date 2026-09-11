import simetri.graphics as sg
from simetri.helpers.illustration import Tag

labels = [
    "line_width: 4",
    "fill_color: yellow",
    "line_dash_array: [5, 2]",
    "draw_fillets: True",
    "double_distance: 3",
]

canvas = sg.Canvas()

for label in labels:
    # Draw the text label
    canvas.text(label, (0, 0))

    # Compute the Tag bounding box in local canvas space (no xform),
    # then canvas.draw() applies the current transform — same as canvas.text()
    tag = Tag(label, (0, 0))
    canvas.draw(tag.b_box, line_color=sg.red, fill=False)

    canvas.translate(0, -30)

canvas.save("tag_bbox_test.svg", overwrite=True)
canvas.save("tag_bbox_test.pdf", overwrite=True)
