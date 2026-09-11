import simetri.graphics as sg

edges = sg.Group(
    [
        sg.Shape([(12.5, -12.5), (37.5, -12.5)]),
        sg.Shape([(37.5, -12.5), (37.5, 12.5)]),
        sg.Shape([(37.5, 12.5), (12.5, 12.5)]),
        sg.Shape([(12.5, 12.5), (12.5, -12.5)]),
        sg.Shape([(37.5, -12.5), (62.5, -12.5)]),
        sg.Shape([(62.5, -12.5), (62.5, 12.5)]),
        sg.Shape([(62.5, 12.5), (37.5, 12.5)]),
        sg.Shape([(37.5, 12.5), (37.5, -12.5)]),
        sg.Shape([(-12.5, 12.5), (12.5, 12.5)]),
        sg.Shape([(12.5, 12.5), (12.5, 37.5)]),
        sg.Shape([(12.5, 37.5), (-12.5, 37.5)]),
        sg.Shape([(-12.5, 37.5), (-12.5, 12.5)]),
        sg.Shape([(37.5, 12.5), (62.5, 12.5)]),
        sg.Shape([(62.5, 12.5), (62.5, 37.5)]),
        sg.Shape([(62.5, 37.5), (37.5, 37.5)]),
        sg.Shape([(37.5, 37.5), (37.5, 12.5)]),
    ]
)

canvas = sg.Canvas()
canvas.draw(edges, fill=False, indices=True)
polygon = edges.merge_shapes(remove_duplicate_edges=True)
canvas.translate(100, 0)
canvas.draw(polygon, fill=False, remove_duplicate_edges=True, indices=True)

canvas.save("c:/tmp/merge_no_dup_edges.svg", overwrite=True)
