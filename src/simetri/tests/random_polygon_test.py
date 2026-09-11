import simetri.graphics as sg

pi = sg.pi
d = 80

polygon = sg.random_polygon(
    n_min_edges=5,
    n_max_edges=12,
    max_x=d,
    max_y=d,
    min_angle=pi / 6,
    max_angle=3 * pi / 2,
    min_edge_length=5,
    max_edge_length=30,
)

canvas = sg.Canvas()

canvas.draw(polygon, fill=False, indices=True)
print(sg.polygon_internal_angles(polygon.vertices))
canvas.save("c:/tmp/random_polygon_test.svg", overwrite=True)
