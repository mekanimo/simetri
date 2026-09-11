import simetri.geom.polygons.polygon_utils
import simetri.graphics as sg

canvas = sg.Canvas()

verts = (
    (10.0, 30.0),
    (30.0, 30.0),
    (30.0, -10.0),
    (50.0, -10.0),
    (50.0, 10.0),
    (-10.0, 10.0),
    (-10.0, -10.0),
    (10.0, -10.0),
)

polygon = sg.Shape(verts, closed=True)

print(simetri.geom.polygons.polygon_utils.is_simple(verts))

canvas.draw(polygon)

canvas.save("c:/tmp/is_simple_test.svg", overwrite=True)
