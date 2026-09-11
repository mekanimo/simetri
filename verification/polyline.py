import simetri.graphics as sg

canvas = sg.Canvas()
seg = sg.Shape([(0, 0), (20, 20)])

polyline = seg.mirror(about=seg.right, reps=1).translate(40, 0, reps=5)
polyline = polyline.merge_shapes()
canvas.draw(polyline, draw_double=True, double_distance=5, line_width=3)
canvas.save("c:/tmp/polyline.svg", overwrite=True)
canvas.save("c:/tmp/polyline.pdf", overwrite=True)
