import simetri.graphics as sg

canvas = sg.Canvas()

C = sg.Circle(50, (150, 0))
R = sg.Rectangle((100, 100), 40, 60, fill_color=sg.yellow)
F = sg.letter_F(fill=False, line_join=sg.LineJoin.ROUND, line_width=4)
E = sg.ellipse_shape(100, 60, pos=(200, 100), fill_color=sg.orange, fill_alpha=.2)
polyline = sg.Shape([(0, 0), (20, 40), (40, 0)], line_width=8, line_color=sg.acid_green, line_miter_limit=4, line_cap=sg.LineCap.ROUND).translate(50, 0, reps=5)
canvas.draw([C, R, F, E, polyline])

canvas.circle(2, (150, 0), line_color=sg.red, fill_color=sg.red)
canvas.circle(2, (100, 100), line_color=sg.red, fill_color=sg.red)

canvas.save("c:/tmp/svg_basic_shapes_test.svg", overwrite=True)