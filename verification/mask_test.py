import simetri.geom.segments.line_utils
import simetri.graphics as sg

gap = 150
a = 20
a = a
b = a / 2
rect = sg.Shape(
    [(0.0, 75.0), (0.0, -75.0), (550.0, -75.0), (550.0, 75.0)],
    closed=True,
    fill=False,
)
canvas = sg.Canvas()
F = sg.letter_F()
F.mirror(F.left).rotate(-sg.pi / 2, about=F.southeast)
pattern = F.glide(
    simetri.geom.segments.line_utils.offset_line(F.bottom, -b),
    F.width,
    reps=5,
)
canvas.draw(
    sg.Shape(
        [(0.0, 75.0), (0.0, -75.0), (600.0, -75.0), (600.0, 75.0)],
        closed=True,
        color=sg.light_aqua,
    )
)
canvas.draw(pattern, mask=rect)

canvas.save("c:/tmp/mask_test_2_.pdf", overwrite=True)
canvas.save("c:/tmp/mask_test_2_.svg", overwrite=True)
