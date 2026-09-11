import random
import simetri.graphics as sg

canvas = sg.Canvas()

F = sg.letter_F(fill=False)

for mt in sg.MarkerType:
    canvas.draw(F, marker_type=mt, marker_size=5, draw_markers=True)
    canvas.text(mt, (-100, 0))
    canvas.translate(0, -120)


# canvas.save("c:/tmp/marker_test2.tex", overwrite=True)
canvas.save("c:/tmp/marker_test2_.pdf", overwrite=True)
canvas.save("c:/tmp/marker_test2_.svg", overwrite=True)
