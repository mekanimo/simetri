from typing import Union
import simetri.graphics as sg

canvas = sg.Canvas()


F = sg.letter_F()

canvas.draw(
    sg.Group(F), draw_double=True, double_distance=3, double_color=sg.yellow
)

canvas.translate(200, 0)
canvas.draw(F, draw_double=True, double_distance=3, double_color=sg.yellow)

canvas.translate(-200, -200)
F2 = F.copy()

canvas.draw(
    F2,
    draw_double=True,
    double_distance=3,
    double_color=sg.yellow,
    fillet_radius=6,
    draw_fillets=True,
)

canvas.save("double_line_test_.pdf", overwrite=True)
# canvas.save("double_line_test.tex", overwrite=True)
canvas.save("double_line_test_.svg", overwrite=True)
