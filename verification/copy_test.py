from typing import Union
import simetri.graphics as sg

canvas = sg.Canvas()


F = sg.letter_F()
F.line_dash_array = [3, 3]
F.fill_color = sg.light_eggplant

circ = sg.Circle(10, (20, 100), fill=False, line_width=3)

rect = sg.Rectangle((-50, -50), 80, 60, fill=False)

canvas.draw(F.copy())
canvas.draw(circ.copy())
canvas.draw(rect.copy())

canvas.save("copy_test_.pdf", overwrite=True)
# canvas.save("copy_test.tex", overwrite=True)
canvas.save("copy_test_.svg", overwrite=True)
