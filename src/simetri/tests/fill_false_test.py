import simetri.graphics as sg

canvas = sg.Canvas(border=100)

F = sg.letter_F()
F.color = sg.red
F.fill_color = sg.blue
canvas.draw(F, color=sg.green)
grp = sg.Group(F)
grp.line_color = sg.orange

canvas.save("c:/tmp/fill_false_test.pdf", overwrite=True)
