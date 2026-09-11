import simetri.graphics as sg

canvas = sg.Canvas()

F = sg.Circle(50, (150, 0))
F.line_dash_array = [2, 4]
F.fill_color = sg.yellow
batch = F.translate(100, 0, 3)
batch[2].fill_color=sg.navy
batch.rotate(sg.pi/4, reps=7)

canvas.draw(batch)
canvas.back_color = sg.light_blue
canvas.border = 50

print(sg.get_styles_dict(canvas))
canvas.save("c:/tmp/canvas_save_test3.svg", overwrite=True)