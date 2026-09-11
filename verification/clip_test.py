import simetri.graphics as sg

canvas = sg.Canvas(back_color=sg.light_gold)

gradient = sg.Gradient(axis=((0, 0), (1, 1)))
rect = sg.Rectangle((0, 0), 200, 150, gradient=gradient)
F = sg.letter_F()
canvas.clip(rect, F)
# canvas.draw([rect, F])
# canvas.save("c:/tmp/clip_verification_.svg", overwrite=True)
canvas.save("c:/tmp/clip_verification_.pdf", overwrite=True)

# canvas.display()
