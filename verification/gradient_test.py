import simetri.graphics as sg

canvas = sg.Canvas()

gradient = sg.Gradient(axis=((0, 0), (1, 1)))
rect = sg.Rectangle((0, 0), 200, 100, gradient=gradient)

canvas.draw(rect)
canvas.save("c:/tmp/gradient_verification_.pdf", overwrite=True)
canvas.save("c:/tmp/gradient_verification_.svg", overwrite=True)
_
canvas.display()
