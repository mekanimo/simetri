import simetri.graphics as sg

text = sg.Tag("just a test", pos=(100, -50), font_size=24)

canvas = sg.Canvas()

canvas.draw(text)
F = sg.letter_F()
canvas.draw(F, fill=False)
canvas.save("c:/tmp/text_test.svg", overwrite=True)