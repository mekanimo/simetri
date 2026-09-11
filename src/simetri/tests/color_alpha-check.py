import simetri.graphics as sg

canvas = sg.Canvas()

shp = sg.letter_F()

# print(shp.color)
# print(shp.alpha)
# print(shp.line_color)
# print(shp.line_alpha)

shp.line_color = sg.red
shp.line_width = 3

# print("#############")
# print(shp.color)
# print(shp.alpha)
# print(shp.line_color)
# print(shp.line_alpha)
canvas.rotate(sg.pi / 4)
canvas.scale(0.5, 0.5)
canvas.draw(shp)
# sg.doc(sg.Shape)
# sg.doc(sg.Canvas)
sg.doc(sg.Canvas.draw)
canvas.save("c:/tmp/color_alpha_check.svg", overwrite=True)
