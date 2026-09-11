from typing import Union
import simetri.graphics as sg

styles = [
    {"line_width": 4, "line_color": sg.blue, "fill_color": sg.yellow},
    {
        "line_width": 2,
        "line_color": sg.red,
        "fillet_radius": 5,
        "draw_fillets": True,
        "fill_color": "light_cyan",
    },
    {
        "line_width": 2,
        "line_color": sg.navy,
        "line_dash_array": [5, 2],
        "fill_color": "#00FF00",
    },
    {"draw_double": True, "double_distance": 3, "fill_color": sg.light_cyan},
]

canvas = sg.Canvas()
# canvas.back_color = sg.yellow
seg = sg.Shape([(0, 0), (20, 20)])

polygon = sg.reg_star_polygon(8, 3, rad=40)
canvas.draw(polygon)
canvas.translate(0, -50)

for style in styles:
    for key, value in style.items():
        canvas.text(f"{key}: {value}", (0, 0))
        canvas.translate(0, -20)
    canvas.translate(0, -40)
    canvas.draw(sg.Group(polygon), **style)
    # canvas.draw(polygon, **style)
    canvas.translate(0, -70)

canvas.save("polygon2_2.pdf", overwrite=True)
# canvas.save("polygon2_2.tex", overwrite=True)
canvas.save("polygon2_2.svg", overwrite=True)
