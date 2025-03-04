import os

import simetri.graphics as sg


dir_path = "c:/tmp/simetri_test_dir/"
file_name = "simetri_shape_test_2_"
file_path = os.path.join(dir_path, file_name)
extensions = [".svg", ".png", ".pdf", ".ps", ".eps", ".tex"]

canvas = sg.Canvas()

bs = sg.BackStyle

for i, back_style in enumerate(
    [
        bs.COLOR,
        bs.PATTERN,
        bs.SHADING,
    ]
):
    x = i * 200
    combs = sg.product([True, False], repeat=3)
    for j, comb in enumerate(combs):
        fill, stroke, closed = comb
        F = sg.letter_F(fill_color=sg.red, line_width=3, fill=fill, stroke=stroke)
        F.back_style = back_style
        F.closed = closed
        y = 140 * j
        F.translate(x, y)
        text = "fill={}, stroke={}, closed={}".format(fill, stroke, closed)
        canvas.text(text, (x, y - 20))
        canvas.draw(F)


for ext in extensions:
    canvas.save(f"{file_path}{ext}", show=False, overwrite=True)
