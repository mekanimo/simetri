import simetri.graphics as sg

canvas = sg.Canvas()

bs = sg.BackStyle

for i, back_style in enumerate([bs.COLOR, bs.PATTERN, bs.SHADING,]):
    x = i * 200
    combs = sg.product([True, False], repeat=3)
    for j, comb in enumerate(combs):
        fill, stroke, closed = comb
        F = sg.letter_F(fill_color = sg.red, line_width = 3,  fill=fill, stroke=stroke)
        F.back_style = back_style
        F.closed = closed
        y = 140 * j
        F.translate(x, y)
        text = 'fill={}, stroke={}, closed={}'.format(fill, stroke, closed)
        canvas.text(text, (x, y-20))
        canvas.draw(F)

# fill =True
# stroke = False
# F = sg.letter_F(fill_color = sg.red, line_width = 3,  fill=fill, stroke=stroke)
# F.closed = False
# F.back_style = bs.SHADING
# canvas.draw(F)
canvas.save('c:/tmp/shape_test_2.pdf', overwrite=True)
