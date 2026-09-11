import random
import simetri.graphics as sg

canvas = sg.Canvas()


def rosette(factors, x=0, y=0, dx=60, dy=60, sf=0.5):
    # dots = sg.Dots(marker_type=sg.MarkerType.FCIRCLE, marker_size=radius, fill_color=sg.red,
    # line_color=sg.red)
    dots = sg.Group(sg.Shape([(0, 0)]))
    dots.translate(x + dx, y + dy)
    for n in factors:
        dots.rotate(sg.two_pi / n, (x, y), reps=n - 1)
        dots.scale(sf, about=dots.center)
        dots.translate(dx, dy)

    return dots


ns = [2, 3, 4, 5, 6]
set1 = set(sg.product(ns, ns, ns, ns))
# combinations = list(set1)


def grid(
    canvas, combinations, sf=0.5, width=160, height=160, columns=6, margin=80
):
    for i, combination in enumerate(combinations):
        row, col = i // columns, i % columns
        x, y = (col * width) + margin, (row * height) + margin
        canvas.draw(
            rosette(combination, x, y, sf=sf),
            marker_type=sg.MarkerType.FCIRCLE,
            marker_size=2,
            draw_markers=True,
            color=sg.red,
        )


n_pics = 24
# set2 = set(random.sample(combinations))
set2 = set(random.sample(list(set1), n_pics))
grid(canvas, set2)

canvas.help_lines()

dots = sg.Group(
    sg.Shape([(0, 0)], marker_type=sg.MarkerType.CIRCLE, marker_size=25)
)
canvas.draw(dots, draw_markers=True)

canvas.save("c:/tmp/marker_test_.pdf", overwrite=True)
canvas.save("c:/tmp/marker_test_.svg", overwrite=True)
