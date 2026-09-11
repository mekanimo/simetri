import simetri.graphics as sg

canvas = sg.Canvas()

d = 10
kernel = sg.Shape(
    [
        (0, d),
        (0, 6 * d),
        (2 * d, 6 * d),
        (2 * d, 2 * d),
        (-2 * d, 2 * d),
        (-2 * d, 4 * d),
        (3 * d, 4 * d),
    ]
)

braid = kernel.rotate(sg.pi, about=(0, d), reps=1)
braid.mirror(braid.right, reps=1)
braid.translate(braid.width, 0, reps=2)
canvas.circle(2, (braid.width, -4 * d))
# braid.rotate(sg.pi / 2, about=(braid.width - 2 * d, -3 * d), reps=1)

lace = sg.Lace(braid.merge_shapes(), offset=5)
canvas.draw(lace)
# canvas.draw(braid)

canvas.save("c:/tmp/braided_border.pdf", overwrite=True)
