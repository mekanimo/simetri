import simetri.graphics as sg

canvas = sg.Canvas()
grid = sg.CircularGrid(n=10, radius=100)

p1 = grid.intersect((1, 7), (2, 8))
p2 = grid.intersect((0, 3), (1, 7))
p3 = grid.points[0]
kernel = sg.Shape([p1, p2, p3])
petal = kernel.mirror(sg.axis_x, reps=1)
star = petal.rotate(sg.pi / 5, reps=9)
swatch = sg.swatches_255[49]
lace = sg.Lace(star, offset=4, swatch=swatch)
canvas.draw(lace)
canvas.save("c:/tmp/broug_pg205_test.pdf", overwrite=True)
