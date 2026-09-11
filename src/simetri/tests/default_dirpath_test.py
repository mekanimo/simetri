import simetri.graphics as sg

grid = sg.CircularGrid(n=20, radius=100)
p1 = grid.intersect((0, 10), (2, 14))
p2 = grid.intersect((0, 6), (2, 14))
p3 = grid.points[0]

n = 10
kernel = sg.Shape([p1, p2, p3])
petal = kernel.mirror(sg.axis_x, reps=1)
star = petal.rotate(2 * sg.pi / n, reps=n - 1)

lace = sg.Lace(star, offset=4)


canvas = sg.Canvas()
canvas.draw(lace, fill=False)
canvas.save("config_test.svg")
