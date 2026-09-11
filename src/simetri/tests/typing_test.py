import simetri.graphics as sg

# grid = sg.CircularGrid(n=20, radius=100)
# p1 = grid.intersect((0, 10), (2, 14))
# p2 = grid.intersect((0, 6), (2, 14))
# p3 = grid.points[0]
# n = 10
# kernel = sg.Shape([p1, p2, p3])
# petal = kernel.mirror(sg.axis_x, reps=1)
# star = petal.rotate(2 * sg.pi / n, reps=n - 1)
# lace = sg.Lace(star, offset=4)
# canvas = sg.Canvas()
# # canvas.draw(lace)
# shp = sg.reg_poly_shape(6)
# shp2 = shp.copy().translate(30, 30)

# # shp.color = sg.red
# shp.fill_color = sg.red
# shp2.fill_color = sg.green
# shp2.alpha = 0.5
# canvas.draw([shp, shp2], fill_color=sg.blue)

# canvas.save("c:/tmp/typing_tes.svg", overwrite=True)
d = 400
t1 = sg.Shape([(0, 0), (d, 0), (d / 2, d)], closed=True)

print(sg.triangle_centroid3(*t1))
