import cProfile
import pstats

import simetri.graphics as sg

canvas = sg.Canvas()

points = [(0, 0), (40, 0), (40, 40), (0, 40)]
square = sg.Shape(points, closed=True)
n = 6
squares = square.translate(20, 30, reps=1).translate(50, 0, reps=n)
squares.translate(-10, 50, reps=n)


# Profile the lace creation
with cProfile.Profile() as pr:
    lace = sg.Lace(squares, offset=3)


# # Format and print the results
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME).print_stats(20)


# print(pattern.all_vertices[-1], pattern.all_shapes[-1][-1])
# print(pattern.all_vertices[0], pattern.all_shapes[0][0])

canvas.draw(lace)
# print(len(pattern.all_vertices))
canvas.save("c:/tmp/lace_test.svg", overwrite=True)
