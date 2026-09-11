import cProfile
import pstats
import time
from collections import defaultdict

import networkx as nx
import numpy as np

import simetri.graphics as sg
from simetri.geom.polygons.polygon import symmetric_difference


def draw_symm_diff(shapes, length_bounds=10):
    canvas = sg.Canvas()
    partitions, _union = symmetric_difference(shapes, length_bounds)
    # canvas.draw(shapes, fill=False)
    # canvas.translate(0, -dy)
    # canvas.draw(union, fill=False)
    # canvas.translate(0, -dy)
    # print("***", len(partitions))
    for part in partitions:
        if not part.fill:
            color = sg.yellow
        else:
            color = sg.navy
        canvas.draw(part, fill=True, fill_color=color)
    canvas.save("c:/tmp/partition_test3.svg", overwrite=True)


# star_12 = sg.stars.Star(12, 50)
# p1, p2, p3 = [(48.3, 48.3), (248.55, 48.3), (261.05, 0.0)]
# kernel = sg.Shape([p1, p2, p3])
# r = sg.distance(p2, p3)
# R = sg.distance(sg.origin, p3)
# p4 = sg.extended_line(r, (p2, p3))[1]
# kernel2 = sg.Shape([p1, p2, p4])
# petal = kernel.mirror(sg.axis_x, reps=1)
# petal2 = kernel2.mirror(sg.axis_x, reps=1)
# star1 = petal.rotate(2 * sg.pi / 6, reps=5)
# petal2.rotate(2 * sg.pi / 12)
# star2 = petal2.rotate(2 * sg.pi / 6, reps=5)
# stars = sg.Batch([star1, star2]).merge_shapes()

# stars2 = stars.copy().translate(2 * R, 0)
# stars2.rotate(2 * sg.pi / 6, reps=5)

# all_stars = sg.Group([stars, stars2])
# segments = all_stars.all_segments
# shapes = sg.Group([sg.Shape([p1, p2]) for (p1, p2) in segments])
# merged = all_stars.merge_shapes()

grid = sg.CircularGrid(n=20, radius=100)
p1 = grid.intersect((0, 10), (2, 14))
p2 = grid.intersect((0, 6), (2, 14))
p3 = grid.points[0]
n = 10
kernel = sg.Shape([p1, p2, p3])
petal = kernel.mirror(sg.axis_x, reps=1)
star = petal.rotate(2 * sg.pi / n, reps=n - 1)
# canvas.draw(lace)

draw_symm_diff(star, 10)
lat = sg.lattice_pm
# canvas = sg.Canvas()
# canvas.draw(merged, fill=False, line_width=3)
# canvas.save("c:/tmp/partition_test3.svg", overwrite=True)
# with cProfile.Profile() as pr:
# draw_symm_diff(merged, 10)


# # # Format and print the results
# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME).print_stats(20)
