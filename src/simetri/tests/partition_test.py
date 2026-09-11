# from collections import defaultdict
# import cProfile
# import pstats
# import time
# import networkx as nx
# import numpy as np
import time

import simetri.graphics as sg
from simetri.geom.polygons.polygon import symmetric_difference


def draw_symm_diff(shapes, length_bounds=10):
    canvas = sg.Canvas()
    start = time.perf_counter_ns()
    partitions, union = symmetric_difference(shapes, length_bounds)
    end = time.perf_counter_ns()
    print("elapsed", (end - start) / 1e06)
    dy = sg.Group(partitions).height * 1.15
    canvas.draw(shapes, fill=False)
    canvas.translate(0, -dy)
    canvas.draw(union, fill=False)
    canvas.translate(0, -dy)
    for part in partitions:
        if not part.fill:
            color = sg.yellow
        else:
            color = sg.navy
        canvas.draw(part, fill=True, color=color)
    canvas.save("c:/tmp/partition_test.svg", overwrite=True)


points = [(0, 0), (40, 0), (40, 40), (0, 40)]
square = sg.Shape(points, closed=True)
squares = square.translate(20, 30, reps=1).translate(50, 0, reps=10)
squares.translate(-10, 50, reps=10)

draw_symm_diff(squares, 13)
# with cProfile.Profile() as pr:
#     draw_symm_diff(squares, 13)


# # # Format and print the results
# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME).print_stats(20)
