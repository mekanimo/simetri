"""warning test"""

import simetri.graphics as sg

sg.doc(sg.random_rectangle)

points = [(0, 0), (80, 0), (50, 40)]
triangle = sg.Shape(points, closed=True)

canvas = sg.Canvas()
canvas.draw(triangle)
canvas.help_lines()
# canvas.display()
