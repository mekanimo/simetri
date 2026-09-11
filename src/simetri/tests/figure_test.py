import simetri.graphics as sg
from simetri.extensions.polyominoes import iter_polyominoes
from simetri.shapes.figure import Figure

canvas = sg.Canvas()

n = 6
size = 15
polyos = iter_polyominoes(n=n, size=size)

polyo = next(polyos)
polyo = next(polyos)
print(type(polyo.geometry))
print(type(polyo.skin))
print(type(polyo))
# p1 = polyo[5]
# p2 = polyo[6]
# p3 = polyo[0]
# p4 = polyo[1]
# edges = polyo.edges

# p_1 = sg.intersect(edges[5], edges[0])
# p_2 = sg.intersect(edges[3], edges[0])

# fig_skin = sg.Group(
#     [
#         sg.Shape((p1, p2))
#         for (p1, p2) in [(polyo[3], polyo[6]), (polyo[6], p_1), (polyo[3], p_2)]
#     ]
# )
# fig_skin.append(sg.Shape([polyo.edge_midpoint(0), polyo.edge_midpoint(4)]))

# # fig_skin.set_attribs("line_color", sg.gray)
# fig_skin.set_attribs("line_width", 0.5)
# fig = Figure(polyo, fig_skin)

# canvas.draw(fig, fill=False, vertices=True, indices=True)
# polyo.skin.set_attribs("fill", False)
canvas.draw(polyo, vertices=True, fill_color=sg.red, alpha=0.25)

tag1 = sg.Tag(
    "(0, 45.5)",
    pos=(0, -100),
    font_size=8,
    fill=False,
)

canvas.draw(tag1)
# canvas.draw(tag1.b_box)
canvas.save("c:/tmp/figure_test.svg", overwrite=True)
