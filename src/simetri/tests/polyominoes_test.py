import simetri.geom.polygons.polygon_utils
import simetri.graphics as sg
from simetri.extensions.polyominoes import iter_polyominoes

canvas = sg.Canvas()

n = 6
size = 15
polyos = iter_polyominoes(n=n, size=size)
for i in range(640):
    pos = sg.get_cell_position(
        index=i,
        n_columns=12,
        cell_width=(size - 4) * n,
        cell_height=(size - 2) * n,
        gap=size / 8,
        margin=size,
    )
    polyo = next(polyos, None)

    if polyo is None:
        print(i)
        break
    if len(polyo) == 2 or not simetri.geom.polygons.polygon_utils.is_simple(
        polyo[0]
    ):
        color = sg.red
        alpha = 1
    else:
        color = sg.random_color()
        alpha = 0.5
    canvas.draw(polyo.move_to(pos), fill_color=color, alpha=alpha)

canvas.save(f"c:/tmp/polyominoes_test_{n}.svg", overwrite=True)
