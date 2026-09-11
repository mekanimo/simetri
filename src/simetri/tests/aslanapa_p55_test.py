import simetri.graphics as sg
from simetri.interlace import Lace
from simetri.wallpapers.wallpaper import wallpaper_pg

c = sg.Canvas()

points = [(45, 45), (72.42, 30), (72.42, 0)]
batch = sg.Batch([sg.Shape(points)])
batch.mirror(sg.axis_x, reps=1).rotate(sg.pi / 2, reps=3)
batch = batch.merge_shapes()
w = 30
wallpaper_pg(batch, [(0, -w), (1, -w)], 2 * w, 4 * w, 4 * w, 5, 5)
lace = Lace(batch, offset=4)
d = 100
c.draw(lace)
# c.shrink(top=d, left=d, bottom=d, right=d)
c.save("c:/tmp/aslanapa_p55_.svg", overwrite=True)
