import simetri.graphics as sg
from simetri.interlace import Lace

c = sg.Canvas()
points = [(15, 0), (15, 15), (40, 15), (55, 0), (70, 15), (70, 70)]
batch = sg.Batch(sg.Shape(points))
batch.mirror([(0, 0), (1, 1)], reps=1).mirror(sg.axis_x, reps=1).mirror(
    sg.axis_y, reps=1
)
batch.translate(200, 200)
# c.draw(batch)
#
batch = batch.merge_shapes()
# batch.fill=False
# c.draw(batch)
# print(len(batch))
shape = batch[0]
shape.closed = True
shape.fill = False

# lace = Lace([shape], offset=1)
c.draw(batch)
# for i, v in enumerate(shape.vertices):
#     # c.draw(sg.Circle(*v, 2))
#     x, y = v
#     # c.draw(sg.Text(x + 5, y + 5, str(i), fontSize=6))
c.save("c:/tmp/aslanapa_p34_test.svg", overwrite=True)
