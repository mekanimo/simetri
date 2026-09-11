import cProfile
import pstats

import simetri.geom.points.point_utils
import simetri.graphics as sg


def hilbert(batch, n=3, mirror=False):
    def connect(b1, b2):
        dist1 = simetri.geom.points.point_utils.distance(
            b1.all_shapes[0][0], b2.all_shapes[0][0]
        )
        dist2 = simetri.geom.points.point_utils.distance(
            b1.all_shapes[-1][-1], b2.all_shapes[-1][-1]
        )
        if dist1 < dist2:
            return sg.Shape([b1.all_shapes[0][0], b2.all_shapes[0][0]])
        else:
            return sg.Shape([b1.all_shapes[-1][-1], b2.all_shapes[-1][-1]])

    def step(batch):
        batch2 = batch.copy()
        bbox = batch.b_box
        # x, y = batch.midpoint[:2]
        x, y = bbox.midpoint[:2]
        if mirror:
            batch2.mirror([(x, y), (x, y - 1)])
        batch2.rotate(-sg.pi / 2, bbox.southwest)
        batch2.translate(0, -20)
        seg = connect(batch, batch2)
        pattern = sg.Batch([batch, batch2, seg]).merge_shapes()
        pattern.mirror(pattern.offset_line(sg.Side.RIGHT, 10), reps=1)
        # print(pattern)
        # print(connect(sg.Batch(pattern[0]), sg.Batch(pattern[1])))
        pattern.append(connect(sg.Batch(pattern[0]), sg.Batch(pattern[1])))
        res = pattern.merge_shapes()
        return res

    pattern = step(batch)
    for i in range(n - 1):
        pattern = step(pattern)

    return pattern.merge_shapes()


u = sg.Shape(
    [(25, 25), (25, 50), (0, 75), (50, 50), (100, 75), (75, 50), (75, 25)]
)
u.scale(0.25)
g = sg.Group([u])

# Profile the hilber function
# with cProfile.Profile() as pr:
#     pattern = hilbert(g, 6)

# # Format and print the results
# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME).print_stats(20)


pattern = hilbert(g, 5)
# print(pattern.all_vertices[-1], pattern.all_shapes[-1][-1])
# print(pattern.all_vertices[0], pattern.all_shapes[0][0])
canvas = sg.Canvas()
canvas.draw(pattern, line_width=2)
# print(len(pattern.all_vertices))
canvas.save("c:/tmp/hilbert_test.svg", overwrite=True)
