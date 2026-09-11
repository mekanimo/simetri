import simetri.geom.segments.line_utils
import simetri.graphics as sg

canvas = sg.Canvas()

seal = sg.stars.Star(8).level(0).merge_shapes()[0]
oct = sg.reg_poly_shape(8, 30)
d = 90
oct.translate(d, -d)
x1, y1 = oct[3]
x2, y2 = oct[4]
xp = simetri.geom.segments.line_utils.intersect(
    [(x1, y1), (x2 + 3, y2)], seal.edges[14]
)
canvas.circle(2, xp)
shp = sg.Shape([seal[14], xp, oct[3]])
canvas.draw([seal, oct, shp], fill=False, indices=False)
triangle = sg.Shape([(0, 0), (d, -d), (d, 0)], closed=True)
canvas.draw(triangle, alpha=0.3)
kernel = sg.clip(sg.Batch([seal, oct, shp]), triangle, exclude_clipper=True)
canvas.draw(kernel, line_width=2, indices=True)
kernel = kernel.merge_shapes()[0]
x, y = kernel[0][:2]
kernel[0] = (x, y - 3)
unit = kernel.mirror(sg.axis_x, reps=1).rotate(sg.pi / 2, reps=3)
# canvas.draw(unit, line_width=2)
uw = unit.width
pattern = unit.translate(uw, 0, reps=3).translate(0, uw, reps=3)
# canvas.draw(pattern, line_width=1)
swatch = sg.d_name_palette["seq_SUNSET_7"]
lace = sg.Lace(pattern, swatch=swatch, offset=4)
canvas.draw(lace, swatch=swatch)
canvas.save("c:/tmp/demiriz_pg_104_test.pdf", overwrite=True)
