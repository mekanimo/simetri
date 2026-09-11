import simetri.graphics as sg

# from The Grammar of Ornament, Owen Jones, page 104
canvas = sg.Canvas()
canvas.render = 'TEX'
a = 5 * 2.5
b = 2*a
d = 2*b
e = 2*d

lw = 2

verts = [(0, 0), (-a, -a), (-b, 0), (-b, b)]

kernel = sg.Shape(verts)
pattern = kernel.mirror((kernel[0], kernel[-1]), reps=1).rotate(sg.pi, kernel[1], reps=1)
pattern.mirror(pattern.right, reps=1).translate(e, 0, reps=6)
# pattern = pattern.merge_shapes()[0]
# canvas.draw(pattern, indices=True)
lace = sg.Lace(pattern, offset=7)
##
# box = sg.Shape(lace.b_box.corners, closed=True)
box = sg.offset_box(lace.b_box.corners, offsets=[-10, 10, -10, 10])
print([f"({x[0]:.1f}, {x[1]:.1f})" for x in lace.corners])
print([f"({x[0]:.1f}, {x[1]:.1f})" for x in box.corners])
mask = sg.Mask(
  box,
  stops=[
    sg.Stop(offset=0.0, opacity=1),
    sg.Stop(offset=.6, opacity=0),
    sg.Stop(offset=1.0, opacity=0)
  ],
  axis=((0.0, 0.0), (1.0, 0.0)),
  mask_units=sg.SvgUnits.OBJECT_BOUNDING_BOX,
)
mask2 = sg.Mask(
  box,
  stops=[
    sg.Stop(offset=0.0, opacity=0),
    sg.Stop(offset=.2, opacity=0.1),
    sg.Stop(offset=1.0, opacity=1)
  ],
  axis=((0.0, 0.0), (1.0, 0.0)),
  mask_units=sg.SvgUnits.OBJECT_BOUNDING_BOX,
)
# canvas.draw(lace, mask=mask2, swatch=sg.random_swatch())
# canvas.draw(box, fill=False)
# canvas.draw(pattern, line_width=.5, mask=mask)
# for p in lace.plaits:
#     canvas.draw(p, fill_color=sg.aqua_blue)
# canvas.draw(lace.plaits, fill_color=sg.aqua_blue)
canvas.draw(lace, fillet_radii=(3, 3), line_width=1,             plait_fill_color=sg.aqua_green)
# canvas.draw(kernel, line_width=lw, line_color=sg.navy)
# kernel.translate(e, 0)
# canvas.draw(kernel.mirror((kernel[0], kernel[-1]), reps=1).merge_shapes(),
#             line_width=lw, fill=False)
# kernel.translate(e, 0)
# canvas.draw(kernel.mirror((kernel[0], kernel[-1]), reps=1).rotate(sg.pi, kernel[1], reps=1).merge_shapes(),
#             line_width=lw, fill=False)
# kernel.translate(e, 0)
# kp = kernel.mirror((kernel[0], kernel[-1]), reps=1).rotate(sg.pi, kernel[1], reps=1)
# kp.mirror(kp.right, reps=1)
# canvas.draw(kp.merge_shapes(), line_width=lw, fill=False)
##canvas.line((-d, -d-a), (12*d, -d-a), line_width=lw)
##canvas.line((-d, d+a), (12*d, d+a), line_width=lw)
canvas.text(r"$((K, \mathbf{M}^1 _{(K[0], K[-1])} \mathbf{R}_{K[1]}^1 \pi \mathbf{M}^1_R), \mathbf{T}^\infty \langle d, 0 \rangle )$", pos = (6*d, 2*d-d/2-5), font_size=20, fill=False)
canvas.text(r"$a = 15, \,b = 2a,\, d = 8a$", pos=(6*d, 2*(d+2*a)-d/2),
            font_size=18, fill=False)
canvas.text(r"$K \equiv [(0, 0), (-a, -a), (-b, 0), (-b, b)]$", pos=(6*d, 2*(d + a)-d/2),
            font_size=18, fill=False)
##canvas.limits = (d, -2*d, 11*d, 4*d)
##canvas.back_color=sg.aqua_blue
canvas.circle(2, (0, 0), color=sg.yellow)
# canvas.circle(3, (0, 0), fill=False, line_width=.5)
# canvas.text('$0$', (3, 3), fill=False)
canvas.back_color = sg.Color(.95, .95, .95)

canvas.save('c:/tmp/rounded_lace2.pdf', overwrite=True)
