import simetri.graphics as sg
from simetri.shapes.shape import clip

canvas = sg.Canvas()

star = sg.reg_star_polygon(5, 2, 100, line_width=2)
star2 = sg.reg_star_polygon(5, 2, 100)
star2.rotate(sg.pi / 10)

rect = sg.Rectangle((0, 0), 100, 110)
rect2 = sg.Rectangle((0, 0), 60, 80)
rect3 = sg.Rectangle((0, 0), 90, 110).translate(30, 40)


canvas.draw([star, rect], fill=False)
res = clip(star, rect).merge_shapes()
canvas.draw(res, fill_color=sg.acid_green, alpha=0.5, line_width=3)

canvas.translate(200, 0)

res2 = clip(star, rect, exclude_clipper=True).merge_shapes()
canvas.draw(res2, fill_color=sg.acid_green, alpha=0.5, line_width=3)

canvas.translate(-200, -200)
# canvas.draw([star, rect], fill=False)

res3 = diff(star, rect, exclude_clipper=False)
canvas.draw(
    res3,
    fill_color=sg.acid_green,
    alpha=0.5,
    line_width=3,
)

canvas.translate(200, 0)
canvas.draw(
    difference(star, rect, exclude_clipper=True),
    fill_color=sg.acid_green,
    alpha=0.5,
    line_width=3,
)
# canvas.draw(
#     clip(star, star2), fill_color=sg.acid_green, alpha=0.5, line_width=2
# )
# canvas.translate(200, 0)
# canvas.draw([star, star2], fill=False)
# canvas.draw(
#     clip(star, star2, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star, star2], fill=False)
# canvas.draw(
#     union(star, star2), fill_color=sg.acid_green, alpha=0.5, line_width=2
# )


# canvas.translate(200, 0)
# canvas.draw([star, star2], fill=False)
# canvas.draw(
#     diff(star, star2, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star, star2], fill=False)
# canvas.draw(
#     diff(star, star2, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star, star2], fill=False)
# canvas.draw(
#     xor(star, star2, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star, star2], fill=False)
# canvas.draw(
#     xor(star, star2, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# #################################################
# canvas.translate(-1200, 200)
# canvas.draw([star2, star], fill=False)
# canvas.draw(
#     clip(star2, star), fill_color=sg.acid_green, alpha=0.5, line_width=2
# )
# canvas.translate(200, 0)
# canvas.draw([star2, star], fill=False)
# canvas.draw(
#     clip(star2, star, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star2, star], fill=False)
# canvas.draw(
#     union(star2, star), fill_color=sg.acid_green, alpha=0.5, line_width=2
# )


# canvas.translate(200, 0)
# canvas.draw([star2, star], fill=False)
# canvas.draw(
#     diff(star2, star, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star2, star], fill=False)
# canvas.draw(
#     diff(star2, star, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star2, star], fill=False)
# canvas.draw(
#     xor(star2, star, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star2, star], fill=False)
# canvas.draw(
#     xor(star2, star, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# #################################################
# canvas.translate(-1200, 200)
# canvas.draw([star, rect], fill=False)
# canvas.draw(
#     clip(star, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star, rect], fill=False)
# canvas.draw(
#     clip(star, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star, rect], fill=False)
# canvas.draw(
#     union(star, rect), fill_color=sg.acid_green, alpha=0.5, line_width=2
# )

# canvas.translate(200, 0)
# canvas.draw([star, rect], fill=False)
# canvas.draw(
#     diff(star, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star, rect], fill=False)
# canvas.draw(
#     diff(star, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star, rect], fill=False)
# canvas.draw(
#     xor(star, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([star, rect], fill=False)
# canvas.draw(
#     xor(star, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# #################################################
# canvas.translate(-1200, 200)
# canvas.draw([rect, star], fill=False)
# canvas.draw(
#     clip(rect, star, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect, star], fill=False)
# canvas.draw(
#     clip(rect, star, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect, star], fill=False)
# canvas.draw(
#     union(rect, star), fill_color=sg.acid_green, alpha=0.5, line_width=2
# )

# canvas.translate(200, 0)
# canvas.draw([rect, star], fill=False)
# canvas.draw(
#     diff(rect, star, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect, star], fill=False)
# canvas.draw(
#     diff(rect, star, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect, star], fill=False)
# canvas.draw(
#     xor(rect, star, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect, star], fill=False)
# canvas.draw(
#     xor(rect, star, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# #################################################
# canvas.translate(-1200, 200)
# canvas.draw([rect3, rect], fill=False)
# canvas.draw(
#     clip(rect3, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect3, rect], fill=False)
# canvas.draw(
#     clip(rect3, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect3, rect], fill=False)
# canvas.draw(
#     union(rect3, rect), fill_color=sg.acid_green, alpha=0.5, line_width=2
# )

# canvas.translate(200, 0)
# canvas.draw([rect3, rect], fill=False)
# canvas.draw(
#     diff(rect3, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect3, rect], fill=False)
# canvas.draw(
#     diff(rect3, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect3, rect], fill=False)
# canvas.draw(
#     xor(rect3, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect3, rect], fill=False)
# canvas.draw(
#     xor(rect3, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )
# #################################################
# canvas.translate(-1200, 200)
# canvas.draw([rect, rect3], fill=False)
# canvas.draw(
#     clip(rect, rect3, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect, rect3], fill=False)
# canvas.draw(
#     clip(rect, rect3, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect, rect3], fill=False)
# canvas.draw(
#     union(rect, rect3), fill_color=sg.acid_green, alpha=0.5, line_width=2
# )

# canvas.translate(200, 0)
# canvas.draw([rect, rect3], fill=False)
# canvas.draw(
#     diff(rect, rect3, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect, rect3], fill=False)
# canvas.draw(
#     diff(rect, rect3, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect, rect3], fill=False)
# canvas.draw(
#     xor(rect, rect3, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect, rect3], fill=False)
# canvas.draw(
#     xor(rect, rect3, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )
# #################################################
# canvas.translate(-1200, 200)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     clip(rect2, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     clip(rect2, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     union(rect2, rect), fill_color=sg.acid_green, alpha=0.5, line_width=2
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     diff(rect2, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     diff(rect2, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     xor(rect2, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     xor(rect2, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )


# #################################################

# canvas.translate(-1200, 200)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     clip(rect2, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     clip(rect2, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     union(rect2, rect), fill_color=sg.acid_green, alpha=0.5, line_width=2
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     diff(rect2, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     diff(rect2, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     xor(rect2, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     xor(rect2, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )
# #################################################


# ############################################
# canvas.translate(-1200, 200)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     clip(rect2, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     clip(rect2, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     union(rect2, rect), fill_color=sg.acid_green, alpha=0.5, line_width=2
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     diff(rect2, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     diff(rect2, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     xor(rect2, rect, exclude_clipper=True),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )

# canvas.translate(200, 0)
# canvas.draw([rect2, rect], fill=False)
# canvas.draw(
#     xor(rect2, rect, exclude_clipper=False),
#     fill_color=sg.acid_green,
#     alpha=0.5,
#     line_width=2,
# )
# ####################################################
# canvas.translate(-1200, 100)

# canvas.text("Clip", (0, 0), font_size=22)

# canvas.translate(200, 0)
# canvas.text("Clip, exclude", (0, 0), font_size=22)

# canvas.translate(200, 0)
# canvas.text("Union", (0, 0), font_size=22)

# canvas.translate(200, 0)
# canvas.text("Diff", (0, 0), font_size=22)

# canvas.translate(200, 0)
# canvas.text("Diff, exclude", (0, 0), font_size=22)

# canvas.translate(200, 0)
# canvas.text("XOR", (0, 0), font_size=22)

# canvas.translate(200, 0)
# canvas.text("XOR, exclude", (0, 0), font_size=22)


canvas.save("c:/tmp/clipper_test3.pdf", overwrite=True)
canvas.save("c:/tmp/clipper_test3.svg", overwrite=True)
canvas.save("c:/tmp/clipper_test3.tex", overwrite=True)
