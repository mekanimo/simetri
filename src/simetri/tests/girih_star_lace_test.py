import simetri.geom.points.point_utils
import simetri.geom.segments.line_utils
import simetri.graphics as sg

# from simetri.jupyter import show
# from simetri.star_patterns import Star

canvas = sg.Canvas()

star_12 = sg.stars.Star(12, 50)
# print(star_9[1][:3])
# print(star_9.all_vertices[:3])
p1, p2, p3 = [(48.3, 48.3), (248.55, 48.3), (261.05, 0.0)]
kernel = sg.Shape([p1, p2, p3])
r = simetri.geom.points.point_utils.distance(p2, p3)
R = simetri.geom.points.point_utils.distance(sg.origin, p3)
# print(sg.extended_line(r, (p2, p3)))
p4 = simetri.geom.segments.line_utils.extended_line(r, (p2, p3))[1]
kernel2 = sg.Shape([p1, p2, p4])
# canvas.draw(star_12, fill=False, line_width= 3)
petal = kernel.mirror(sg.axis_x, reps=1)
petal2 = kernel2.mirror(sg.axis_x, reps=1)
star1 = petal.rotate(2 * sg.pi / 6, reps=5)
petal2.rotate(2 * sg.pi / 12)
star2 = petal2.rotate(2 * sg.pi / 6, reps=5)
stars = sg.Batch([star1, star2]).merge_shapes()
# canvas.draw(star1, fill=False, line_width= 3)
# canvas.draw(star2, fill=False, line_width= 3)
# canvas.draw(stars, fill=False, line_width= 3)
stars2 = stars.copy().translate(2 * R, 0)
stars2.rotate(2 * sg.pi / 6, reps=5)
# merged = sg.Batch(stars2 + stars).merge_shapes(n_round=0)
# canvas.draw(merged[0], fill=False, line_width= 3)
# canvas.draw(merged[11], fill=False, line_width= .5)
# print(merged[0][0], merged[0][-1])
# print(merged[11][0], merged[11][-1])
# canvas.draw(stars, fill=False, line_width= 3)

# lace = sg.Lace(
#     polygon_shapes=sg.Batch(merged[0]),
#     polyline_shapes=sg.Batch(merged[1:]),
#     swatch=sg.random_swatch(),
#     offset=5,
# )
# lace = sg.Lace(polyline_shapes=stars2, offset=5)


# canvas.draw(lace.scale(0.5), swatch=sg.random_swatch(), line_width=3)

# canvas.draw(stars2)
all_stars = sg.Group([stars, stars2])
segments = all_stars.all_segments
shapes = sg.Group([sg.Shape([p1, p2]) for (p1, p2) in segments])
merged = all_stars.merge_shapes()

# for x in merged:
#     canvas.draw(x, fill=False, line_width=3)
#     canvas.draw(merged, fill=False)
#     canvas.translate(0, -merged.width * 1.2)

canvas.draw(merged, fill=False, line_width=3)
lace = sg.Lace(merged, offset=4, debug=True)

# canvas.draw(lace)
# print("merged", merged, len(merged))
# edges = merged.all_segments
# segments = sg.Group([sg.Shape(x) for x in edges])
# # print("segments", segments, len(segments))
# res = segments.merge_shapes(merge_angle_tol=0.02)

# print(len(res))
# print(edges, len(edges))
# all_segments = sg.Group(merged.all_edges)
# res = all_segments.merge_shapes()
# print(res)
# canvas.draw(merged[16], fill=False)
# canvas.draw(merged[9], fill=False)
# canvas.draw(merged[1], fill=False)
# for merge in merged:
#     canvas.draw(merge, fill=False)
# grp = sg.Group([merged[16], merged[9]])
# res = grp.merge_shapes()

# canvas.draw(res[0], line_width=2, line_color=sg.red)
# canvas.draw(res[1], line_width=4, line_color=sg.teal)

# print(len(res))
# canvas.draw(res[1])
# canvas.draw(res[0], indices=True)
# edge1 = res[1].edges[3]
# edge2 = res[0].edges[11]
# angle1 = sg.inclination_angle(*edge1)
# angle2 = sg.inclination_angle(*edge2)

# print("Is collinear:", sg.collinear_segments(edge1, edge2))
# print(sg.degrees(abs(angle1 - sg.pi)))
# print(sg.degrees(abs(angle2 - sg.pi)))
# print(edge1)
# p1, p2 = edge1
# p3, p4 = edge2
# canvas.draw(sg.Shape([p1, p2]), line_width=2, line_color=sg.red)
# canvas.draw(sg.Shape([p3, p4]), line_width=4, line_color=sg.teal)
# shp1 = merged[16]
# shp2 = sg.Shape(merged[9][9:16])

# shapes = sg.Group([shp1, shp2])
# merged_shapes = shapes.merge_shapes(n_round=3)
# # print(len(merged_shapes))
# canvas.draw(merged_shapes[0], line_color=sg.red, indices=True)
# canvas.draw(merged_shapes[1], line_color=sg.green, line_width=3, indices=True)
# print("0", [sg.round_point(x, 3) for x in merged_shapes[0].vertices])
# print("1", [sg.round_point(x, 3) for x in merged_shapes[1].vertices])
# canvas.draw(sg.Shape(merged[9][9:16]), line_width=3)
# print(len(merged))
# for i, x in enumerate(merged):
#     # print(i, len(x))
#     canvas.draw(x, fill=False)
#     canvas.text(str(i), (0, 0))
#     canvas.translate(0, -x.height * 1.2)

# lace = sg.Lace(sg.Group([*merged[:17]]), offset=3)
# canvas.draw(merged[0], fill=False)
# canvas.draw(merged[18])
# canvas.draw(merged[-3])

# canvas.draw(all_stars.merge_shapes(), fill=False)
# canvas.draw(merged[0], fill=False, line_width= .5)
# # all_stars.scale(5)
# # for i, poly in enumerate(all_stars):
# #     canvas.draw(poly.translate(0, -i*400).scale(.2), fill=False, line_width= 3)
# all_stars2 = all_stars.copy()

# canvas.draw(all_stars, fill=False, line_width= .1, line_color=sg.colors.white)

# # canvas.draw(all_stars2[0].scale(.5), fill=False, line_width= 2)
# # for poly in all_stars2[1:]:
# #     canvas.draw(poly.scale(.5), fill=False, line_width= 2, line_color = sg.random_color())
# # canvas.draw(all_stars2[1].scale(.4), fill=False, line_width= 2, line_color = sg.random_color())
# # lace = sg.Lace(sg.Batch([all_stars[0]]), sg.Batch(all_stars[1:]), offset=2)
# # canvas.draw(lace.scale(.2), fill=False)
# lace = sg.Lace(polyline_shapes=stars, offset=5)
# # lace2 = lace.copy().translate(2*R, 0)
# canvas.draw(lace, fill=False)
# print(len(merged[16]))
# print(len(merged[17]))
# canvas.draw(merged[16], fill=False, line_width=3)
# canvas.draw(merged[17], fill=False, line_width=2)
# print([sg.round_point(v, 1) for v in merged[16].vertices])
# lace = sg.Lace(sg.Group([merged[13:]]), offset=5)
# lace = sg.Lace(sg.Group([merged[:16]]), offset=5)
# lace = sg.Lace(res, offset=5)
# canvas.draw(lace.translate(2*R, 0), fill=False)
# canvas.draw(lace.rotate(2*sg.pi/6), fill=False)

# canvas.draw(merged[-1], fill=False, line_width= 3)

# for poly in merged:
#     canvas.draw(poly, fill=False, line_width= 2, line_color = sg.random_color())

# canvas.draw(polylines[7])
# lace = sg.Lace(polygon_shapes=polygons, polyline_shapes=polylines, offset=5)
# lace = sg.Lace(polygons, polylines, offset=2, debug=True)
# canvas.draw(polylines, fill=False, line_width= 2)
# canvas.draw(lace, fill=False, line_width=1)
# canvas.save('c:/tmp/girih_star_streaks.pdf')
# canvas.draw(merged[1:], fill=False, line_width= 2)
canvas.save("c:/tmp/girih_star_lace_test4.svg", overwrite=True)
