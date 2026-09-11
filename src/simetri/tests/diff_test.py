import simetri.graphics as sg

# def multi_split_segment(
#     segment: sg.LineType, points: sg.Sequence, dist_tol=0.1
# ):
#     """Splits the segment into multiple pieces by using the given points.
#     Returns multiple segments.
#     """
#     p1, p2 = segment

#     distances = []
#     for i, pnt in enumerate(points):
#         dist = sg.distance(p1, pnt)
#         distances.append((dist, i))
#     distances.sort()
#     points = [points[ind] for (_, ind) in distances]

#     if len(points) == 2:
#         close_p1 = sg.close_points_square(points[0], p1)
#         close_p2 = sg.close_points_square(points[1], p2)
#         if close_p1 and close_p2:
#             return [segment]

#     for i, pnt in enumerate(points):
#         if sg.close_points_square(p1, pnt):
#             continue
#         if not sg.point_on_line_segment(pnt, segment):
#             print("Point not on segment.")
#             return []

#     segments = []
#     start = p1
#     for point in points:
#         if sg.distance(start, point) < dist_tol:
#             continue
#         segments.append((start, point))
#         start = point

#     return segments


# def diff(
#     shape1: "Shape",
#     shape2: "Shape",
#     exclude_clipper: bool = False,
#     dist_tol: float = 0.01,
# ):
#     """
#     shape1 Shape: shape to be clipped
#     shape2 Shape: clipping region
#     exclude_clipper bool: If True, clipper's edges are excluded.
#     """
#     if not (shape1.closed and shape2.closed):
#         raise Warning("Both shapes must be closed")

#     segments = [[p1[:2], p2[:2]] for (p1, p2) in shape1.edges] + [
#         [p1[:2], p2[:2]] for (p1, p2) in shape2.edges
#     ]
#     intersections = sg.all_intersections(segments)

#     all_segments_ = []
#     for key, value in intersections[0].items():
#         segment = segments[key]
#         points = [x[0] for x in value]
#         points = sg.remove_duplicate_points(points)
#         split_segments = multi_split_segment(segment, points)
#         for x in split_segments:
#             if sg.distance(*x) < 0.001:
#                 print("split seg", x)
#         all_segments_.append(split_segments)

#     diff_ = sg.Group()
#     shape_vertices = shape1.vertices
#     shape2_vertices = shape2.vertices
#     for segs in all_segments_:
#         for seg in segs:
#             in1 = sg.in_polygon(sg.midpoint(*seg), shape_vertices)
#             in2 = sg.in_polygon(
#                 sg.midpoint(*seg), shape2_vertices, not exclude_clipper
#             )
#             if in1 and not in2:
#                 diff_.append(sg.Shape(seg))

#     return diff_.merge_shapes()


points_a = [(-10, 40), (60, 30), (40, 21), (80, -10), (10, -12), (35, 10)]
a = sg.Shape(points_a, closed=True, fill_color=sg.blue, alpha=0.65)
points_b = [(0, 0), (35, 50), (60, 0), (40, -22), (35, -5), (20, 10)]
b = sg.Shape(points_b, closed=True, fill_color=sg.gold, alpha=0.75)
a.scale(2)
b.scale(2)
diff_a_b = sg.polygon_diff(a, b)
diff_b_a = sg.polygon_diff(b, a)
# res = sg.Group()
# for x in diff_a_b:
#     # print(x)
#     if sg.distance(*x.vertices) > 0.001:
#         res.append(x)
#     else:
#         print(x)

canvas = sg.Canvas()
canvas.draw(sg.polygon_intersection(a, b))
canvas.translate(0, -200)

canvas.draw(diff_a_b, fill_color=sg.teal)
canvas.draw(diff_b_a, fill_color=sg.orange)


canvas.save("c:/tmp/diff_test.svg", overwrite=True)
