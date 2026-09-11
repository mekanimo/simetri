import simetri.geom.points.point_utils
import simetri.geom.segments.line_utils
import simetri.graphics as sg


def multi_split_segment_old(
    segment: sg.LineType, points: sg.Sequence, dist_tol=0.1
):
    """Splits the segment into multiple pieces by using the given points.
    Returns multiple segments.
    """
    p1, p2 = segment
    distances = []
    for i, pnt in enumerate(points):
        dist = simetri.geom.points.point_utils.distance(p1, pnt)
        distances.append((dist, i))
    distances.sort()
    points = [points[ind] for (_, ind) in distances]

    if len(points) == 2:
        close_p1 = simetri.geom.points.point_utils.close_points_square(
            points[0], p1
        )
        close_p2 = simetri.geom.points.point_utils.close_points_square(
            points[1], p2
        )
        if close_p1 and close_p2:
            return [segment]

    for i, pnt in enumerate(points):
        if simetri.geom.points.point_utils.close_points_square(p1, pnt):
            continue
        if not simetri.geom.points.point_utils.point_on_line_segment(
            pnt, segment
        ):
            print("point not on line")
            return None

    segments = []
    start = p1
    for point in points:
        if (
            simetri.geom.points.point_utils.distance(start, point)
            < dist_tol
        ):
            continue
        segments.append((start, point))
        start = point
    segments.append((start, p2))

    return segments


def multi_split_segment(
    segment: sg.LineType,
    points: sg.Sequence[sg.PointType],
    dist_tol: float = 0.1,
) -> sg.Sequence[sg.LineType] | None:
    """Splits the segment into multiple pieces by using the given points.
    Returns multiple segments.
    """
    p1, p2 = segment
    distances = []
    for i, pnt in enumerate(points):
        dist = simetri.geom.points.point_utils.distance(p1, pnt)
        distances.append((dist, i))
    distances.sort()
    points = [points[ind] for (_, ind) in distances]

    if len(points) == 2:
        close_p1 = simetri.geom.points.point_utils.close_points_square(
            points[0], p1
        )
        close_p2 = simetri.geom.points.point_utils.close_points_square(
            points[1], p2
        )
        if close_p1 and close_p2:
            return [segment]

    for i, pnt in enumerate(points):
        if simetri.geom.points.point_utils.close_points_square(p1, pnt):
            continue
        if not simetri.geom.points.point_utils.point_on_line_segment(
            pnt, segment
        ):
            print("point not on line")
            return None

    segments = []
    start = p1
    for point in points:
        if (
            simetri.geom.points.point_utils.distance(start, point)
            < dist_tol
        ):
            continue
        segments.append((start, point))
        start = point

    return segments


def test_multi_split_segment():
    seg = ((0, 0), (10, 0))
    points = [
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 0),
        (9, 0),
        (10, 0),
    ]
    res1 = multi_split_segment(seg, points)
    res2 = multi_split_segment_old(seg, points)

    # print(res1 == res2)
    for i, x in enumerate(res1):
        print(i, x == res2[i])
    print(len(res1), len(res2))
    print(res1[9], res2[10])


test_multi_split_segment()
print(
    simetri.geom.segments.line_utils.all_intersections(
        [[(0, 0), (50, 50)], [(60, 0), (60, 60)]], return_points_list=True
    )
)
