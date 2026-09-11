from collections.abc import Sequence
import math

import simetri.geom.points.point_utils
import simetri.geom.polygons.polygon
import simetri.geom.segments.line_utils
import simetri.graphics as sg

Point = tuple[float, float]


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _mul(a: Point, s: float) -> Point:
    return (a[0] * s, a[1] * s)


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _len(v: Point) -> float:
    return math.hypot(v[0], v[1])


def _unit(v: Point) -> Point:
    L = _len(v)
    if L == 0:
        raise ValueError("Zero-length vector.")
    return (v[0] / L, v[1] / L)


def _cross_z(a: Point, b: Point) -> float:  # 2D z-component of cross product
    return a[0] * b[1] - a[1] * b[0]


def fillet_points(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    radius: float,
    n: int,
    *,
    clamp_radius: bool = False,
    eps: float = 1e-12,
) -> list[Point]:
    """
    Return n points along the circular fillet of given radius at vertex p2
    between segments p1->p2 and p2->p3 (2D). Points include tangency endpoints
    when n >= 2. If n == 1, returns the midpoint of the arc. If n <= 0, returns [].

    Parameters:
      p1, p2, p3: 2D points (x, y). p1-p2 and p2-p3 must not be collinear.
      radius: desired fillet radius (> 0).
      n: number of sample points along the arc (>= 1 recommended).
      clamp_radius: if True, reduces radius to the maximal feasible value
                    when the requested radius is too large for the given segments.
      eps: small numeric tolerance.

    Raises:
      ValueError if geometry is degenerate or radius is infeasible (unless clamped).
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if len(p1) != 2 or len(p2) != 2 or len(p3) != 2:
        raise ValueError("This function expects 2D points (x, y).")

    # Convert to Vector objects
    v1 = sg.Vector(float(p1[0]), float(p1[1]))
    v2 = sg.Vector(float(p2[0]), float(p2[1]))
    v3 = sg.Vector(float(p3[0]), float(p3[1]))

    # Direction vectors along the polyline
    v_in = (v2 - v1).normalize()  # direction into p2
    v_out = (v3 - v2).normalize()  # direction out of p2

    # Rays from the corner along each segment
    u1 = -v_in  # from p2 toward p1
    u2 = v_out  # from p2 toward p3

    # Interior angle between u1 and u2
    c = max(-1.0, min(1.0, u1.dot(u2)))
    theta = math.acos(c)
    if theta < eps or abs(theta - math.pi) < eps:
        raise ValueError(
            "Points are collinear or angle too close to 0/180 degrees."
        )

    # Distances and feasibility of the radius
    L1 = (v2 - v1).mag()
    L2 = (v3 - v2).mag()
    # Tangency distance along each leg
    t = radius / math.tan(theta / 2.0)
    # Max radius allowed by each leg: r_max_i = Li * tan(theta/2)
    rmax = min(L1, L2) * math.tan(theta / 2.0)

    if t > L1 + eps or t > L2 + eps:
        if clamp_radius:
            # Clamp radius to feasible value (slightly inside to avoid degeneracy)
            radius = max(0.0, min(radius, rmax * (1.0 - 1e-9)))
            t = radius / math.tan(theta / 2.0)
        else:
            raise ValueError(
                f"Radius too large for given segments. Max feasible ~ {rmax:.6f}"
            )

    # Tangency points on each leg
    A = v2 + u1 * t
    B = v2 + u2 * t

    # Bisector direction (inside the angle)
    bis = u1 + u2
    if bis.mag() < eps:
        raise ValueError("Angle too close to 180°, bisector undefined.")
    w_hat = bis.normalize()

    # Center of the fillet circle
    center_dist = radius / math.sin(theta / 2.0)
    C = v2 + w_hat * center_dist

    # Angles of tangency points around center
    a1 = math.atan2(A.y - C.y, A.x - C.x)
    a2 = math.atan2(B.y - C.y, B.x - C.x)

    # Determine sweep direction based on the turn (left/CCW or right/CW)
    turn = v_in.cross(v_out)  # >0 => left turn (CCW), <0 => right turn (CW)
    delta = a2 - a1
    # Normalize delta to follow the turn direction along the minor arc
    if turn > 0:  # CCW
        if delta < 0:
            delta += 2.0 * math.pi
    else:  # CW
        if delta > 0:
            delta -= 2.0 * math.pi

    # Generate points on the arc
    if n <= 0:
        return []
    if n == 1:
        ang = a1 + 0.5 * delta
        pt = C + sg.Vector(radius * math.cos(ang), radius * math.sin(ang))
        return [(pt.x, pt.y)]
    # n >= 2: include both tangency endpoints
    pts: list[Point] = []
    for i in range(n):
        tfrac = i / (n - 1)
        ang = a1 + tfrac * delta
        pt = C + sg.Vector(radius * math.cos(ang), radius * math.sin(ang))
        pts.append((pt.x, pt.y))
    return pts


def fillet_points2(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    radius: float,
    n: int,
    *,
    clamp_radius: bool = False,
    eps: float = 1e-12,
) -> list[Point]:
    """
    Return n points along the circular fillet of given radius at vertex p2
    between segments p1->p2 and p2->p3 (2D). Points include tangency endpoints
    when n >= 2. If n == 1, returns the midpoint of the arc. If n <= 0, returns [].

    Parameters:
      p1, p2, p3: 2D points (x, y). p1-p2 and p2-p3 must not be collinear.
      radius: desired fillet radius (> 0).
      n: number of sample points along the arc (>= 1 recommended).
      clamp_radius: if True, reduces radius to the maximal feasible value
                    when the requested radius is too large for the given segments.
      eps: small numeric tolerance.

    Raises:
      ValueError if geometry is degenerate or radius is infeasible (unless clamped).
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")
    p1 = p1[:2]
    p2 = p2[:2]
    p3 = p3[:2]

    p1 = (float(p1[0]), float(p1[1]))
    p2 = (float(p2[0]), float(p2[1]))
    p3 = (float(p3[0]), float(p3[1]))

    # Direction vectors along the polyline
    v_in = _unit(_sub(p2, p1))  # direction into p2
    v_out = _unit(_sub(p3, p2))  # direction out of p2

    # Rays from the corner along each segment
    u1 = _mul(v_in, -1.0)  # from p2 toward p1
    u2 = v_out  # from p2 toward p3

    # Interior angle between u1 and u2
    c = max(-1.0, min(1.0, _dot(u1, u2)))
    theta = math.acos(c)
    if theta < eps or abs(theta - math.pi) < eps:
        raise ValueError(
            "Points are collinear or angle too close to 0/180 degrees."
        )

    # Distances and feasibility of the radius
    L1 = _len(_sub(p2, p1))
    L2 = _len(_sub(p3, p2))
    # Tangency distance along each leg
    t = radius / math.tan(theta / 2.0)
    # Max radius allowed by each leg: r_max_i = Li * tan(theta/2)
    rmax = min(L1, L2) * math.tan(theta / 2.0)

    if t > L1 + eps or t > L2 + eps:
        if clamp_radius:
            # Clamp radius to feasible value (slightly inside to avoid degeneracy)
            radius = max(0.0, min(radius, rmax * (1.0 - 1e-9)))
            t = radius / math.tan(theta / 2.0)
        else:
            raise ValueError(
                f"Radius too large for given segments. Max feasible ~ {rmax:.6f}"
            )

    # Tangency points on each leg
    A = _add(p2, _mul(u1, t))
    B = _add(p2, _mul(u2, t))

    # Bisector direction (inside the angle)
    bis = _add(u1, u2)
    if _len(bis) < eps:
        raise ValueError("Angle too close to 180°, bisector undefined.")
    w_hat = _unit(bis)

    # Center of the fillet circle
    center_dist = radius / math.sin(theta / 2.0)
    C = _add(p2, _mul(w_hat, center_dist))

    # Angles of tangency points around center
    a1 = math.atan2(A[1] - C[1], A[0] - C[0])
    a2 = math.atan2(B[1] - C[1], B[0] - C[0])

    # Determine sweep direction based on the turn (left/CCW or right/CW)
    turn = _cross_z(v_in, v_out)  # >0 => left turn (CCW), <0 => right turn (CW)
    delta = a2 - a1
    # Normalize delta to follow the turn direction along the minor arc
    if turn > 0:  # CCW
        if delta < 0:
            delta += 2.0 * math.pi
    else:  # CW
        if delta > 0:
            delta -= 2.0 * math.pi

    # Generate points on the arc
    if n <= 0:
        return []
    if n == 1:
        ang = a1 + 0.5 * delta
        return [(_add(C, (radius * math.cos(ang), radius * math.sin(ang))))]
    # n >= 2: include both tangency endpoints
    pts: list[Point] = []
    for i in range(n):
        tfrac = i / (n - 1)
        ang = a1 + tfrac * delta
        pts.append(_add(C, (radius * math.cos(ang), radius * math.sin(ang))))
    return pts


def fillet_corners(shape, indices, radius, n=12):
    """Create a new shape with rounded corners (using the corresponding vertices
    in the indices list.)"""
    vertices = shape.vertices
    count = len(vertices)

    # Build set of indices to fillet for quick lookup
    fillet_set = set(indices)

    # Build new vertex list
    new_vertices = []
    for i in range(count):
        if i not in fillet_set:
            # Keep original vertex
            new_vertices.append(vertices[i][:2])
        else:
            # Replace with fillet arc
            prev_idx = (i - 1) % count
            next_idx = (i + 1) % count

            p1 = vertices[prev_idx][:2]
            p2 = vertices[i][:2]
            p3 = vertices[next_idx][:2]

            # Generate fillet points
            arc_points = fillet_points(p1, p2, p3, radius, n, clamp_radius=True)
            new_vertices.extend(arc_points)

    # Create new shape with same properties
    return sg.Shape(new_vertices, closed=shape.closed)


def fillet_corners2(shape, d_vert_radius, n=12):
    """Create a new shape with rounded corners (using the corresponding vertices
    in the indices list.)"""
    vertices = shape.vertices
    count = len(vertices)

    # Build set of indices to fillet for quick lookup
    indices = d_vert_radius.keys()
    fillet_set = set(indices)

    # Build new vertex list
    new_vertices = []
    fillet_count = 0
    for i in range(count):
        if i not in fillet_set:
            # Keep original vertex
            new_vertices.append(vertices[i][:2])
        else:
            # Replace with fillet arc
            prev_idx = (i - 1) % count
            next_idx = (i + 1) % count

            p1 = vertices[prev_idx][:2]
            p2 = vertices[i][:2]
            p3 = vertices[next_idx][:2]

            # Generate fillet points
            radius = d_vert_radius[i]
            arc_points = fillet_points(p1, p2, p3, radius, n, clamp_radius=True)
            new_vertices.extend(arc_points)
            fillet_count += 1

    # Create new shape with same properties
    return sg.Shape(new_vertices, closed=shape.closed)


arc = fillet_points(
    (0, 0), (100, 0), (0, 50), radius=5, n=12, clamp_radius=True
)


def inside_corner(a, b, c):
    return (
        simetri.geom.segments.line_utils.angle_between_lines3(a, b, c) < sg.pi
    )


# p1 = [(0, 0)]
# p2 = [(0, 50)]
# points = p1 + arc + p2

# shp = sg.Shape(points)
# canvas = sg.Canvas()
# canvas.scale = (5, 5)
# canvas.draw(shp)
# canvas.save('c:/tmp/fillet_radius.pdf', overwrite=True)
# print(arc)


def internal_angles(vertices):
    """
    Computes internal angles for a polygon given as a list of (x, y) tuples.
    Works for both convex and concave polygons.
    """
    n = len(vertices)
    if n < 3:
        return []

    # 1. Determine Winding Order (Signed Area)
    # Positive = CCW, Negative = CW
    area = simetri.geom.polygons.polygon.polygon_area(vertices)
    is_ccw = area > 0
    if not is_ccw:
        vertices = list(vertices)[:]
        vertices.reverse()
    angles = []
    pi = sg.pi
    for i in range(n):
        # Define three consecutive points
        p_prev = vertices[(i - 1) % n]
        p_curr = vertices[i]
        p_next = vertices[(i + 1) % n]

        # Vector 1: Incoming (from previous to current)
        v1 = sg.v_from_points(p_prev, p_curr)
        # Vector 2: Outgoing (from current to next)
        v2 = sg.v_from_points(p_curr, p_next)

        cross_prod = v1.cross(v2)
        dot_prod = v1.dot(v2)

        turning_angle = math.atan2(cross_prod, dot_prod)
        # Convert Turning Angle to Internal Angle
        internal_angle = pi - turning_angle
        angles.append(internal_angle)

    return angles


def internal_angles_gpt(vertices):
    """
    Computes internal angles for a polygon given as a list of (x, y) tuples.
    Works for both convex and concave polygons.
    """
    n = len(vertices)
    if n < 3:
        return []

    # 1. Determine Winding Order (Signed Area)
    # Positive = CCW, Negative = CW
    area = 0
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        area += p1[0] * p2[1] - p2[0] * p1[1]

    is_ccw = area > 0
    angles = []

    for i in range(n):
        # Define three consecutive points
        p_prev = vertices[(i - 1) % n]
        p_curr = vertices[i]
        p_next = vertices[(i + 1) % n]

        # Vector 1: Incoming (from previous to current)
        v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        # Vector 2: Outgoing (from current to next)
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        cross_prod = v1[0] * v2[1] - v1[1] * v2[0]
        dot_prod = v1[0] * v2[0] + v1[1] * v2[1]

        turning_angle = math.atan2(cross_prod, dot_prod)
        turning_degrees = math.degrees(turning_angle)

        # 2. Convert Turning Angle to Internal Angle
        if is_ccw:
            internal_angle = 180.0 - turning_degrees
        else:
            internal_angle = 180.0 + turning_degrees

        angles.append(sg.radians(internal_angle))

    return angles


canvas = sg.Canvas()
d = 65
pi = sg.pi
rec = sg.Rectangle((0, 0), 100, 100, fill=False)
pattern = rec.translate(d, d, reps=3)
pattern.translate(d, -d, reps=3)
lace = sg.Lace(pattern, offset=5)
r1 = 5
r2 = 10
m = len(lace.fragments) - 1
for j, frag in enumerate(lace.fragments):
    # if j !=4:
    #     continue
    vertices = frag.vertices
    edges = frag.edges
    n = len(vertices)
    angles = simetri.geom.polygons.polygon.polygon_internal_angles(vertices)
    d_vert_radius2 = {}
    for i, angle in enumerate(angles):
        if angle < pi:
            rad = r1
        else:
            rad = r2
        d_vert_radius2[i] = rad
    for i, sec in enumerate(frag.sections):
        edge = edges[i]
        start = sec.start.point
        end = sec.end.point
        if angles[i] < pi:
            # inside corner
            rad = r1
        else:
            rad = r2

        if sec.end.overlap is not None:
            if simetri.geom.points.point_utils.close_points_square(
                edge[1], end
            ):
                if i + 1 in d_vert_radius2:
                    del d_vert_radius2[i + 1]
            else:
                d_vert_radius2.pop(i, None)

        if sec.start.overlap is not None:
            if simetri.geom.points.point_utils.close_points_square(
                edge[1], start
            ):
                if i + 1 in d_vert_radius2:
                    del d_vert_radius2[i + 1]
            else:
                d_vert_radius2.pop(i, None)
    color = sg.blue
    if j == 0:
        color = sg.green
    elif j == m:
        color = sg.red

    canvas.draw(fillet_corners2(frag, d_vert_radius2), color=color)
    canvas.text(str(j), frag.midpoint)


plait = lace.plaits[0]

for plait in lace.plaits:
    ends = plait.ends
    edges = plait.edges
    d_vert_radius = {}
    for conn in plait.connections:
        v1, v2 = conn
        if v1 not in ends and v2 not in ends:
            if simetri.geom.points.point_utils.distance(
                *edges[v1]
            ) > simetri.geom.points.point_utils.distance(*edges[v2 - 1]):
                d_vert_radius[v1] = r2
                d_vert_radius[v2] = r1

            else:
                d_vert_radius[v1] = r1
                d_vert_radius[v2] = r2

    canvas.draw(fillet_corners2(plait, d_vert_radius), fill=False)

F = sg.letter_F().scale(2)
d_v_r = {}

for i in range(len(F.vertices)):
    rad = 8 if i % 2 else 16
    d_v_r[i] = rad
canvas.draw(fillet_corners2(F, d_v_r))

canvas.save("c:/tmp/lace_radius.svg", overwrite=True)
# canvas.save("c:/tmp/F_radius.svg", overwrite=True)
