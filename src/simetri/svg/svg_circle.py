"""Circle and circular-arc helpers for SVG path construction.

Utilities for intersecting circles/arcs and converting arc parameters to
SVG ``A`` path commands.
"""

import math


def circle_intersections(circ1, circ2):
    """Return intersection points of two circles.

    Args:
        circ1: ``((x, y), radius)`` for the first circle.
        circ2: ``((x, y), radius)`` for the second circle.

    Returns:
        list: Zero or two ``(x, y)`` intersection points.
    """
    x1, y1 = circ1[0][:2]
    x2, y2 = circ2[0][:2]
    r1 = circ1[1]
    r2 = circ2[1]
    dx, dy = x2 - x1, y2 - y1
    d = math.hypot(dx, dy)

    # No intersection or one circle inside the other
    if d >= r1 + r2 or d <= abs(r1 - r2) or d == 0:
        return []

    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = math.sqrt(r1 * r1 - a * a)

    xm = x1 + a * dx / d
    ym = y1 + a * dy / d

    rx = -dy * (h / d)
    ry = dx * (h / d)

    return [(xm + rx, ym + ry), (xm - rx, ym - ry)]


def svg_arc_to(x, y, r, sweep):
    """Format an SVG circular arc command ending at ``(x, y)``.

    Args:
        x: Arc end x.
        y: Arc end y.
        r: Arc radius.
        sweep: SVG sweep-flag (0 or 1).

    Returns:
        str: ``A …`` path fragment.
    """
    return f"A {r} {r} 0 0 {sweep} {x:.4f} {y:.4f}"


def is_inside(p, circle):
    """Return whether point ``p`` lies inside or on ``circle``.

    Args:
        p: Point ``(x, y)``.
        circle: ``((cx, cy), radius)``.

    Returns:
        bool: True if the point is within the circle (with a small tolerance).
    """
    return (
        math.hypot(p[0] - circle[0][0], p[1] - circle[0][1]) < circle[1] + 1e-6
    )


def union_of_circles(circles):
    """Build an SVG path for the union outline of intersecting circles.

    Args:
        circles: Sequence of ``((x, y), r)`` circles. Currently handles
            pairs of intersecting circles most completely.

    Returns:
        str: SVG path ``d`` string, or empty if no outline is produced.
    """
    if not circles:
        return ""

    # For simplicity, this implementation will handle the union of two circles.
    # A general solution for n circles is significantly more complex.
    if len(circles) == 2:
        c1, c2 = circles
        intersections = circle_intersections(c1, c2)

        if not intersections:
            # One circle is inside another or they are separate.
            d = math.hypot(c1[0][0] - c2[0][0], c1[0][1] - c2[0][1])
            if d <= c1[1] - c2[1]:  # c2 is inside c1
                return f"M {c1[0][0] - c1[1]},{c1[0][1]} a {c1[1]},{c1[1]} 0 1,0 {2 * c1[1]},0 a {c1[1]},{c1[1]} 0 1,0 {-2 * c1[1]},0"
            if d <= c2[1] - c1[1]:  # c1 is inside c2
                return f"M {c2[0][0] - c2[1]},{c2[0][1]} a {c2[1]},{c2[1]} 0 1,0 {2 * c2[1]},0 a {c2[1]},{c2[1]} 0 1,0 {-2 * c2[1]},0"
            # Circles are separate, this implementation will not draw a path for disjoint circles.
            return ""

        p1, p2 = intersections

        # Determine which arc to draw for each circle
        # We need a point on each arc to check if it's inside the other circle.
        # A simple way is to take the midpoint of the arc.
        # However, a simpler logic for two circles is to connect the arcs that are "outside" the other circle.

        # Angle of intersections relative to circle centers
        a1_p1 = math.atan2(p1[1] - c1[0][1], p1[0] - c1[0][0])
        a1_p2 = math.atan2(p2[1] - c1[0][1], p2[0] - c1[0][0])
        # a2_p1 = math.atan2(p1[1] - c2[0][1], p1[0] - c2[0][0])
        # a2_p2 = math.atan2(p2[1] - c2[0][1], p2[0] - c2[0][1])

        # A point on the arc of c1 between p1 and p2
        mid_angle1 = (a1_p1 + a1_p2) / 2
        point_on_arc1 = (
            c1[0][0] + c1[1] * math.cos(mid_angle1),
            c1[0][1] + c1[1] * math.sin(mid_angle1),
        )

        path = []
        if not is_inside(point_on_arc1, c2):
            path.append(f"M {p1[0]:.4f} {p1[1]:.4f}")
            path.append(svg_arc_to(p2[0], p2[1], c1[1], 0))
            path.append(svg_arc_to(p1[0], p1[1], c2[1], 0))
        else:
            path.append(f"M {p1[0]:.4f} {p1[1]:.4f}")
            path.append(svg_arc_to(p2[0], p2[1], c1[1], 1))
            path.append(svg_arc_to(p1[0], p1[1], c2[1], 1))

        return " ".join(path)


def arc_arc_intersection(arc1, arc2):
    """Find intersection points of two circular arcs.

    Args:
        arc1: ``(center, radius, start_angle, sweep_angle)`` in radians.
        arc2: Same format as ``arc1``.

    Returns:
        list: Intersection points that lie on both arcs.
    """
    c1, r1, start1, sweep1 = arc1
    c2, r2, start2, sweep2 = arc2

    # Find intersections of the parent circles
    intersections = circle_intersections((c1, r1), (c2, r2))
    if not intersections:
        return []

    valid_intersections = []
    for p in intersections:
        # Check if the intersection point is on the first arc
        angle1 = math.atan2(p[1] - c1[1], p[0] - c1[0])
        end1 = start1 + sweep1
        on_arc1 = False
        if sweep1 >= 0:
            if (
                start1 <= angle1 <= end1
                or start1 <= angle1 + 2 * math.pi <= end1
            ):
                on_arc1 = True
        else:  # Negative sweep
            if (
                end1 <= angle1 <= start1
                or end1 <= angle1 + 2 * math.pi <= start1
            ):
                on_arc1 = True

        # Check if the intersection point is on the second arc
        angle2 = math.atan2(p[1] - c2[1], p[0] - c2[0])
        end2 = start2 + sweep2
        on_arc2 = False
        if sweep2 >= 0:
            if (
                start2 <= angle2 <= end2
                or start2 <= angle2 + 2 * math.pi <= end2
            ):
                on_arc2 = True
        else:  # Negative sweep
            if (
                end2 <= angle2 <= start2
                or end2 <= angle2 + 2 * math.pi <= start2
            ):
                on_arc2 = True

        if on_arc1 and on_arc2:
            valid_intersections.append(p)

    return valid_intersections


def convert_arc(center, radius, start_angle, sweep_angle):
    """Convert center/radius/angles to an SVG path with a single arc.

    Args:
        center: Arc center ``(x, y)``.
        radius: Arc radius.
        start_angle: Start angle in radians.
        sweep_angle: Signed sweep in radians.

    Returns:
        str: SVG path ``d`` string starting with ``M`` then ``A``.
    """
    # Calculate start point
    start_x = center[0] + radius * math.cos(start_angle)
    start_y = center[1] + radius * math.sin(start_angle)

    # Calculate end point
    end_angle = start_angle + sweep_angle
    end_x = center[0] + radius * math.cos(end_angle)
    end_y = center[1] + radius * math.sin(end_angle)

    # Determine large-arc-flag (1 if the arc spans more than 180 degrees)
    large_arc_flag = 1 if abs(sweep_angle) > math.pi else 0

    # Determine sweep-flag (1 if positive/counter-clockwise, 0 if negative/clockwise)
    sweep_flag = 1 if sweep_angle > 0 else 0

    # Generate SVG path
    return f"M {start_x:.4f} {start_y:.4f} A {radius} {radius} 0 {large_arc_flag} {sweep_flag} {end_x:.4f} {end_y:.4f}"


def convert_svg_arc(
    start_point, end_point, rx, ry, x_axis_rotation, large_arc_flag, sweep_flag
):
    """Convert SVG endpoint arc parameters to center parameterization.

    Assumes a circular arc (``rx == ry``).

    Args:
        start_point: Arc start ``(x, y)``.
        end_point: Arc end ``(x, y)``.
        rx: X radius.
        ry: Y radius (treated as circular with ``rx``).
        x_axis_rotation: Unused for circular arcs; kept for SVG parity.
        large_arc_flag: SVG large-arc flag.
        sweep_flag: SVG sweep flag.

    Returns:
        tuple: ``((cx, cy), start_angle, sweep_angle)``.
    """
    x1, y1 = start_point[:2]
    x2, y2 = end_point[:2]
    r = rx  # Assume circular arc

    # If start and end points are the same, no arc
    if math.hypot(x2 - x1, y2 - y1) < 1e-10:
        return ((x1, y1), 0, 0)

    # Calculate the center point
    # Midpoint between start and end
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2

    # Distance from midpoint to start
    d = math.hypot(x2 - x1, y2 - y1) / 2

    # If radius is too small, adjust it
    if d > r:
        r = d

    # Distance from midpoint to center
    h = math.sqrt(r * r - d * d)

    # Perpendicular direction from midpoint
    if sweep_flag == large_arc_flag:
        cx = mx - h * (y2 - y1) / (2 * d)
        cy = my + h * (x2 - x1) / (2 * d)
    else:
        cx = mx + h * (y2 - y1) / (2 * d)
        cy = my - h * (x2 - x1) / (2 * d)

    # Calculate start and end angles
    start_angle = math.atan2(y1 - cy, x1 - cx)
    end_angle = math.atan2(y2 - cy, x2 - cx)

    # Calculate sweep angle
    sweep_angle = end_angle - start_angle

    # Adjust for sweep direction and large arc flag
    if sweep_flag:
        if sweep_angle < 0:
            sweep_angle += 2 * math.pi
        if large_arc_flag and sweep_angle < math.pi:
            sweep_angle -= 2 * math.pi
    else:
        if sweep_angle > 0:
            sweep_angle -= 2 * math.pi
        if large_arc_flag and sweep_angle > -math.pi:
            sweep_angle += 2 * math.pi

    return ((cx, cy), start_angle, sweep_angle)


def circles_to_arcs(circle1, circle2):
    """Split two intersecting circles into four SVG arc path strings.

    Args:
        circle1: ``((cx, cy), radius)``.
        circle2: ``((cx, cy), radius)``.

    Returns:
        list: Four SVG path strings, or empty if the circles do not intersect
        at exactly two points.
    """
    # Find intersection points
    intersections = circle_intersections(circle1, circle2)

    if len(intersections) != 2:
        # If circles don't intersect at exactly 2 points, return empty or full circles
        return []

    p1, p2 = intersections
    c1, r1 = circle1
    c2, r2 = circle2

    # Calculate angles of intersection points for circle1
    angle1_p1 = math.atan2(p1[1] - c1[1], p1[0] - c1[0])
    angle1_p2 = math.atan2(p2[1] - c1[1], p2[0] - c1[0])

    # Calculate sweep angle for the two arcs of circle1
    sweep1_a = (angle1_p2 - angle1_p1) % (2 * math.pi)
    sweep1_b = 2 * math.pi - sweep1_a

    # Create first two arcs from circle1
    arc1_a = convert_arc(c1, r1, angle1_p1, sweep1_a)
    arc1_b = convert_arc(c1, r1, angle1_p2, sweep1_b)

    # Calculate angles of intersection points for circle2
    angle2_p1 = math.atan2(p1[1] - c2[1], p1[0] - c2[0])
    angle2_p2 = math.atan2(p2[1] - c2[1], p2[0] - c2[0])

    # Calculate sweep angle for the two arcs of circle2
    sweep2_a = (angle2_p2 - angle2_p1) % (2 * math.pi)
    sweep2_b = 2 * math.pi - sweep2_a

    # Create second two arcs from circle2
    arc2_a = convert_arc(c2, r2, angle2_p1, sweep2_a)
    arc2_b = convert_arc(c2, r2, angle2_p2, sweep2_b)

    return [arc1_a, arc1_b, arc2_a, arc2_b]


arc1 = ((0, 0), 50, 0, math.pi)
arc2 = ((20, 0), 50, 0, math.pi)

arcs = [
    convert_arc((20, 20), 40, 0, math.pi / 2),
    convert_arc((-20, -20), 40, math.pi, math.pi / 4),
]
for arc in arcs:
    print(f' <path d="{arc}" style="fill: none; stroke:black"/>')
