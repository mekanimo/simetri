from typing import List, Tuple, Dict
import re
from math import sqrt, cos, sin, acos, radians, pi, degrees

import numpy as np

from ..graphics.common import Point
from ..geometry.geometry import (
    offset_polygon,
    double_offset_polygons,
    double_offset_polylines,
)
from ..graphics.all_enums import PathOperation as PathOps


# Helper to format floats to avoid excessive precision in SVG
def fmt(val, digits=3):
    return f"{val:.{digits}f}".rstrip("0").rstrip(".")


def round_corner(points: List["Point"], radius: float) -> str:
    """Given a list of three points generates an svg path corresponding to a
    polyline with two segments and a rounded corner between them"""
    if len(points) != 3:
        raise ValueError("round_corner expects exactly 3 points")

    p0, p1, p2 = points

    v1 = np.array(p1) - np.array(p0)
    v2 = np.array(p2) - np.array(p1)

    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)

    if l1 < 1e-9 or l2 < 1e-9:
        return f"L {fmt(p1[0])},{fmt(p1[1])}"

    # Angle calculation
    dot_val = np.dot(-v1, v2)
    div = l1 * l2
    if div == 0:
        return f"L {fmt(p1[0])},{fmt(p1[1])}"

    cos_theta = dot_val / div
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Distance from corner to tangent points
    dist = radius * np.tan(theta / 2)

    # Clamp distance to half of the shortest segment to avoid overlap
    limit = min(l1 / 2, l2 / 2)
    eff_radius = radius
    if dist > limit:
        dist = limit
        # Recalculate effective radius if constrained
        if abs(theta) > 1e-9:
            eff_radius = dist / np.tan(theta / 2)
        else:
            eff_radius = 0
    else:
        eff_radius = radius

    # Tangent points
    p_start = np.array(p1) - (v1 / l1) * dist
    p_end = np.array(p1) + (v2 / l2) * dist

    # Sweep flag: 1 if turning "right" (positive cross product in SVG coords)
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    sweep = 1 if cross > 0 else 0

    return f"L {fmt(p_start[0])},{fmt(p_start[1])} A {fmt(eff_radius)},{fmt(eff_radius)} 0 0 {sweep} {fmt(p_end[0])},{fmt(p_end[1])}"


def round_corners(
    points: List["Point"],
    radius: float = 0,
    fillets: List[tuple[int, float]] = None,
    closed: bool = False,
) -> str:
    """Given a list of points generates an svg path corresponding to a polyline
    with rounded corners

    Args:
        points: A list of points (x, y)
        radius: The default radius for all corners (default: 0)
        fillets: A list of tuples (index, radius) to override the default radius for specific corners
        closed: If True, the path is closed and all corners are rounded (default: False)
    """
    if len(points) < 3:
        if closed:
            return "M " + " L ".join(f"{p[0]},{p[1]}" for p in points) + " Z"
        return "M " + " L ".join(f"{p[0]},{p[1]}" for p in points)

    fillet_dict = dict(fillets) if fillets else {}

    if closed:
        # Start at midpoint of first segment
        p0 = points[0]
        p1 = points[1]
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        path = [f"M {fmt(mx)},{fmt(my)}"]

        N = len(points)
        # Order: 1, 2, ..., N-1, 0
        indices = list(range(1, N)) + [0]
        for i in indices:
            r = fillet_dict.get(i, radius)
            if r == 0:
                path.append(f"L {fmt(points[i][0])},{fmt(points[i][1])}")
            else:
                pts = [points[(i - 1) % N], points[i], points[(i + 1) % N]]
                path.append(round_corner(pts, r))
        path.append("Z")
        return " ".join(path)

    path = [f"M {points[0][0]},{points[0][1]}"]

    for i in range(1, len(points) - 1):
        r = fillet_dict.get(i, radius)
        if r == 0:
            path.append(f"L {fmt(points[i][0])},{fmt(points[i][1])}")
        else:
            path.append(round_corner(points[i - 1 : i + 2], r))

    path.append(f"L {points[-1][0]},{points[-1][1]}")
    return " ".join(path)


def _extract_vertices(svg_path: str) -> Tuple[List[Point], bool]:
    tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+)", svg_path)
    points = []
    i = 0
    closed = False

    current_x = 0
    current_y = 0

    while i < len(tokens):
        t = tokens[i]
        lower_t = t.lower()

        # Check if token is a command
        if lower_t in ["m", "l", "a", "h", "v", "z"]:
            cmd = lower_t
            i += 1
        else:
            # Assume Implicit L
            cmd = "l"

        if cmd == "z":
            closed = True
            continue

        if cmd == "m" or cmd == "l":
            if i + 1 >= len(tokens):
                break
            x = float(tokens[i])
            y = float(tokens[i + 1])
            points.append((x, y))
            current_x, current_y = x, y
            i += 2
        elif cmd == "a":
            if i + 6 >= len(tokens):
                break
            x = float(tokens[i + 5])
            y = float(tokens[i + 6])
            points.append((x, y))
            current_x, current_y = x, y
            i += 7
        elif cmd == "h":
            if i >= len(tokens):
                break
            x = float(tokens[i])
            points.append((x, current_y))
            current_x = x
            i += 1
        elif cmd == "v":
            if i >= len(tokens):
                break
            y = float(tokens[i])
            points.append((current_x, y))
            current_y = y
            i += 1

    # Check if last point is same as first for closed
    if points and len(points) > 1:
        p0 = points[0]
        pend = points[-1]
        dist = (p0[0] - pend[0]) ** 2 + (p0[1] - pend[1]) ** 2
        if dist < 1e-9:
            closed = True
            points.pop()  # Remove duplicate point

    return points, closed


def _points_to_svg(points: List[Point], closed: bool) -> str:
    if not points:
        return ""
    parts = [f"M {fmt(points[0][0])},{fmt(points[0][1])}"]
    for p in points[1:]:
        parts.append(f"L {fmt(p[0])},{fmt(p[1])}")
    if closed:
        parts.append("Z")
    return " ".join(parts)


def double_lines(svg_path: str, offset: float, offset_side: str = "outer") -> str:
    """Given an svg path, creates an offset contour using the offset value.
    Offset can be applied inward, outward, or centered.

    Returns an SVG path with double lines.
    """
    points, closed = _extract_vertices(svg_path)
    if not points:
        return ""

    res_paths = []

    if closed:
        if offset_side == "centered" or offset_side == "both":
            polys = double_offset_polygons(points, offset)
            for p in polys:
                res_paths.append(_points_to_svg(p, True))
        elif offset_side == "outer":
            # Geometry module uses offset for outer?
            # offset_polygon uses -offset internally if passed positive.
            # Assuming offset_polygon(points, positive_val) does "default" offset
            p = offset_polygon(points, offset)
            res_paths.append(_points_to_svg(p, True))
        elif offset_side == "inner":
            p = offset_polygon(points, -offset)
            res_paths.append(_points_to_svg(p, True))
    else:
        # Open path
        p1, p2 = double_offset_polylines(points, offset)
        if offset_side == "centered" or offset_side == "both":
            res_paths.append(_points_to_svg(p1, False))
            res_paths.append(_points_to_svg(p2, False))
        elif offset_side == "outer":
            res_paths.append(_points_to_svg(p1, False))
        elif offset_side == "inner":
            res_paths.append(_points_to_svg(p2, False))

    return " ".join(res_paths)


def set_style(svg_shape: str, d_style: dict) -> str:
    """Given an svg shape (line, circle, ellipse, or path), applies the
    style values (line style, fill style, and gradient) given as a dictionary.
    Returns a string representing the stylized svg_shape.
    """
    mapping = {
        "stroke_width": "stroke-width",
        "stroke_dasharray": "stroke-dasharray",
        "stroke_linecap": "stroke-linecap",
        "stroke_linejoin": "stroke-linejoin",
        "fill_opacity": "fill-opacity",
        "stroke_opacity": "stroke-opacity",
        "stop_color": "stop-color",
        "stop_opacity": "stop-opacity",
        "clip_path": "clip-path",
        "font_family": "font-family",
        "font_size": "font-size",
        "text_anchor": "text-anchor",
        "dominant_baseline": "dominant-baseline",
        "even_odd": "fill-rule",
    }

    attribs = []
    for k, v in d_style.items():
        if v is None:
            continue
        key = mapping.get(k, k.replace("_", "-"))
        attribs.append(f'{key}="{v}"')

    style_str = " ".join(attribs)
    if not style_str:
        return svg_shape

    # Inject into the opening tag
    # Capture (start of tag)(end of tag which is either > or />)
    # The first group matches <tagname and attributes lazily until it sees the closing part
    return re.sub(r"(<\w+[^>]*?)(\s*/?>)", f"\\1 {style_str}\\2", svg_shape, count=1)


def convert_arc(center, radius, start_angle, sweep_angle):
    """Given an arc by center, radius, start and sweep angles,
    returns and svg path with an arc."""
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
    """Given an SVG arc in endpoint parameterization,
    returns ((cx, cy), start_angle, sweep_angle) in center parameterization.
    Assumes circular arcs (rx == ry).
    """
    x1, y1 = start_point
    x2, y2 = end_point
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


def svg_path_to_linpath(svg_path: str) -> "LinPath":
    """Given an SVG path returns the equivalent LinPath object."""
    from ..graphics.path import LinPath

    if not svg_path:
        return LinPath()

    # Tokenizer
    tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", svg_path)

    start_point = (0.0, 0.0)
    idx = 0

    # Handle optional start M strictly for initialization
    if idx < len(tokens) and tokens[idx].lower() == "m":
        try:
            x = float(tokens[idx + 1])
            y = float(tokens[idx + 2])
            start_point = (x, y)
            idx = 3
        except IndexError:
            pass

    lp = LinPath(start=start_point)

    current_cmd = ""
    i = 0

    if idx == 3:
        i = 3
        first_cmd = tokens[0]
        if first_cmd == "M":
            current_cmd = "L"
        if first_cmd == "m":
            current_cmd = "l"

    while i < len(tokens):
        t = tokens[i]

        if t.isalpha():
            current_cmd = t
            i += 1
        else:
            if not current_cmd:
                i += 1
                continue
            # Implicit command logic
            if current_cmd == "M":
                current_cmd = "L"
            elif current_cmd == "m":
                current_cmd = "l"

        cmd_lower = current_cmd.lower()
        is_rel = current_cmd == cmd_lower

        def get_nums(count):
            nonlocal i
            nums = []
            for _ in range(count):
                if i < len(tokens):
                    try:
                        nums.append(float(tokens[i]))
                    except ValueError:
                        break
                    i += 1
                else:
                    break
            if len(nums) < count:
                return None
            return nums

        if cmd_lower == "z":
            lp.close()

        elif cmd_lower == "m":
            coords = get_nums(2)
            if coords:
                if is_rel:
                    lp.r_move(*coords)
                else:
                    lp.move_to(coords)

        elif cmd_lower == "l":
            coords = get_nums(2)
            if coords:
                if is_rel:
                    lp.r_line(*coords)
                else:
                    lp.line_to(coords)

        elif cmd_lower == "h":
            coords = get_nums(1)
            if coords:
                val = coords[0]
                if is_rel:
                    lp.r_h_line(val)
                else:
                    lp.h_line_to(val)

        elif cmd_lower == "v":
            coords = get_nums(1)
            if coords:
                val = coords[0]
                if is_rel:
                    lp.r_v_line(val)
                else:
                    lp.v_line_to(val)

        elif cmd_lower == "c":
            coords = get_nums(6)
            if coords:
                c1 = (coords[0], coords[1])
                c2 = (coords[2], coords[3])
                end = (coords[4], coords[5])

                if is_rel:
                    cur_x, cur_y = lp.pos
                    c1 = (cur_x + c1[0], cur_y + c1[1])
                    c2 = (cur_x + c2[0], cur_y + c2[1])
                    end = (cur_x + end[0], cur_y + end[1])

                lp.cubic_to(c1, c2, end)

        elif cmd_lower == "s":
            coords = get_nums(4)
            if coords:
                c2 = (coords[0], coords[1])
                end = (coords[2], coords[3])

                if is_rel:
                    cur_x, cur_y = lp.pos
                    c2 = (cur_x + c2[0], cur_y + c2[1])
                    end = (cur_x + end[0], cur_y + end[1])

                lp.mirror_cubic_to(c2, end)

        elif cmd_lower == "q":
            coords = get_nums(4)
            if coords:
                c1 = (coords[0], coords[1])
                end = (coords[2], coords[3])
                if is_rel:
                    cur_x, cur_y = lp.pos
                    c1 = (cur_x + c1[0], cur_y + c1[1])
                    end = (cur_x + end[0], cur_y + end[1])
                lp.quad_to(c1, end)

        elif cmd_lower == "t":
            coords = get_nums(2)
            if coords:
                end = (coords[0], coords[1])

                if is_rel:
                    cur_x, cur_y = lp.pos
                    end = (cur_x + end[0], cur_y + end[1])

                lp.mirror_quad_to(end)

        elif cmd_lower == "a":
            coords = get_nums(7)
            if coords:
                rx, ry = coords[0], coords[1]
                rot_deg = coords[2]
                large_arc = bool(coords[3])
                sweep = bool(coords[4])
                end = (coords[5], coords[6])

                if is_rel:
                    cur_x, cur_y = lp.pos
                    end = (cur_x + end[0], cur_y + end[1])

                params = _get_svg_arc_params(
                    lp.pos, rx, ry, rot_deg, large_arc, sweep, end
                )

                if params["type"] == "line":
                    lp.line_to(params["end"])
                elif params["type"] == "arc":
                    lp.arc(
                        params["rx"],
                        params["ry"],
                        params["start_angle"],
                        params["span_angle"],
                        rot_angle=params["rot_angle"],
                    )

    return lp


def linpath_to_svg_path(linpath: "LinPath") -> str:
    """Given a LinPath instance, returns the equivalent SVG path string."""
    parts = [f"M {fmt(linpath.start[0])},{fmt(linpath.start[1])}"]

    # Iterate through operations and convert to SVG path commands
    obj_idx = 0
    PO = PathOps

    for op in linpath.operations:
        if isinstance(op, tuple):
            # Style operation - skip
            continue

        st = op.subtype
        data = op.data

        # Current geometry object (if applicable)
        current_obj = (
            linpath.objects[obj_idx] if obj_idx < len(linpath.objects) else None
        )

        if st in [PO.MOVE_TO, PO.R_MOVE]:
            # data is point (x,y)
            parts.append(f"M {fmt(data[0])},{fmt(data[1])}")

        elif st in [
            PO.LINE_TO,
            PO.R_LINE,
            PO.H_LINE,
            PO.V_LINE,
            PO.R_H_LINE,
            PO.R_V_LINE,
            PO.FORWARD,
        ]:
            # data is (start, end)
            end = data[1]
            parts.append(f"L {fmt(end[0])},{fmt(end[1])}")

        elif st == PO.SEGMENTS:
            # data is (start, points_list)
            for p in data[1]:
                parts.append(f"L {fmt(p[0])},{fmt(p[1])}")

        elif st in [PO.CUBIC_TO, PO.BLEND_CUBIC]:
            # data: (start, c1, c2, end)
            c1, c2, end = data[1], data[2], data[3]
            parts.append(
                f"C {fmt(c1[0])},{fmt(c1[1])} {fmt(c2[0])},{fmt(c2[1])} {fmt(end[0])},{fmt(end[1])}"
            )

        elif st in [PO.QUAD_TO, PO.BLEND_QUAD]:
            # data: (start, c1, end)
            c1, end = data[1], data[2]
            parts.append(f"Q {fmt(c1[0])},{fmt(c1[1])} {fmt(end[0])},{fmt(end[1])}")

        elif st in [PO.ARC, PO.BLEND_ARC]:
            # data: (pos, tangent_angle, rx, ry, start_angle, span_angle, rot_angle, points)
            rx, ry = data[2], data[3]
            span = data[5]
            rot = degrees(data[6])
            points = data[7]
            end = points[-1]
            large_arc = 1 if abs(span) > pi else 0
            # Simetri convention: span > 0 is CCW.
            # SVG sweep-flag: 1 is positive-angle direction (CW in y-down).
            sweep = 1 if span > 0 else 0
            parts.append(
                f"A {fmt(rx)} {fmt(ry)} {fmt(rot)} {large_arc} {sweep} {fmt(end[0])},{fmt(end[1])}"
            )

        elif st == PO.CLOSE:
            parts.append("Z")

        elif st in [PO.SINE, PO.BLEND_SINE]:
            # data[0] is points
            for p in data[0]:
                parts.append(f"L {fmt(p[0])},{fmt(p[1])}")

        elif st == PO.HOBBY_TO:
            # Use the resolved shape vertices from objects
            if current_obj:
                # Skip the first point since it should match current pos
                verts = current_obj.vertices
                for p in verts[1:]:
                    parts.append(f"L {fmt(p[0])},{fmt(p[1])}")

        obj_idx += 1

    return " ".join(parts)


def linpath_points(linpath: "LinPath", delta: float) -> List[Tuple[float, float]]:
    """Given a LinPath instance, returns a list of points separated by the given length.
    It is not possible to create the points with the exact delta. Delta will be
    adjusted for each part of the LinPath accordingly.
    """
    points = []
    vertices = linpath.vertices

    if not vertices:
        return points

    # Always include the first point
    points.append((vertices[0][0], vertices[0][1]))

    # For each segment between consecutive vertices
    for i in range(len(vertices) - 1):
        p1 = vertices[i]
        p2 = vertices[i + 1]

        # Calculate segment length
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = sqrt(dx * dx + dy * dy)

        if length < 1e-9:
            continue

        # Calculate number of segments (round to nearest integer)
        n_segments = max(1, round(length / delta))

        # Generate intermediate points (not including start point, but including end point)
        for j in range(1, n_segments + 1):
            t = j / n_segments
            x = p1[0] + t * dx
            y = p1[1] + t * dy
            points.append((x, y))

    return points


def svg_path_points(svg_path: str, delta: float) -> List[Tuple[float, float]]:
    """Given an SVG path string, returns a list of points separated by the given length.
    It is not possible to create the points with the exact delta. Delta will be
    adjusted for each part of the SVG path accordingly.
    """
    lp = svg_path_to_linpath(svg_path)
    return linpath_points(lp, delta)


def _get_svg_arc_params(start, rx, ry, phi_deg, fA, fs, end):
    """Convert SVG arc parameters to LinPath arc parameters."""
    x1, y1 = start
    x2, y2 = end

    rx = abs(rx)
    ry = abs(ry)
    phi = radians(phi_deg)

    if rx == 0 or ry == 0:
        return {"type": "line", "end": end}

    if x1 == x2 and y1 == y2:
        return {"type": "none"}

    # Matrix for rotation
    cos_phi = cos(phi)
    sin_phi = sin(phi)

    # Step 1: Prime coords
    dx = (x1 - x2) / 2
    dy = (y1 - y2) / 2
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    # Radii check
    lamb = (x1p**2) / (rx**2) + (y1p**2) / (ry**2)
    if lamb > 1:
        s = sqrt(lamb)
        rx *= s
        ry *= s

    # Step 2: Center prime
    sign = -1 if fA == fs else 1
    num = (rx**2 * ry**2) - (rx**2 * y1p**2) - (ry**2 * x1p**2)
    den = (rx**2 * y1p**2) + (ry**2 * x1p**2)
    # precision check
    if abs(num) < 1e-9:
        num = 0
    if abs(den) < 1e-9:
        coef = 0
    else:
        coef = sign * sqrt(max(0, num / den))

    cxp = coef * (rx * y1p / ry)
    cyp = coef * (-ry * x1p / rx)

    # Step 4: Angles
    def vector_angle(ux, uy, vx, vy):
        sign = 1 if (ux * vy - uy * vx) >= 0 else -1
        dot = ux * vx + uy * vy
        length = sqrt(ux**2 + uy**2) * sqrt(vx**2 + vy**2)
        if length == 0:
            return 0
        val = max(-1, min(1, dot / length))
        return sign * acos(val)

    # start vector
    ux = (x1p - cxp) / rx
    uy = (y1p - cyp) / ry
    theta1 = vector_angle(1, 0, ux, uy)

    # dtheta
    vx = (-x1p - cxp) / rx
    vy = (-y1p - cyp) / ry
    dtheta = vector_angle(ux, uy, vx, vy)

    if not fs and dtheta > 0:
        dtheta -= 2 * pi
    elif fs and dtheta < 0:
        dtheta += 2 * pi

    return {
        "type": "arc",
        "rx": rx,
        "ry": ry,
        "start_angle": theta1,
        "span_angle": dtheta,
        "rot_angle": phi,
    }


# verts = ((0.0, 0.0), (20.0, 0.0), (20.0, 40.0), (40.0, 40.0), (40.0, 60.0), (20.0, 60.0), (20.0, 80.0), (50.0, 80.0), (50.0, 100.0), (0.0, 100.0))

# print(round_corners(verts, 5, closed=True))

# offset, offset_side: outer, inner, both
