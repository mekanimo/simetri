'''Functions for working with ellipses.'''

from math import cos, sin, pi, atan2, sqrt, degrees

from scipy.special import ellipeinc
import numpy as np

from ..graphics.shape import Shape, custom_attributes
from ..graphics.common import Point
from ..graphics.all_enums import Types
from ..settings.settings import defaults
from .geometry import distance, positive_angle


ndarray = np.ndarray

class Arc(Shape):
    """A circular or elliptic arc defined by a center, radius, radius2, start angle,
    and end angle. If radius2 is not provided, the arc is a circular arc."""
    def __init__(
        self,
        center: Point,
        start_angle: float,
        span_angle: float,
        radius: float,
        radius2: float,
        n_points: int = None,
        rot_angle: float = 0,
        xform_matrix: ndarray = None,
        **kwargs,
    ):
        if n_points is None:
            n_points = defaults["n_arc_points"]

        vertices = [tuple(p) for p in elliptic_arc_points(center, radius, radius2, start_angle,
                                                            span_angle, n_points)]
        super().__init__(vertices, subtype=Types.ARC, xform_matrix=xform_matrix, **kwargs)
        self.center = center
        self.radius = radius
        self.radius2 = radius2
        self.start_angle = start_angle
        self.span_angle = span_angle
        self.start_point = vertices[0]
        self.end_point = vertices[-1]
        self.rot_angle = rot_angle

def arc2(start, start_angle, end_angle, radius1, radius2):
    '''Return the vertices of an elliptic arc.'''
    center = (0, 0)



class Ellipse(Shape):
    """An ellipse defined by center, width, and height."""

    def __init__(self, center: Point, width: float, height: float, angle:float=0,
                                                xform_matrix:ndarray = None, **kwargs) -> None:
        n_points = defaults["n_ellipse_points"]
        vertices = [tuple(p) for p in ellipse_points(center, width / 2, height / 2, n_points)]
        super().__init__(vertices, closed=True, xform_matrix=xform_matrix, **kwargs)
        a = width / 2
        b = height / 2
        self.a = a
        self.b = b
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.smooth = True
        self.closed = True
        self.subtype = Types.ELLIPSE

    # @property
    # def center(self):
    #     """Return the center of the ellipse."""
    #     return self.vertices[0]

    # @center.setter
    # def center(self, new_center: Point):
    #     center = self.center
    #     x_diff = new_center[0] - center[0]
    #     y_diff = new_center[1] - center[1]
    #     for i in range(5):
    #         x, y = self.vertices[i]
    #         self[i] = (x + x_diff, y + y_diff)

    # @property
    # def width(self):
    #     """Return the width of the ellipse."""
    #     vertices = self.vertices
    #     return distance(vertices[0], vertices[1]) * 2

    # @width.setter
    # def width(self, new_width: float):
    #     center = self.center
    #     height = self.height
    #     vertices = []
    #     vertices.append(center)
    #     a = new_width / 2
    #     b = height / 2
    #     vertices.append((center[0] + a, center[1]))
    #     vertices.append((center[0], center[1] + b))
    #     vertices.append((center[0] - a, center[1]))
    #     vertices.append((center[0], center[1] - b))
    #     self[:] = vertices

    # @property
    # def height(self):
    #     """Return the height of the ellipse."""
    #     vertices = self.vertices
    #     return distance(vertices[0], vertices[2]) * 2

    # @height.setter
    # def height(self, new_height: float):
    #     center = self.center
    #     width = self.width
    #     vertices = []
    #     a = width / 2
    #     b = new_height / 2
    #     vertices.append(center)
    #     vertices.append((center[0] + a, center[1]))
    #     vertices.append((center[0], center[1] + b))
    #     vertices.append((center[0] - a, center[1]))
    #     vertices.append((center[0], center[1] - b))
    #     self[:] = vertices

    @property
    def closed(self):
        """Return True ellipse is always closed."""
        return True

    @closed.setter
    def closed(self, value: bool):
        pass

    def copy(self):
        """Return a copy of the ellipse."""
        center = self.center
        width = self.width
        height = self.height
        ellipse = Ellipse(center, width, height)
        # ellipse.style = self.style.copy()
        custom_attribs = custom_attributes(self)
        for attrib in custom_attribs:
            setattr(ellipse, attrib, getattr(self, attrib))

        return ellipse


def ellipse_tangent(a, b, x, y, tol=.001):
    """Calculates the angle of the tangent line to an ellipse at the point (x, y)."""
    if abs((x**2 / a**2) + (y**2 / b**2) - 1) >= tol:
        res = False
    else:
        # res = atan2(-(b**2 * x), (a**2 * y))
        res = atan2((b**2 * x), -(a**2 * y))

    return res


def r_central(a, b, theta):
    '''Return the radius (distance between the center and the intersection point)
    of the ellipse at the given angle.'''
    return (a * b) / sqrt((b * cos(theta))**2 + (a * sin(theta))**2)


def ellipse_line_intersection(a, b, point):
    '''Return the intersection points of an ellipse and a line segment
    connecting the given point to the ellipse center at (0, 0).'''
    # adapted from http://mathworld.wolfram.com/Ellipse-LineIntersection.html
    # a, b is the ellipse width/2 and height/2 and (x_0, y_0) is the point

    x_0, y_0 = point[:2]
    x =	((a * b) / (sqrt(a**2 * y_0**2 + b**2 * x_0**2))) * x_0
    y =	((a * b) / (sqrt(a**2 * y_0**2 + b**2 * x_0**2))) * y_0

    return [(x, y), (-x, -y)]


def elliptic_arc_points(center, rx, ry, start_angle, span_angle, num_points):
    """Generate points on an elliptic arc.
    These are generated from the parametric equations of the ellipse.
    They are not evenly spaced.
    Args:
        center (tuple): (x, y) coordinates of the ellipse center.
        rx (float): Length of the semi-major axis.
        ry (float): Length of the semi-minor axis.
        start_angle (float): Starting angle of the arc.
        end_angle (float): Ending angle of the arc.
        num_points (int): Number of points to generate.

    Returns:
        numpy.ndarray: Array of (x, y) coordinates of the ellipse points.
    """
    start_angle = positive_angle(start_angle)
    clockwise = span_angle < 0
    if clockwise:
        if start_angle + span_angle < 0:
            end_angle = positive_angle(start_angle + span_angle)
            t0 = get_ellipse_t_for_angle(end_angle, rx, ry)
            t1 = get_ellipse_t_for_angle(2*pi, rx, ry)
            t = np.linspace(t0, t1, num_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            slice1 = np.column_stack((x, y))
            t0 = get_ellipse_t_for_angle(0, rx, ry)
            t1 = get_ellipse_t_for_angle(start_angle, rx, ry)
            t = np.linspace(t0, t1, num_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            slice2 = np.column_stack((x, y))
            res = np.flip(np.concatenate((slice1, slice2)), axis=0)
        else:
            end_angle = start_angle + span_angle
            t0 = get_ellipse_t_for_angle(end_angle, rx, ry)
            t1 = get_ellipse_t_for_angle(start_angle, rx, ry)
            t = np.linspace(t0, t1, num_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            res = np.flip(np.column_stack((x, y)), axis=0)

    else:
        if start_angle + span_angle > 2*pi:
            t0 = get_ellipse_t_for_angle(start_angle, rx, ry)
            t1 = get_ellipse_t_for_angle(2*pi, rx, ry)
            t = np.linspace(t0, t1, num_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            slice1 = np.column_stack((x, y))

            t0 = get_ellipse_t_for_angle(0, rx, ry)
            t1 = get_ellipse_t_for_angle(start_angle + span_angle - 2*pi, rx, ry)
            t = np.linspace(t0, t1, num_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            slice2 = np.column_stack((x, y))

            res = np.concatenate((slice1, slice2))
        else:
            end_angle = start_angle + span_angle
            t0 = get_ellipse_t_for_angle(start_angle, rx, ry)
            t1 = get_ellipse_t_for_angle(end_angle, rx, ry)
            t = np.linspace(t0, t1, num_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            res = np.column_stack((x, y))

    return res




    # t0 = get_ellipse_t_for_angle(start_angle, rx, ry)
    # t1 = get_ellipse_t_for_angle(end_angle, rx, ry)
    # t = np.linspace(t0, t1, num_points)
    # x = center[0] + rx * np.cos(t)
    # y = center[1] + ry * np.sin(t)

    # # to do: rotate the points

    # return np.column_stack((x, y))


def ellipse_points(center, a, b, num_points):
    """Generate points on an ellipse.
    These are generated from the parametric equations of the ellipse.
    They are not evenly spaced.
    Args:
        center (tuple): (x, y) coordinates of the ellipse center.
        a (float): Length of the semi-major axis.
        b (float): Length of the semi-minor axis.
        num_points (int): Number of points to generate.

    Returns:
        numpy.ndarray: Array of (x, y) coordinates of the ellipse points.
    """
    t = np.linspace(0, 2 * pi, num_points)
    x = center[0] + a * np.cos(t)
    y = center[1] + b * np.sin(t)

    return np.column_stack((x, y))


def elliptic_arclength(t_0, t_1, a, b):
    '''Return the arclength of an ellipse between the given parametric angles.
    The ellipse has semi-major axis a and semi-minor axis b.
    from: https://www.johndcook.com/blog/2022/11/02/elliptic-arc-length/
    '''
    m = 1 - (b / a)**2
    t1 = ellipeinc(t_1 - 0.5 * pi, m)
    t0 = ellipeinc(t_0 - 0.5 * pi, m)
    return a*(t1 - t0)


def central_to_parametric_angle(a, b, phi):
    """
    Converts a central angle to a parametric angle on an ellipse.

    Parameters:
    a, b: Semi-major and semi-minor axes of the ellipse
    phi: Angle of the line intersecting the center and the point

    Return:
    t: Parametric angle (in radians)
    """
    t = atan2((a/b) * sin(phi), cos(phi))
    if t < 0:
        t += 2 * pi

    return t


def parametric_to_central_angle(a, b, t):
    """
    Converts a parametric angle on an ellipse to a central angle.

    Parameters:
    a, b: Semi-major and semi-minor axes of the ellipse
    t: Parametric angle (in radians)

    Return:
    phi: Angle of the line intersecting the center and the point
    """
    phi = atan2((b/a) * sin(t), cos(t))
    if phi < 0:
        phi += 2 * pi

    return phi


def ellipse_point(a, b, angle):
    '''Return a point on an ellipse with the given a=width/2, b=height/2, and angle.
    angle is the central-angle and in radians.
    '''
    r = r_central(a, b, angle)

    return (r * cos(angle), r * sin(angle))


def ellipse_param_point(a, b, t):
    '''Return a point on an ellipse with the given a=width/2, b=height/2, and parametric angle.
    t is the parametric angle and in radians.
    '''
    return (a * cos(t), b * sin(t))


def get_ellipse_t_for_angle(angle, a, b):
    """
    Calculates the parameter t for a given angle on an ellipse.

    Args:
        angle: The angle in radians.
        a: Semi-major axis of the ellipse.
        b: Semi-minor axis of the ellipse.

    Returns:
         The parameter t.
    """

    t = atan2(a * sin(angle), b * cos(angle))
    if t < 0:
        t += 2 * pi
    return t


def ellipse_central_angle(t, a, b):
  """
  Calculates the central angle of an ellipse for a given parameter t.

  Args:
    t: The parameter value.
    a: The semi-major axis of the ellipse.
    b: The semi-minor axis of the ellipse.

  Returns:
    The central angle in radians.
  """
  theta = atan2(a * sin(t), b * cos(t))

  return theta




# // svg : [A | a] (rx ry x-axis-rotation large-arc-flag sweep-flag x y)


# see https://www.w3.org/TR/SVG11/implnote.html#ArcConversionEndpointToCenter
# function  radian( ux, uy, vx, vy ) {
#     var  dot = ux * vx + uy * vy;
#     var  mod = Math.sqrt( ( ux * ux + uy * uy ) * ( vx * vx + vy * vy ) );
#     var  rad = Math.acos( dot / mod );
#     if( ux * vy - uy * vx < 0.0 ) {
#         rad = -rad;
#     }
#     return rad;
# }

# //conversion_from_endpoint_to_center_parameterization
# //sample :  svgArcToCenterParam(200,200,50,50,0,1,1,300,200)
# // x1 y1 rx ry φ fA fS x2 y2
# // φ(degree) is degrees as same as SVG not radians.
# // startAngle, deltaAngle, endAngle are radians not degrees.
# function svgArcToCenterParam(x1, y1, rx, ry, degree, fA, fS, x2, y2) {
#     var cx, cy, startAngle, deltaAngle, endAngle;
#     var PIx2 = Math.PI * 2.0;
#     var phi = degree * Math.PI / 180;

#     if (rx < 0) {
#         rx = -rx;
#     }
#     if (ry < 0) {
#         ry = -ry;
#     }
#     if (rx == 0.0 || ry == 0.0) { // invalid arguments
#         throw Error('rx and ry can not be 0');
#     }

#     // SVG use degrees, if your input is degree from svg,
#     // you should convert degree to radian as following line.
#     // phi = phi * Math.PI / 180;
#     var s_phi = Math.sin(phi);
#     var c_phi = Math.cos(phi);
#     var hd_x = (x1 - x2) / 2.0; // half diff of x
#     var hd_y = (y1 - y2) / 2.0; // half diff of y
#     var hs_x = (x1 + x2) / 2.0; // half sum of x
#     var hs_y = (y1 + y2) / 2.0; // half sum of y

#     // F6.5.1
#     var x1_ = c_phi * hd_x + s_phi * hd_y;
#     var y1_ = c_phi * hd_y - s_phi * hd_x;

#     // F.6.6 Correction of out-of-range radii
#     //   Step 3: Ensure radii are large enough
#     var lambda = (x1_ * x1_) / (rx * rx) + (y1_ * y1_) / (ry * ry);
#     if (lambda > 1) {
#         rx = rx * Math.sqrt(lambda);
#         ry = ry * Math.sqrt(lambda);
#     }

#     var rxry = rx * ry;
#     var rxy1_ = rx * y1_;
#     var ryx1_ = ry * x1_;
#     var sum_of_sq = rxy1_ * rxy1_ + ryx1_ * ryx1_; // sum of square
#     if (!sum_of_sq) {
#         throw Error('start point can not be same as end point');
#     }
#     var coe = Math.sqrt(Math.abs((rxry * rxry - sum_of_sq) / sum_of_sq));
#     if (fA == fS) { coe = -coe; }

#     // F6.5.2
#     var cx_ = coe * rxy1_ / ry;
#     var cy_ = -coe * ryx1_ / rx;

#     // F6.5.3
#     cx = c_phi * cx_ - s_phi * cy_ + hs_x;
#     cy = s_phi * cx_ + c_phi * cy_ + hs_y;

#     var xcr1 = (x1_ - cx_) / rx;
#     var xcr2 = (x1_ + cx_) / rx;
#     var ycr1 = (y1_ - cy_) / ry;
#     var ycr2 = (y1_ + cy_) / ry;

#     // F6.5.5
#     startAngle = radian(1.0, 0.0, xcr1, ycr1);

#     // F6.5.6
#     deltaAngle = radian(xcr1, ycr1, -xcr2, -ycr2);
#     while (deltaAngle > PIx2) { deltaAngle -= PIx2; }
#     while (deltaAngle < 0.0) { deltaAngle += PIx2; }
#     if (fS == false || fS == 0) { deltaAngle -= PIx2; }
#     endAngle = startAngle + deltaAngle;
#     while (endAngle > PIx2) { endAngle -= PIx2; }
#     while (endAngle < 0.0) { endAngle += PIx2; }

#     var outputObj = { /* cx, cy, startAngle, deltaAngle */
#         cx: cx,
#         cy: cy,
#         rx: rx,
#         ry: ry,
#         phi: phi,
#         startAngle: startAngle,
#         deltaAngle: deltaAngle,
#         endAngle: endAngle,
#         clockwise: (fS == true || fS == 1)
#     }

#     return outputObj;
# }

# Usage example:

# svg

# <path d="M 0 100 A 60 60 0 0 0 100 0"/>
# js

# var result = svgArcToCenterParam(0, 100, 60, 60, 0, 0, 0, 100, 0);
# console.log(result);
# /* will output:
# {
#     cx: 49.99999938964844,
#     cy: 49.99999938964844,
#     rx: 60,
#     ry: 60,
#     startAngle: 2.356194477985314,
#     deltaAngle: -3.141592627780225,
#     endAngle: 5.497787157384675,
#     clockwise: false
# }
# */


# see https://codepen.io/vschroeter/pen/GRbQQBv
# this could be due to parametric vs central angle


# /**
#  * Computes the angle in radians between two 2D vectors
#  * @param vx The x component of the first vector
#  * @param vy The y component of the first vector
#  * @param ux The x component of the second vector
#  * @param uy The y component of the second vector
#  * @returns The angle in radians between the two vectors
#  */
# function vectorAngle(ux, uy, vx, vy) {
#     const dotProduct = ux * vx + uy * vy;
#     const magnitudeU = Math.sqrt(ux * ux + uy * uy);
#     const magnitudeV = Math.sqrt(vx * vx + vy * vy);

#     let cosTheta = dotProduct / (magnitudeU * magnitudeV);

#     // Fix rounding errors leading to NaN
#     if (cosTheta > 1) {
#         cosTheta = 1;
#     } else if (cosTheta < -1) {
#         cosTheta = -1;
#     }

#     const angle = Math.acos(cosTheta);

#     // Determine the sign based on cross product (ux * vy - uy * vx)
#     const sign = (ux * vy - uy * vx) >= 0 ? 1 : -1;
#     return sign * angle;
# }

# /**
#  * Calculate the center of the ellipse for the arc.
#  *
#  * Based on the official W3C formula: https://www.w3.org/TR/SVG11/implnote.html#ArcConversionEndpointToCenter
#  *
#  * @param x1 The x coordinate of the start point
#  * @param y1 The y coordinate of the start point
#  * @param x2 The x coordinate of the end point
#  * @param y2 The y coordinate of the end point
#  * @param rx The x radius of the ellipse
#  * @param ry The y radius of the ellipse
#  * @param rotation The rotation of the ellipse
#  * @param largeArc The large arc flag
#  * @param sweep The sweep flag
#  */
# function getCenterParameters(x1, y1, x2, y2, rx, ry, rotation, largeArc, sweep) {
#     const fS = sweep === 1;
#     const fA = largeArc === 1;
#     const phi = rotation * (Math.PI / 180);

#     // Step 0: Ensure valid parameters
#     // F.6.6

#     // Step 0.1: Ensure radii are non-zero
#     if (rx === 0 || ry === 0) {
#         // Treat as a straight line and stop further processing
#         return {
#             center: {x: (x1 + x2) / 2, y: (y1 + y2) / 2},
#         };
#     }

#     // Step 1: Compute (x1′, y1′)
#     // F.6.5.1

#     const cosPhi = Math.cos(phi);
#     const sinPhi = Math.sin(phi);

#     const dx = (x1 - x2) / 2;
#     const dy = (y1 - y2) / 2;

#     const x1Prime = cosPhi * dx + sinPhi * dy;
#     const y1Prime = -sinPhi * dx + cosPhi * dy;

#     // Step 2: Compute (cx′, cy′)
#     // F.6.5.2

#     // Compute the square of radii
#     const rx2 = rx * rx;
#     const ry2 = ry * ry;

#     // Compute the square of the transformed points
#     const x1Prime2 = x1Prime * x1Prime;
#     const y1Prime2 = y1Prime * y1Prime;


#     // Step 0.3: Ensure radii are large enough
#     const Lambda = (x1Prime2 / (rx * rx)) + (y1Prime2 / (ry * ry));

#     if (Lambda > 1) {
#         const scale = Math.sqrt(Lambda);
#         rx *= scale;
#         ry *= scale;
#     }


#     // Compute the denominator
#     const denom = rx2 * y1Prime2 + ry2 * x1Prime2;

#     // Compute the numerator
#     const num = rx2 * ry2 - rx2 * y1Prime2 - ry2 * x1Prime2;

#     // Handle the case where the numerator becomes negative, which would result in an imaginary number
#     // This can happen if the radii are too small. So we clamp the number to 0 if it's negative.
#     const adjustedNum = Math.max(num, 0);

#     // Choose the sign for the square root based on fA and fS
#     const sign = fA !== fS ? 1 : -1;

#     // Compute (cx', cy')
#     const sqrtTerm = sign * Math.sqrt(adjustedNum / denom);

#     const cxPrime = sqrtTerm * (rx * y1Prime / ry);
#     const cyPrime = sqrtTerm * (-ry * x1Prime / rx);


#     // Step 3: Compute (cx, cy) from (cx', cy')
#     // F.6.5.3

#     // Compute the midpoints of the endpoints
#     const midX = (x1 + x2) / 2;
#     const midY = (y1 + y2) / 2;

#     // Calculate (cx, cy)
#     const cx = cosPhi * cxPrime - sinPhi * cyPrime + midX;
#     const cy = sinPhi * cxPrime + cosPhi * cyPrime + midY;

#     // Step 4: Compute theta1 and deltaTheta

#     // F.6.5.5
#     // Compute theta1
#     // CAUTION: This does not result in the correct angle if rx and ry are different
#     // const theta1 = vectorAngle(1, 0, (x1Prime - cxPrime) / rx, (y1Prime - cyPrime) / ry);

#     const theta1 = vectorAngle(1, 0, (x1Prime - cxPrime), (y1Prime - cyPrime));

#     // For global points rotate the vector (1, 0) by phi
#     const rotatedXVector = [Math.cos(-phi), Math.sin(-phi)];
#     const globalTheta1 = vectorAngle(rotatedXVector[0], rotatedXVector[1], (x1Prime - cxPrime), (y1Prime - cyPrime));

#     // F.6.5.6
#     // Compute deltaTheta

#     // CAUTION: This does not result in the correct angle if rx and ry are different
#     // const deltaTheta = vectorAngle(
#     //     (x1Prime - cxPrime) / rx, (y1Prime - cyPrime) / ry,
#     //     (-x1Prime - cxPrime) / rx, (-y1Prime - cyPrime) / ry
#     // );

#     const deltaTheta = vectorAngle(
#         (x1Prime - cxPrime), (y1Prime - cyPrime),
#         (-x1Prime - cxPrime), (-y1Prime - cyPrime)
#     );

#     // Modulo deltaTheta by 360 degrees
#     let deltaThetaDegrees = deltaTheta * (180 / Math.PI) % 360;
#     let globalTheta1Degrees = globalTheta1 * (180 / Math.PI) % 360;

#     // Adjust deltaTheta based on the sweep flag fS
#     if (!fS && deltaThetaDegrees > 0) {
#         deltaThetaDegrees -= 360;
#     } else if (fS && deltaThetaDegrees < 0) {
#         deltaThetaDegrees += 360;
#     }

#     if (!fS && globalTheta1Degrees > 0) {
#         globalTheta1Degrees -= 360;
#     } else if (fS && globalTheta1Degrees < 0) {
#         globalTheta1Degrees += 360;
#     }

#     // Convert theta1 to degrees
#     const theta1Degrees = theta1 * (180 / Math.PI);


#     return {
#         // Center of the ellipse
#         center: { x: cx, y: cy },
#         rx: rx,
#         ry: ry,
#         // Local angles (relative to the ellipse's x-axis)
#         startAngleDeg: theta1Degrees,
#         deltaAngleDeg: deltaThetaDegrees,
#         endAngleDeg: theta1Degrees + deltaThetaDegrees,
#         startAngle: theta1,
#         deltaAngle: deltaTheta,
#         endAngle: theta1 + deltaTheta,
#         // Global angles (relative to the global x-axis with vector (1, 0))
#         startAngleGlobalDeg: globalTheta1Degrees,
#         deltaAngleGlobalDeg: deltaThetaDegrees,
#         endAngleGlobalDeg: globalTheta1Degrees + deltaThetaDegrees,
#         startAngleGlobal: globalTheta1,
#         deltaAngleGlobal: deltaTheta,
#         endAngleGlobal: globalTheta1 + deltaTheta
#     };

# }