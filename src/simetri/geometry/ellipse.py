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
        """
        Args:
            center (Point): The center of the arc.
            start_angle (float): The starting angle of the arc.
            span_angle (float): The span angle of the arc.
            radius (float): The radius of the arc.
            radius2 (float): The second radius for elliptical arcs.
            n_points (int, optional): Number of points to generate. Defaults to None.
            rot_angle (float, optional): Rotation angle. Defaults to 0.
            xform_matrix (ndarray, optional): Transformation matrix. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
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
    '''Return the vertices of an elliptic arc.

    Args:
        start (tuple): Starting point of the arc.
        start_angle (float): Starting angle of the arc.
        end_angle (float): Ending angle of the arc.
        radius1 (float): First radius of the arc.
        radius2 (float): Second radius of the arc.

    Returns:
        list: Vertices of the elliptic arc.
    '''
    center = (0, 0)



class Ellipse(Shape):
    """An ellipse defined by center, width, and height."""

    def __init__(self, center: Point, width: float, height: float, angle:float=0,
                                                xform_matrix:ndarray = None, **kwargs) -> None:
        """
        Args:
            center (Point): The center of the ellipse.
            width (float): The width of the ellipse.
            height (float): The height of the ellipse.
            angle (float, optional): Rotation angle. Defaults to 0.
            xform_matrix (ndarray, optional): Transformation matrix. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
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

    @property
    def closed(self):
        """Return True ellipse is always closed.

        Returns:
            bool: Always returns True.
        """
        return True

    @closed.setter
    def closed(self, value: bool):
        pass

    def copy(self):
        """Return a copy of the ellipse.

        Returns:
            Ellipse: A copy of the ellipse.
        """
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
    """Calculates the angle of the tangent line to an ellipse at the point (x, y).

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.
        tol (float, optional): Tolerance for point on ellipse check. Defaults to .001.

    Returns:
        float: Angle of the tangent line in radians.
    """
    if abs((x**2 / a**2) + (y**2 / b**2) - 1) >= tol:
        res = False
    else:
        # res = atan2(-(b**2 * x), (a**2 * y))
        res = atan2((b**2 * x), -(a**2 * y))

    return res


def r_central(a, b, theta):
    '''Return the radius (distance between the center and the intersection point)
    of the ellipse at the given angle.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        theta (float): Angle in radians.

    Returns:
        float: Radius at the given angle.
    '''
    return (a * b) / sqrt((b * cos(theta))**2 + (a * sin(theta))**2)


def ellipse_line_intersection(a, b, point):
    '''Return the intersection points of an ellipse and a line segment
    connecting the given point to the ellipse center at (0, 0).

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        point (tuple): Point coordinates (x, y).

    Returns:
        list: Intersection points.
    '''
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
        span_angle (float): Span angle of the arc.
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

    Args:
        t_0 (float): Starting parametric angle.
        t_1 (float): Ending parametric angle.
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.

    Returns:
        float: Arclength of the ellipse.
    '''
    m = 1 - (b / a)**2
    t1 = ellipeinc(t_1 - 0.5 * pi, m)
    t0 = ellipeinc(t_0 - 0.5 * pi, m)
    return a*(t1 - t0)


def central_to_parametric_angle(a, b, phi):
    """
    Converts a central angle to a parametric angle on an ellipse.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        phi (float): Angle of the line intersecting the center and the point.

    Returns:
        float: Parametric angle (in radians).
    """
    t = atan2((a/b) * sin(phi), cos(phi))
    if t < 0:
        t += 2 * pi

    return t


def parametric_to_central_angle(a, b, t):
    """
    Converts a parametric angle on an ellipse to a central angle.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        t (float): Parametric angle (in radians).

    Returns:
        float: Angle of the line intersecting the center and the point.
    """
    phi = atan2((b/a) * sin(t), cos(t))
    if phi < 0:
        phi += 2 * pi

    return phi


def ellipse_point(a, b, angle):
    '''Return a point on an ellipse with the given a=width/2, b=height/2, and angle.
    angle is the central-angle and in radians.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        angle (float): Central angle in radians.

    Returns:
        tuple: Coordinates of the point on the ellipse.
    '''
    r = r_central(a, b, angle)

    return (r * cos(angle), r * sin(angle))


def ellipse_param_point(a, b, t):
    '''Return a point on an ellipse with the given a=width/2, b=height/2, and parametric angle.
    t is the parametric angle and in radians.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        t (float): Parametric angle in radians.

    Returns:
        tuple: Coordinates of the point on the ellipse.
    '''
    return (a * cos(t), b * sin(t))


def get_ellipse_t_for_angle(angle, a, b):
    """
    Calculates the parameter t for a given angle on an ellipse.

    Args:
        angle (float): The angle in radians.
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.

    Returns:
        float: The parameter t.
    """
    t = atan2(a * sin(angle), b * cos(angle))
    if t < 0:
        t += 2 * pi
    return t


def ellipse_central_angle(t, a, b):
    """
    Calculates the central angle of an ellipse for a given parameter t.

    Args:
        t (float): The parameter value.
        a (float): The semi-major axis of the ellipse.
        b (float): The semi-minor axis of the ellipse.

    Returns:
        float: The central angle in radians.
    """
    theta = atan2(a * sin(t), b * cos(t))

    return theta