import numpy as np
import cmath

from ..graphics.shape import Shape
from ..geometry.bezier import bezier_points

"""
Implementation of John Hobby's Bezier curve algorithm in Python.
The algorithm is very simple and efficient. Details are on page 112, 113 in Knuth's METAFONT: The Program.
"""
# Taken from https://github.com/ltrujello/Hobby_Curve_Algorithm 2/7/2025


class HobbyPoint(complex):
    """A class for associating numerical quantities from Hobby's algorithm with points that appear on a Hobby curve.

    We subclass `complex` to perform complex arithmetic with points on the Hobby curve, as required in the algorithm.

    Attributes:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        alpha (float): The reciprocal of tension for incoming segment.
        beta (float): The reciprocal of tension for outgoing segment.
        d_val (float): Distance between this point and next.
        theta (float): Angle of polygonal line from this point to next.
        phi (float): Offset angle.
        psi (float): Another offset angle.
    """

    def __new__(cls, x: float, y: float, tension: float) -> "HobbyPoint":
        """Create a new instance of HobbyPoint.

        Args:
            x: The x-coordinate of the point.
            y: The y-coordinate of the point.
            tension: The tension value for the curve.

        Returns:
            A new HobbyPoint instance with complex value (x + y*j).
        """
        return super().__new__(cls, x, y)

    def __init__(self, x: float, y: float, tension: float) -> None:
        """Initialize a HobbyPoint with coordinates and tension.

        Args:
            x: The x-coordinate of the point.
            y: The y-coordinate of the point.
            tension: The tension value for the curve.
        """
        self.x = x
        self.y = y
        # In what follows, we use Knuth's notation in our variable names.
        self.alpha = 1 / tension
        self.beta = 1 / tension
        self.d_val = 0  # Distance between this point and next.
        self.theta = 0  # Angle of polygonal line from this point to next.
        self.phi = 0  # Offset angle.
        self.psi = 0  # Another offset angle.

    def debug_info(self) -> str:
        """Return a string with the point's information.

        Returns:
            A string containing the point's coordinates and all of its computational values.
        """
        return (
            f"{(self.x, self.y)} "
            f"alpha={self.alpha}, "
            f"beta={self.beta}, "
            f"theta={self.theta}, "
            f"psi={self.psi}, "
            f"phi={self.phi}, "
            f"d_val={self.d_val}"
        )

    def __repr__(self) -> str:
        """Return a string representation of the point.

        Returns:
            A string representation of the point's coordinates.
        """
        return f"{(self.x, self.y)}"


class HobbyCurve:
    """A class for calculating the control points required to draw a Hobby curve.

    Attributes:
        points (list[HobbyPoint]): The list of points defining the curve.
        ctrl_pts (list[tuple]): The calculated control points.
        is_cyclic (bool): Whether the curve is closed.
        begin_curl (float): Curl value for the beginning of the curve.
        end_curl (float): Curl value for the end of the curve.
        n_points (int): Number of points in the curve.
        debug_mode (bool): Whether to print debug information.
    """

    def __init__(
        self,
        points: list[tuple],
        tension: float = 1,
        cyclic: bool = False,
        begin_curl: float = 1,
        end_curl: float = 1,
        debug: bool = False,
    ) -> None:
        """Initialize a HobbyCurve with the given parameters.

        Args:
            points: List of (x, y) tuples representing the curve's points.
            tension: Tension parameter controlling the "tightness" of the curve.
            cyclic: Whether the curve should be closed.
            begin_curl: Curl value for the beginning of the curve.
            end_curl: Curl value for the end of the curve.
            debug: Whether to print debug information.

        Raises:
            ValueError: If fewer than 2 points are provided.
        """
        if len(points) < 2:
            raise ValueError("Algorithm needs more than 2 points")
        self.points = [HobbyPoint(*point, tension) for point in points]
        self.ctrl_pts = []
        self.is_cyclic = cyclic
        self.begin_curl = begin_curl
        self.end_curl = end_curl
        self.n_points = len(points)
        self.debug_mode = debug

    def get_ctrl_points(self) -> list[tuple]:
        """Calculate and return all of the control points of the Hobby curve.

        Executes the Hobby algorithm by calculating distance values, psi values,
        theta values, and phi values, then uses these to compute control points.

        Returns:
            A list of (x, y) tuples representing the Bezier control points.
        """
        self.calculate_d_vals()
        self.calculate_psi_vals()
        self.calculate_theta_vals()
        self.calculate_phi_vals()
        self.show_debug_msg()
        self.ctrl_pts = self.calculate_ctrl_pts()
        return self.ctrl_pts

    def calculate_d_vals(self) -> None:
        """Calculate the pairwise distances between consecutive points in the curve."""
        # Skip last point if path is non-cyclic
        point_inds = (
            range(self.n_points) if self.is_cyclic else range(self.n_points - 1)
        )
        for i in point_inds:
            z_i = self.points[i % self.n_points]
            z_j = self.points[(i + 1) % self.n_points]
            z_i.d_val = abs(z_i - z_j)

    def calculate_psi_vals(self) -> None:
        """Calculate the psi values by finding the angle of the polygonal turns.

        Raises:
            ZeroDivisionError: If consecutive points have the same coordinates.
        """
        # Skip first and last point if path is non-cyclic
        point_inds = (
            range(self.n_points) if self.is_cyclic else range(1, self.n_points - 1)
        )
        for i in point_inds:
            z_h = self.points[i - 1]
            z_i = self.points[i]
            z_j = self.points[(i + 1) % self.n_points]
            try:
                polygonal_turn = (z_j - z_i) / (z_i - z_h)
                # print(z_j - z_i, z_i - z_h)
            except ZeroDivisionError:
                raise ZeroDivisionError(
                    f"Consecutive points {(z_h.x, z_h.y)} and {(z_i.x, z_i.y)} cause zero division."
                )
            z_i.psi = np.arctan2(polygonal_turn.imag, polygonal_turn.real)

    def calculate_theta_vals(self) -> None:
        """Calculate the theta values by solving a linear system of equations.

        This is the core of Hobby's algorithm, creating and solving a system of equations
        to find the optimal angles for smooth splines.
        """
        A = np.zeros(
            self.n_points
        )  # Inappropriate names, but they mirror Knuth's notation.
        B = np.zeros(self.n_points)
        C = np.zeros(self.n_points)
        D = np.zeros(self.n_points)
        R = np.zeros(self.n_points)

        # Calculate the entries of the five vectors.
        # Skip first and last point if path is non-cyclic.
        point_ind = (
            range(self.n_points) if self.is_cyclic else range(1, self.n_points - 1)
        )
        for i in point_ind:
            z_h = self.points[i - 1]
            z_i = self.points[i]
            z_j = self.points[(i + 1) % self.n_points]

            A[i] = z_h.alpha / (z_i.beta**2 * z_h.d_val)
            B[i] = (3 - z_h.alpha) / (z_i.beta**2 * z_h.d_val)
            C[i] = (3 - z_j.beta) / (z_i.alpha**2 * z_i.d_val)
            D[i] = z_j.beta / (z_i.alpha**2 * z_i.d_val)
            R[i] = -B[i] * z_i.psi - D[i] * z_j.psi

        # Set up matrix M such that the soln. Mx = R are the theta values.
        M = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_points):
            # Fill i-th row of M
            M[i][i - 1] = A[i]
            M[i][i] = B[i] + C[i]
            M[i][(i + 1) % self.n_points] = D[i]

        # Special formulas for first and last rows of M with non-cyclic paths.
        if not self.is_cyclic:
            # First row of M
            alpha_0 = self.points[0].alpha
            beta_1 = self.points[1].beta
            xi_0 = (alpha_0**2 * self.begin_curl) / beta_1**2
            M[0][0] = alpha_0 * xi_0 + 3 - beta_1
            M[0][1] = (3 - alpha_0) * xi_0 + beta_1
            R[0] = -((3 - alpha_0) * xi_0 + beta_1) * self.points[1].psi
            # Last row of M
            alpha_n_1 = self.points[-2].alpha
            beta_n = self.points[-1].beta
            xi_n = (beta_n**2 * self.end_curl) / alpha_n_1**2
            M[-1][-2] = (3 - beta_n) * xi_n + alpha_n_1
            M[-1][-1] = beta_n * xi_n + 3 - alpha_n_1
            R[-1] = 0

        # Solve for theta values.
        thetas = np.linalg.solve(M, R)
        for i, point in enumerate(self.points):
            point.theta = thetas[i]

    def calculate_phi_vals(self) -> None:
        """Calculate the phi values using the relationship theta + phi + psi = 0."""
        for point in self.points:
            point.phi = -(point.psi + point.theta)

    def calculate_ctrl_pts(self) -> list[tuple]:
        """Calculate the Bezier control points between consecutive points.

        Returns:
            A list of (x, y) tuples representing the control points, with two control points
            for each curve segment between consecutive points.
        """
        ctrl_pts = []
        # Skip last point if path is non-cyclic
        point_inds = (
            range(self.n_points) if self.is_cyclic else range(self.n_points - 1)
        )
        for i in point_inds:
            z_i = self.points[i]
            z_j = self.points[(i + 1) % self.n_points]
            rho_coefficient = z_i.alpha * velocity(z_i.theta, z_j.phi)
            sigma_coefficient = z_j.beta * velocity(z_j.phi, z_i.theta)
            ctrl_pt_a = z_i + (1 / 3) * rho_coefficient * cmath.exp(
                complex(0, z_i.theta)
            ) * (z_j - z_i)
            ctrl_pt_b = z_j - (1 / 3) * sigma_coefficient * cmath.exp(
                complex(0, -z_j.phi)
            ) * (z_j - z_i)
            ctrl_pts.append((ctrl_pt_a.real, ctrl_pt_a.imag))
            ctrl_pts.append((ctrl_pt_b.real, ctrl_pt_b.imag))
        return ctrl_pts

    def show_debug_msg(self) -> None:
        """Display debug information for each point if debug mode is enabled."""
        if self.debug_mode:
            for point in self.points:
                print(point.debug_info())

    def __repr__(self) -> str:
        """Return a string representation of the curve.

        Returns:
            A string representation of the curve's points in Cartesian coordinates.
        """
        cartesian_points = [(point.real, point.imag) for point in self.points]
        return repr(cartesian_points)


def hobby_ctrl_points(
    points: list[tuple],
    tension: float = 1,
    cyclic: bool = False,
    begin_curl: float = 1,
    end_curl: float = 1,
    debug: bool = False,
) -> list[tuple]:
    """Calculate cubic Bezier control points using John Hobby's algorithm.

    Args:
        points: List of (x, y) tuples representing the curve's points.
        tension: Controls the "tightness" of the curve (lower is tighter).
        cyclic: Whether the curve should be closed.
        begin_curl: Curl value for the beginning of the curve.
        end_curl: Curl value for the end of the curve.
        debug: Whether to print debug information.

    Returns:
        A list of (x, y) tuples representing the Bezier control points.
    """
    curve = HobbyCurve(
        points,
        tension=tension,
        cyclic=cyclic,
        begin_curl=begin_curl,
        end_curl=end_curl,
        debug=debug,
    )
    ctrl_points = curve.get_ctrl_points()

    # Calculate whitespace padding for pretty print.
    max_pad = 0
    for ctrl_point in ctrl_points:
        x, y = ctrl_point[:2]
        # Calculate number of digits in x, y before decimal, and take the
        # max for nice padding.
        padding = max(
            1 if abs(x) <= 0.1 else int(np.ceil(np.log10(abs(x)))) + 1,
            1 if abs(y) <= 0.1 else int(np.ceil(np.log10(abs(y)))) + 1,
        )
        if max_pad < padding:
            max_pad = padding

    if debug:
        # Pretty print control points.
        precision = 10
        space = precision + max_pad + 1  # +1 for negative sign
        i = 0
        while i < len(ctrl_points) - 1:
            x_1, y_1 = ctrl_points[i]
            x_2, y_2 = ctrl_points[i + 1]
            print(
                f"({x_1:<{space}.{precision}f}, {y_1:<{space}.{precision}f}) "
                f"and "
                f"({x_2:<{space}.{precision}f}, {y_2:<{space}.{precision}f})"
            )
            i += 2

    return ctrl_points


def velocity(theta: float, phi: float) -> float:
    """Calculate the "velocity" function used in Metafont's curve algorithm.

    This function implements the specific velocity formula from Knuth's Metafont.

    Args:
        theta: The theta angle value.
        phi: The phi angle value.

    Returns:
        The computed velocity value used in control point calculations.
    """
    numerator = 2 + np.sqrt(2) * (np.sin(theta) - (1 / 16) * np.sin(phi)) * (
        np.sin(phi) - (1 / 16) * np.sin(theta)
    ) * (np.cos(theta) - np.cos(phi))
    denominator = (
        1
        + (1 / 2) * (np.sqrt(5) - 1) * np.cos(theta)
        + (1 / 2) * (3 - np.sqrt(5)) * np.cos(phi)
    )
    return numerator / denominator

def hobby_shape(points, cyclic=False, tension=1, begin_curl=1, end_curl=1, n_points=None):
    """Create a Shape object from points using John Hobby's algorithm.

    This function calculates cubic Bezier control points using Hobby's algorithm,
    then creates a Shape object by generating points along the resulting Bezier curves.

    Args:
        points: List of (x, y) tuples representing the curve's points.
        cyclic: Whether the curve should be closed.
        tension: Controls the "tightness" of the curve (lower is tighter).
        begin_curl: Curl value for the beginning of the curve.
        end_curl: Curl value for the end of the curve.
        debug: Whether to print debug information.

    Returns:
        A Shape object containing points along the smooth Hobby curve.
    """
    controls = hobby_ctrl_points(points, tension=tension, cyclic=cyclic,
                                            begin_curl=begin_curl, end_curl=end_curl)
    n = len(points)
    res = []
    if cyclic:
        for i in range(n):
            ind = i * 2
            p0 = points[i]
            p1 = controls[ind]
            p2 = controls[ind + 1]
            p3 = points[(i + 1)%n]
            bez_pnts = bezier_points(p0, p1, p2, p3, 10)
            res.extend(bez_pnts)
    else:
        for i in range(n - 1):
            ind = i * 2
            p0 = points[i]
            p1 = controls[ind]
            p2 = controls[ind + 1]
            p3 = points[(i + 1)]
            bez_pnts = bezier_points(p0, p1, p2, p3, 20)
            res.extend(bez_pnts)
    return Shape(res)