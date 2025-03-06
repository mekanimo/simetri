"""Geometric constraint solver for points, segments, circles.
Uses Sequential Least Squares Programming (SLSQP) to solve the given constraints.
"""

from dataclasses import dataclass

from scipy.optimize import minimize

from simetri.graphics.all_enums import ConstraintType as ConstType
from simetri.helpers.vector import Vector2D
from simetri.geometry.geometry import (
    direction,
    distance,
    is_line,
    angle_between_two_lines,
    point_to_line_distance
)
from simetri.geometry.circle import Circle_ as Circle


@dataclass
class Constraint:
    """Constraint class for geometric constraints."""

    item1: "point segment or circle"
    item2: "point segment or circle"
    type: ConstType
    value: float = None
    value2: float = None  # used for equal_value_eq

    def __post_init__(self):
        """Set item sizes for circles and segments."""
        self.equation = d_equations[self.type]
        if self.type == ConstType.EQUAL_SIZE:
            if isinstance(self.item1, Circle):
                self.size1 = self.item1.radius
            elif is_line(self.item1):
                self.size1 = distance(*self.item1)

            if isinstance(self.item2, Circle):
                self.size2 = self.item2.radius
            elif is_line(self.item2):
                self.size2 = distance(*self.item2)

    def check(self):
        """Check the constraint value.

        Returns:
            float: The result of the constraint equation.
        """
        return self.equation(self)


# Constraint equations
# These equations return zero if the constraint is satisfied.
# To check if a segment is horizontal use parallel_eq with the x-axis
# To check if a segment is vertical use paralell_eq with the y-axis
# For concentric circles use distance_eq (center to center dist = 0)
# For point on a circle use distance_eq (point to center dist = radius)


def distance_eq(constraint):
    """Return the difference between the target and current distance.

    Args:
        constraint (Constraint): The constraint object.

    Returns:
        float: The difference between the target and current distance.
    """
    if isinstance(constraint.item1, Circle):
        p1 = constraint.item1.center
    else:
        p1 = constraint.item1

    if isinstance(constraint.item2, Circle):
        p2 = constraint.item2.center
    else:
        p2 = constraint.item2

    value = constraint.value

    return distance(p1, p2) - value


def parallel_eq(constraint):
    """Return the cross product. If the segments are parallel, the cross product is 0.

    Args:
        constraint (Constraint): The constraint object.

    Returns:
        float: The cross product of the vectors.
    """
    vec1 = Vector2D(*constraint.item1)
    vec2 = Vector2D(*constraint.item2)

    return vec1.cross(vec2)


def perpendicular_eq(constraint):
    """Return the dot product. If the segments are perpendicular, the dot product is 0.

    Args:
        constraint (Constraint): The constraint object.

    Returns:
        float: The dot product of the vectors.
    """
    seg1 = constraint.item1
    seg2 = constraint.item2

    vec1 = Vector2D(*seg1)
    vec2 = Vector2D(*seg2)

    return vec1.dot(vec2)


def equal_size_eq(constraint):
    """Return the difference between the sizes of the items.

    For segments, item size is the length of the segment.
    For circles, item size is the radius of the circle.

    Args:
        constraint (Constraint): The constraint object.

    Returns:
        float: The difference between the sizes of the items.
    """

    return constraint.item1.size1 - constraint.item2.size2


def outer_tangent_eq(constraint):
    """Return the difference between the distance of the circles and the sum of the radii.

    If the circles are tangent, the difference is 0.

    Args:
        constraint (Constraint): The constraint object.

    Returns:
        float: The difference between the distance of the circles and the sum of the radii.
    """
    if is_line(constraint.item1):
        circle = constraint.item2
        res = circle.radius -  point_to_line_distance(circle.center, constraint.item1)
    elif is_line(constraint.item2):
        circle = constraint.item1
        res = circle.radius - point_to_line_distance(circle.center, constraint.item2)
    else:
        circle_1 = constraint.item1
        circle_2 = constraint.item2

        dist = distance(circle_1.center, circle_2.center)
        rad1 = circle_1.radius
        rad2 = circle_2.radius

        res = dist - (rad1 + rad2)

    return res

def inner_tangent_eq(constraint):
    """Return the difference between the distance of the circles and the sum of the radii.

    If the circles are tangent, the difference is 0.

    Args:
        constraint (Constraint): The constraint object.

    Returns:
        float: The difference between the distance of the circles and the sum of the radii.
    """
    circle_1 = constraint.item1
    circle_2 = constraint.item2

    dist = distance(circle_1.center, circle_2.center)
    rad1 = circle_1.radius
    rad2 = circle_2.radius

    return dist - abs(rad1 - rad2)


def collinear_eq(constraint):
    """Return the difference in direction for collinear items.

    Items can be: segments, segment and a circle, or a segment and a point.

    Args:
        constraint (Constraint): The constraint object.

    Returns:
        float: The difference in direction.
    """
    # for now only segments are implemented
    a1, b1 = constraint.item1
    a2, b2 = constraint.item2

    return direction(a1, b1, a2) - direction(a1, b1, b2)


def equal_value_eq(constraint):
    """Return the difference between the values.

    Args:
        constraint (Constraint): The constraint object.

    Returns:
        float: The difference between the values.
    """
    return constraint.value - constraint.value2


def line_angle_eq(constraint):
    """Return the angle between two segments.

    Args:
        constraint (Constraint): The constraint object.

    Returns:
        float: The angle between the two segments.
    """
    seg1 = constraint.item1
    seg2 = constraint.item2
    return constraint.value - angle_between_two_lines(seg1, seg2)


d_equations = {
    ConstType.COLLINEAR: collinear_eq,
    ConstType.DISTANCE: distance_eq,
    ConstType.EQUAL_SIZE: equal_size_eq,
    ConstType.EQUAL_VALUE: equal_value_eq,
    ConstType.LINE_ANGLE: line_angle_eq,
    ConstType.PARALLEL: parallel_eq,
    ConstType.PERPENDICULAR: perpendicular_eq,
    ConstType.INNER_TANGENT: inner_tangent_eq,
    ConstType.OUTER_TANGENT: outer_tangent_eq,
}


def solve(constraints, update_func, initial_guess, bounds=None, tol=1e-04):
    """Solve the geometric constraints.

    Args:
        constraints (list): List of Constraint objects.
        update_func (function): Function that updates the constraint items.
        initial_guess (list): Initial guess for the solution.
        bounds (list, optional): Bounds for the solution. Defaults to None.
        tol (float, optional): Tolerance for the solution. Defaults to 1e-04.

    Returns:
        OptimizeResult: The optimization result represented as a `OptimizeResult` object.
    """

    def objective(x):
        """Objective function for the minimization.

        Args:
            x (list): Current values of the variables.

        Returns:
            float: The sum of the constraint checks.
        """
        update_func(x)

        return sum((constr.check() for constr in constraints))

    def check_constraints(x):
        """Return constraint results.

        Args:
            x (list): Current values of the variables.

        Returns:
            list: List of constraint check results.
        """
        update_func(x)

        return [constr.check() for constr in constraints]

    res = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": check_constraints},
        options={"eps": tol},
    )

    return res


# # Example:
# # Given 2 circles and a radius, find the position of a circle with the given radius
# # that is tangent to both circles.
# x1 = y1 = 0
# r1 = 40
# c1 = Circle((x1, y1), r1) # this would be sg.Circle((x1, y1), r1)
# x2 = 100
# y2 = 0
# r2 = 35
# c2 = Circle((x2, y2), r2)

# # c3 position is estimated at this point
# # c3 radius is fixed
# guess = (45, 10)
# x3 = 45
# y3 = 45
# r3 = 50
# c3 = Circle((x3, y3), r3)

# const1 = Constraint(c1, c3, ConstType.OUTER_TANGENT)
# const2 = Constraint(c2, c3, ConstType.OUTER_TANGENT)

# def update(x):
#     '''Update the position of the circle.'''
#     c3.center = (x[0], x[1])

# bounds = [(35, 100), (35, 100)]
# print(solve([const1, const2], update, guess, bounds ))

# print(c3.center)

# Apollonius problem

# start with 2 identical circles tangent to each other

# c1 = Circle((0, 0), 50)
# c2 = Circle((100, 0), 50)

# c3 is the big circle

# c3 = Circle((50, -200), 200)

# const1 = Constraint(c1, c3, ConstType.INNER_TANGENT)
# const2 = Constraint(c2, c3, ConstType.INNER_TANGENT)
# # const3 = Constraint(c2, c3, ConstType.DISTANCE, 200)

# def update(x):
#     '''Update the position of the circle.'''
#     c3.center = (x[0], x[1])

# guess = (50, -200)
# bounds = [(49, 51), (-350, -50)]
# print(solve([const1, const2], update, guess, bounds))
# print('c3 center:', c3.center)

# # Now we have 3 circles tangent to each other
# # We will add a circle tangent to the 3 circles
# # c4 is the circle we are looking for

# c4 = Circle((50, 55), 5)

# const1 = Constraint(c1, c4, ConstType.OUTER_TANGENT)
# const2 = Constraint(c2, c4, ConstType.OUTER_TANGENT)
# const3 = Constraint(c3, c4, ConstType.INNER_TANGENT)

# def update(x):
#     '''Update the position of the circle.'''
#     c4.center = (x[0], x[1])
#     c4.radius = x[2]

# guess = (50, 55, 5) # x4, y4, r4
# bounds = [(49.99, 50.01), (5, 100), (3, 30)]

# print(solve([const1, const2, const3], update, guess, bounds))
# print(c4.center, c4.radius)
