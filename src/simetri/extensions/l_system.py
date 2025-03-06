"""Lindenmayer system (L-system) module."""

from math import ceil

from ..graphics.batch import Batch
from ..graphics.shape import Shape
from .turtle_sg import Turtle


def l_system(
    rules: dict, axiom: str, angle: float, dist: float, n: int, d_actions: dict = None
):
    """Generate a Lindenmayer system (L-system) using the given rules.

    An L-system is a parallel rewriting system that uses recursive rules to
    generate complex patterns. This function interprets the generated string
    as turtle graphics commands.

    Args:
        rules: A dictionary with characters as keys and strings as values.
               Each character in the axiom or resulting string will be replaced
               by its corresponding rule in each iteration.
        axiom: The initial string to start the L-system.
        angle: The angle (in degrees) for turtle rotation commands.
        dist: The distance for turtle forward/backward movement.
        n: The number of iterations to apply the rules.
        d_actions: Optional dictionary mapping characters to turtle methods.
                  This allows extending the default command set.

    Returns:
        Batch: A batch of shapes representing the L-system drawing.

    Example:
        >>> rules = {'F': 'F+F-F-F+F'}  # Koch curve
        >>> batch = l_system(rules, 'F', 60, 10, 3)
    """

    turtle = Turtle(in_degrees=True)
    turtle.def_angle = angle
    turtle.def_dist = dist

    actions = {
        "F": turtle.forward,
        "B": turtle.backward,
        "G": turtle.go,
        "+": turtle.left,
        "-": turtle.right,
        "[": turtle.push,
        "]": turtle.pop,
        "|": turtle.turn_around,
    }
    if d_actions:
        for key, value in d_actions.items():
            method = getattr(turtle, value)
            actions[key] = method

    def expand(axiom, rules):
        """Expand the axiom using the provided rules.

        Args:
            axiom: The string to expand.
            rules: Dictionary of replacement rules.

        Returns:
            str: The expanded string after applying the rules.
        """
        return "".join([rules.get(char, char) for char in axiom])

    for _ in range(n):
        axiom = expand(axiom, rules)

    for char in axiom:
        actions.get(char, lambda: None)()

    # TikZ gives memory error if there are too many vertices in one shape
    shapes = Batch()
    # tot = len(turtle.current_list)
    # part = 200 # partition size
    # for i in range(ceil(tot/part)):
    #     shape = Shape(turtle.current_list[i*part:(i+1)*part])
    #     shapes.append(shape)
    if turtle.current_list:
        turtle.lists.append(turtle.current_list)
    for x in turtle.lists:
        shapes.append(Shape(x))
    return shapes


# rules = {}
# rules['F'] = '-F+F+G[+F+F]-'
# rules['G'] = 'GG'

# axiom = 'F'
# angle = 60
# dist = 15
# n=4

# l_system(rules, axiom, angle, dist, n)

# Examples

# rules = {}
# rules['X'] = 'XF+F+XF-F-F-XF-F+F+F-F+F+F-X'
# axiom = 'XF+F+XF+F+XF+F'
# angle = 60
# n=2


# rules = {}
# rules['X'] = 'F-[[X]+X]+F[+FX]-X'
# rules['F'] = 'FF'
# axiom = 'X'
# angle = 25
# n=6

# rules = {}
# rules['A'] = '+F-A-F+' # Sierpinsky
# rules['F'] = '-A+F+A-'
# axiom = 'A'
# angle = 60
# n = 7

# rules = {}
# rules['F'] = 'F+F-F-F+F' # Koch curve 1
# axiom = 'F'
# angle = 60
# n = 6

# rules = {}
# rules['X'] = 'X+YF+'  # Dragon curve
# rules['Y'] = '-FX-Y'
# axiom = 'FX'
# angle = 90
# n=10


# rules = {}
# rules['X'] = 'F-[[X]+X]+F[+FX]-X'  # Wheat
# rules['F'] = 'FF'
# axiom = 'X'
# angle = 25
# n=6

# rules = {}
# axiom = 'F+F+F+F'
# rules['F'] = 'FF+F-F+F+FF'
# angle = 90
# n=4
