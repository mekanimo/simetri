"""Lindenmayer system (L-system) module."""

from math import ceil

from ..graphics.batch import Batch
from ..graphics.shape import Shape
from .turtle_sg import Turtle


def l_system(
    rules: dict, axiom: str, angle: float, dist: float, n: int, d_actions: dict = None
):
    """Generate a Lindenmayer system (L-system) using the given rules."""

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
