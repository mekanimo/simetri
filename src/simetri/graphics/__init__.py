"""Public entry point for Simetri graphics.

Re-exports shapes, groups, transforms, colors, canvas helpers, and related
utilities so callers can use ``import simetri.graphics as sg``.

Examples:
    >>> import simetri.graphics as sg
    >>> square = sg.Shape([(0, 0), (50, 0), (50, 50), (0, 50)], closed=True)
    >>> square.translate(10, 20)
"""

# status: prototype
# This is a proof of concept.
# Testing is incomplete.
# Everything is subject to change till we release a beta version.

from .. import __version__

__author__ = "Fahri Basegmez"

from math import (
    cos,
    sin,
    pi,
    atan,
    atan2,
    sqrt,
    degrees,
    radians,
    exp,
    log,
    log10,
    e,
    tau,
    ceil,
    floor,
    trunc,
    hypot,
    gcd,
    factorial,
    comb,
    perm,
    prod,
)
from itertools import cycle, combinations, permutations, product
from random import choice, choices, randint, random, uniform, shuffle
from functools import lru_cache as memoize
from numpy import linspace, arange, array, zeros, ones, full, eye, diag

from ..helpers.utilities import *
from ..core.core import *
from ..frieze import frieze
from ..settings.settings import *
from ..core.common import *


set_defaults()
from simetri import colors

from ..canvas.canvas import *
from ..canvas.grids import *
from ..canvas.style_map import *
from ..colors.colors import *
from ..colors.palettes import *
from ..colors.pastels import *
from ..colors.swatches import *
from ..extensions.easing import *
from ..extensions.l_system import l_system
from ..extensions.tree import TreeNode, make_tree
from ..extensions.turtle_sg import Turtle, spirolateral
from ..frieze.frieze_patterns import *
from ..geometry.nonlinear.bezier import *
from ..geometry.nonlinear.circle import *
from ..geometry.collection import *
from ..geometry.polygons.convex_hull import convex_hull
from ..geometry.nonlinear.ellipse import *
from ..geometry.geom_utils import *
from ..geometry.geometry import *
from ..geometry.nonlinear.hobby import *
from ..patterns.lattice import *
from ..geometry.polygons.polygon import *
from ..geometry.nonlinear.sine import *
from ..geometry.vectors import *
from ..core.all_enums import *
from ..shapes.shape import (
    Clipping,
    all_segments,
    clip,
    polygon_diff,
    polygon_difference,
    polygon_intersection,
    polygon_xor,
)
from ..shapes.shapes import *

# Preserve geometric Line class on public namespace.
from ..shapes.shapes import Line as Line
from ..canvas.sketch import *
from ..helpers.box_solver import push_boxes_apart, Box
from ..helpers.constraint_solver import Constraint, solve
from ..helpers.illustration import *
from ..helpers.modifiers import *
from ..helpers.validation import check_version
from ..image.image import Image, open_img
from ..lace import Lace
from ..stars import stars
from ..stars.stars import Star, rosette
from ..svg.filters import *
from ..svg.svg import *
from ..tikz.tikz import *
from ..wallpaper import wallpaper
from ..geometry.affine import *
from ..group.batch import *
from ..shapes.dots import *
from ..canvas.mask import Gradient, Mask, Stop
from ..geometry.nonlinear.path import Operation, Path2D
from ..patterns.pattern import *


set_tikz_defaults()
set_svg_defaults()

# aliases
is_close = isclose
Batch = Group
TM = translation_matrix
RM = rotation_matrix
MM = mirror_matrix
GM = glide_matrix
SM = scale_matrix
SHM = shear_matrix
LinPath = Path2D
