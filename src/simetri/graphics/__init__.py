"""simetri.graphics is a module that provides a simple and intuitive way to create geometric shapes and patterns."""

# status: prototype
# This is a proof of concept.
# Testing is incomplete.
# Everything is subject to change till we release a beta version.

__version__ = "0.0.9"
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
from .core import *
from ..frieze import frieze
from ..settings.settings import *
from ..graphics.common import *


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
from ..geometry.bezier import *
from ..geometry.circle import *
from ..geometry.collection import *
from ..geometry.convex_hull import convex_hull
from ..geometry.ellipse import *
from ..geometry.geom_utils import *
from ..geometry.geometry import *
from ..geometry.hobby import *
from ..geometry.lattice import *
from ..geometry.polygon import *
from ..geometry.sine import *
from ..geometry.vectors import *
from ..graphics.all_enums import *
from ..graphics.shape import (
    Clipping,
    all_segments,
    clip,
    polygon_diff,
    polygon_difference,
    polygon_intersection,
    polygon_xor,
)
from ..graphics.shapes import *

# Preserve geometric Line class on public namespace.
from ..graphics.shapes import Line as Line
from ..graphics.sketch import *
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
from .affine import *
from .batch import *
from .dots import *
from .mask import Gradient, Mask, Stop
from .path import Operation, Path2D
from .pattern import *


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
