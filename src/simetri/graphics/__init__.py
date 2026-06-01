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
from ..geometry.geometry import *
from ..geometry.ellipse import *
from ..geometry.bezier import *
from ..geometry.hobby import *
from ..geometry.circle import *
from ..geometry.sine import *
from ..geometry.vectors import *
from .affine import *
from .dots import *
from ..graphics.sketch import *
from ..graphics.shape import Clipping
from ..geometry.lattice import *
from ..canvas.canvas import *
from ..canvas.style_map import *
from ..canvas.grids import *
from ..helpers.illustration import *
from ..helpers.constraint_solver import Constraint, solve
from ..graphics.shapes import *
from ..helpers.modifiers import *
from ..lace import Lace
from ..colors.colors import *
from ..colors.palettes import *
from ..colors.pastels import *
from ..colors.swatches import *
import simetri.colors as colors
from ..tikz.tikz import *
from ..svg.svg import *
from .mask import Mask, Stop, Gradient
from ..svg.filters import *
from ..helpers.validation import check_version
from ..stars import stars
from ..stars.stars import rosette, Star
from ..wallpaper import wallpaper
from ..graphics.all_enums import *
from ..extensions.turtle_sg import Turtle, spirolateral
from ..extensions.l_system import l_system
from ..extensions.easing import *
from ..frieze.frieze_patterns import *
from ..extensions.tree import make_tree, TreeNode
from .path import LinPath, Operation
from .pattern import *
from ..image.image import Image, open_img
from .batch import *

# Preserve geometric Line class on public namespace.
from ..graphics.shapes import Line as Line

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
