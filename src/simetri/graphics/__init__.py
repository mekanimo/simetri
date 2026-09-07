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
from ..base.core import *
from ..friezes import frieze
from ..config.settings import *
from ..base.common import *


set_defaults()
from simetri import coloring as colors

from ..render.canvas import *
from ..render.grids import *
from ..render.style_map import *
from ..coloring.colors import *
from ..coloring.palettes import *
from ..coloring.pastels import *
from ..coloring.swatches import *
from ..extensions.easing import *
from ..extensions.l_system import l_system
from ..extensions.tree import TreeNode, make_tree
from ..extensions.turtle_sg import Turtle, spirolateral
from ..friezes.frieze_patterns import *
from ..geom.nonlinear.bezier import *
from ..geom.nonlinear.circle import *
from ..geom.polygons.convex_hull import convex_hull
from ..geom.nonlinear.ellipse import *
from ..geom.geom_utils import *
from ..geom.geometry import *
from ..geom.nonlinear.hobby import *
from ..patterns.lattice import *
from ..geom.polygons.polygon import *
from ..geom.nonlinear.sine import *
from ..geom.vectors import *
from ..base.all_enums import *
from ..shapes.shape import (
    Clipping,
    all_segments,
    clip,
    polygon_diff,
    polygon_difference,
    polygon_intersection,
    polygon_xor,
)
from ..shapes.geom_items import *

# Preserve geometric Line class on public namespace.
from ..shapes.geom_items import Line as Line
from ..render.sketch import *
from ..helpers.constraint_solver import Constraint, solve
from ..helpers.illustration import *
from ..helpers.modifiers import *
from ..helpers.validation import check_version
from ..images.image import Image, open_img
from ..interlace import Lace
from ..star_patterns import stars
from ..star_patterns.stars import Star, rosette
from ..render.render_svg.filters import *
from ..render.render_svg.svg import *
from ..render.render_tikz.tikz import *
from ..wallpapers import wallpaper
from ..geom.affine import *
from ..group.batch import *
from ..shapes.dots import *
from ..render.mask import Gradient, Mask, Stop
from ..geom.nonlinear.path import Operation, Path2D
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

# Explicit public re-exports (star-imports can drop or shadow these).
from ..helpers.help_utils import help  # noqa: E402, F401
