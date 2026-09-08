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

from functools import lru_cache as memoize
from itertools import combinations, cycle, permutations, product
from math import (
    atan,
    atan2,
    ceil,
    comb,
    cos,
    degrees,
    e,
    exp,
    factorial,
    floor,
    gcd,
    hypot,
    log,
    log10,
    perm,
    pi,
    prod,
    radians,
    sin,
    sqrt,
    tau,
    trunc,
)
from random import choice, choices, randint, random, shuffle, uniform

from numpy import arange, array, diag, eye, full, linspace, ones, zeros

from ..base.common import *
from ..base.core import *
from ..config.settings import *
from ..friezes import frieze
from ..helpers.utilities import *

set_defaults()
from simetri import coloring as colors

from ..base.all_enums import *
from ..coloring.colors import *
from ..coloring.palettes import *
from ..coloring.pastels import *
from ..coloring.swatches import *
from ..extensions.easing import *
from ..extensions.l_system import l_system
from ..extensions.tree import TreeNode, make_tree
from ..extensions.turtle_sg import Turtle, spirolateral
from ..friezes.frieze_patterns import *
from ..geom.affine import *
from ..geom.geom_utils import *
from ..geom.geometry import *
from ..geom.nonlinear.bezier import *
from ..geom.nonlinear.circle import *
from ..geom.nonlinear.ellipse import *
from ..geom.nonlinear.hobby import *
from ..geom.nonlinear.path import Operation, Path2D
from ..geom.nonlinear.sine import *
from ..geom.points.point_utils import *
from ..geom.polygons.convex_hull import convex_hull
from ..geom.polygons.polygon import *
from ..geom.polygons.polygon_utils import *
from ..geom.segments.line_utils import *
from ..geom.vectors import *
from ..group.batch import *
from ..helpers.constraint_solver import Constraint, solve
from ..helpers.illustration import *
from ..helpers.modifiers import *
from ..helpers.validation import check_version
from ..images.image import Image, open_img
from ..interlace import Lace
from ..patterns.lattice import *
from ..patterns.pattern import *
from ..render.canvas import *
from ..render.gradient import Gradient, Stop
from ..render.grids import *
from ..render.mask import Mask
from ..render.render_svg.filters import *
from ..render.render_svg.svg import *
from ..render.render_tikz.tikz import *
from ..render.sketch import *
from ..render.style_map import *
from ..shapes.dots import *
from ..shapes.geom_items import *

# Preserve geometric Line class on public namespace.
from ..shapes.geom_items import Line as Line
from ..shapes.shape import (
    Clipping,
    all_segments,
    clip,
    polygon_diff,
    polygon_difference,
    polygon_intersection,
    polygon_xor,
)
from ..star_patterns import stars
from ..star_patterns.stars import Star, rosette
from ..wallpapers import wallpaper

set_tikz_defaults()
set_svg_defaults()

# Restore public enum symbols that are shadowed by later star imports.
from ..base.all_enums import Side as Side

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
from ..helpers.help_utils import help
