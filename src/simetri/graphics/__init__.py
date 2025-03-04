"""simetri.graphics is a module that provides a simple and intuitive way to create geometric shapes and patterns."""

# status: prototype
# This is a proof of concept.
# Testing is incomplete.
# Everything is subject to change till we release a beta version.

__version__ = "0.0.5"
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
import logging

from numpy import linspace, arange, array, zeros, ones, full, eye, diag

from ..helpers.utilities import *
from .core import *
from ..frieze import *
from ..settings.settings import *
from ..graphics.common import *


set_defaults()
from ..geometry.geometry import *
from ..geometry.ellipse import *
from ..geometry.bezier import *
from ..geometry.hobby import *
from ..geometry.circle import *
from .affine import *
from .dots import *
from ..graphics.sketch import *
from ..canvas.canvas import *
from ..canvas.grids import *
from ..helpers.illustration import *
from ..helpers.constraint_solver import Constraint, solve
from ..graphics.shapes import *
from ..helpers.modifiers import *
from ..lace import Lace
from ..colors import *
import simetri.colors as colors
from ..tikz import *
from ..helpers.validation import check_version
import simetri.stars as stars
import simetri.wallpaper as wallpaper
import simetri.frieze as frieze
from ..graphics.all_enums import *
from ..extensions.turtle_sg import Turtle, spirolateral
from ..extensions.l_system import l_system


set_tikz_defaults()

import simetri.canvas.style_map as style_map

style_map._set_shape_style_alias_map()
style_map._set_tag_style_alias_map()
style_map._set_line_style_alias_map()
style_map._set_fill_style_alias_map()
style_map._set_marker_style_alias_map()
style_map._set_pattern_style_alias_map()
style_map._set_frame_style_alias_map()
style_map._set_shape_args()
style_map._set_batch_args()


logger = logging.getLogger()
logger.setLevel(defaults["log_level"])  # defaults comes from settings.py

file_handler = logging.FileHandler(defaults["log_file"], mode="w", encoding="utf-8")
file_handler.setLevel(defaults["log_level"])

console_handler = logging.StreamHandler()
console_handler.setLevel(defaults["log_level"])

logger.addHandler(file_handler)
logger.addHandler(console_handler)

defaults["logger"] = logger

# we don't want to use the Python's Path class
from .path import Path # noqa
