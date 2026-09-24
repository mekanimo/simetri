"""Wallpaper (planar) pattern symmetries for Simetri.

Provides the seventeen wallpaper groups and hexagonal/rhombic covering
helpers. Prefer ``lattice`` for newer lattice APIs.

Examples:
    >>> from simetri.config.settings import set_defaults
    >>> set_defaults()
    >>> from simetri.shapes.shape import Shape
    >>> from simetri.wallpapers import wallpaper as wp
    >>> len(wp.wallpaper_p1(Shape([(0, 0), (10, 0)]), (20, 0), (0, 15), 1, 1))
    4
"""
