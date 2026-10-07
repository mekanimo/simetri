"""Wallpaper (planar) pattern symmetries for Simetri.

Provides the seventeen wallpaper groups and hexagonal/rhombic covering
helpers. Prefer ``lattice`` for newer lattice APIs.

Examples:
    >>> import simetri.graphics as sg
    >>> from simetri.wallpapers import wallpaper as wp
    >>> len(wp.wallpaper_p1(sg.Shape([(0, 0), (40, 0)]), (20, 0), (0, 60), 1, 1))
    4
"""
