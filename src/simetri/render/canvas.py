"""Canvas class for drawing shapes and text on a page.

All drawing operations go through ``Canvas``: graphics, text, pages, and
helpers for lines, circles, polygons, and related primitives.

Examples:
    >>> import simetri.graphics as sg
    >>> canvas = sg.Canvas()
    >>> canvas.draw(sg.Circle(20)) is canvas
    True
    >>> canvas.active_page.sketches[0].subtype.name
    'CIRCLE_SKETCH'
    >>> canvas.active_page.sketches[0].radius
    20.0
    >>> canvas.active_page.sketches[0].center
    [0.0, 0.0]
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import networkx as nx
import numpy as np
import pymupdf as fitz
from numpy.typing import NDArray
from PIL import Image as PIL_Image

from simetri.base.all_enums import (
    Align,
    Anchor,
    Axis,
    Drawable,
    FragmentColoring,
    ImageMode,
    PageOrientation,
    PageSize,
    PlaitStyle,
    Renderer,
    SvgLoc,
    TexLoc,
    Types,
    WarningType,
)
from simetri.base.common import (
    PointType,
    VecType,
    _set_Nones,
    alias_argument,
)
from simetri.base.common_style import (
    FILL_STYLE_ATTRS,
    LINE_STYLE_ATTRS,
    coerce_style_overlay,
)
from simetri.coloring.colors import Color
from simetri.config.settings import (
    VOID,
    defaults,
    issue_warning,
    resolve_save_filepath,
    runtime_defaults,
)
from simetri.config.user_config import (
    converter_supports_extension,
    get_converter_for_extension,
    native_save_extensions,
    user_config_path,
)
from simetri.geom.affine import (
    rotation_matrix,
    scale_in_place_matrix,
    scale_matrix,
    translation_matrix,
)
from simetri.geom.bbox import bounding_box
from simetri.geom.homogenize import homogenize
from simetri.geom.matrices import identity_matrix
from simetri.geom.nonlinear.path import Path2D
from simetri.group.batch import Group
from simetri.helpers.file_operations import (
    open_saved_file,
    run_external_converter,
    validate_output_filepath,
)
from simetri.helpers.illustration import logo
from simetri.helpers.utilities import (
    wait_for_file_availability,
)
from simetri.helpers.validation import (
    check_alpha,
    check_color,
    validate_args,
    warn_unknown_kwargs,
)
from simetri.images.image import Image, draw_on_image as draw_sketches_on_image
from simetri.interlace.lace import Lace
from simetri.notebook import display
from simetri.render import draw
from simetri.render import vector_draw
from simetri.render.render_tikz.tikz import get_tex_code
from simetri.render.render_tikz.tikz_sketch import TexSketch
from simetri.render.mask import Mask
from simetri.render.sketch import MaskedSketch
from simetri.render.style_map import canvas_args, get_draw_valid_kwargs
from simetri.render.tex import Tex, remove_aux_files, run_job
from simetri.shapes.shape import Shape


class _CanvasScope:
    """Restore canvas matrix or style when used as a context manager.

    Bare ``translate`` / ``style`` apply immediately and leave the change
    on. ``with`` pushes the saved state on enter and pops on exit.
    """

    def __init__(self, canvas: Canvas, kind: str, saved: Any) -> None:
        self._canvas = canvas
        self._kind = kind
        self._saved = saved

    def __enter__(self) -> Canvas:
        if self._kind == "matrix":
            self._canvas.matrix_stack.append(self._saved)
        elif self._kind == "style":
            self._canvas.style_stack.append(self._saved)
        else:
            raise ValueError(f"Unknown canvas scope kind {self._kind!r}")
        return self._canvas

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._kind == "matrix":
            self._canvas.pop_matrix()
        elif self._kind == "style":
            self._canvas.pop_style()
        else:
            raise ValueError(f"Unknown canvas scope kind {self._kind!r}")
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._canvas, name)


def _save_renderer(extension: str) -> Renderer:
    """Return the renderer family for the output extension."""
    if extension == ".svg":
        return Renderer.SVG
    return Renderer.TEX


def canvas_has_vertex_coord_labels(canvas: Canvas) -> bool:
    """Return True if any sketch on the canvas shows vertex coordinate labels.

    Args:
        canvas: Canvas instance to inspect.

    Returns:
        bool: True if at least one sketch has ``show_vertex_coords``.
    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.canvas import canvas_has_vertex_coord_labels
        >>> canvas_has_vertex_coord_labels(sg.Canvas())
        False
    """
    for page in canvas.pages:
        for sketch in page.sketches:
            if getattr(sketch, "show_vertex_coords", False):
                return True
    return False


def normalize_canvas_border(
    border: float | Sequence[float] | np.ndarray | None,
) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, right, top)`` border values.

    Args:
        border: Scalar border, 4-tuple, or ``None`` (uses defaults).

    Returns:
        tuple[float, float, float, float]: Normalized border sides.
    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.canvas import normalize_canvas_border
        >>> normalize_canvas_border(5)
        (5, 5, 5, 5)
    """
    if border is None:
        border = defaults["border"]
    if isinstance(border, (int, float)):
        return (border, border, border, border)
    if isinstance(border, (list, tuple, np.ndarray)) and len(border) == 4:
        return tuple(border)
    raise ValueError(
        "Canvas.border must be a numeric value or a tuple of 4 numeric values."
    )


def effective_border_for_export(
    canvas: Canvas,
) -> tuple[float, float, float, float]:
    """Return export border, optionally expanded for vertex labels.

    Args:
        canvas: Canvas instance whose border is used.

    Returns:
        tuple[float, float, float, float]: ``(left, bottom, right, top)``.
    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.canvas import effective_border_for_export
        >>> effective_border_for_export(sg.Canvas(border=5))
        (5, 5, 5, 5)
    """
    border_left, border_bottom, border_right, border_top = (
        normalize_canvas_border(canvas.border)
    )
    if defaults[
        "auto_expand_canvas_for_vertices"
    ] and canvas_has_vertex_coord_labels(canvas):
        extra = defaults["vertices_canvas_expand"]
        return (
            border_left + extra,
            border_bottom + extra,
            border_right + extra,
            border_top + extra,
        )
    return border_left, border_bottom, border_right, border_top


def warn_vertex_coord_label_sizing(canvas: Canvas) -> None:
    """Warn once per export about vertex label sizing behavior.

    Args:
        canvas: Canvas instance being exported (mutated).
    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.canvas import warn_vertex_coord_label_sizing
        >>> warn_vertex_coord_label_sizing(sg.Canvas())
    """
    if getattr(canvas, "_vertex_label_sizing_warned", False):
        return
    if not canvas_has_vertex_coord_labels(canvas):
        return

    if defaults["auto_expand_canvas_for_vertices"]:
        extra = defaults["vertices_canvas_expand"]
        issue_warning(
            f"Vertex coordinate labels use extra canvas padding of {extra}pt "
            "per side (auto_expand_canvas_for_vertices). Increase "
            "vertices_canvas_expand or canvas.border if labels are still clipped.",
            warning_type=WarningType.canvas.vertex_pad,
        )
    else:
        issue_warning(
            "Vertex coordinate labels are not included in canvas size. "
            "Increase canvas.border (e.g. 40) if labels are clipped.",
            warning_type=WarningType.canvas.vertex_size,
        )
    canvas._vertex_label_sizing_warned = True


def _oriented_page_points(
    page_size: PageSize, orientation: PageOrientation
) -> tuple[float, float]:
    """Return a standard page size in points for ``orientation``."""
    width, height = page_size.in_points()
    if orientation is PageOrientation.PORTRAIT:
        return (width, height)
    if orientation is PageOrientation.LANDSCAPE:
        return (height, width)
    raise ValueError(
        "page_orientation must be PageOrientation.PORTRAIT "
        "or PageOrientation.LANDSCAPE."
    )


class Canvas:
    """Main drawing surface for shapes, text, and pages.

    All drawing operations go through ``Canvas``. It can draw graphics and
    text objects and provides helpers for lines, circles, polygons, and more.
    Without ``page_size`` and ``page_origin``, output bounds are computed from
    the drawn entities; when they are set, they fix the output size/position.
    Canvas units are points (1 in = 72 pt), and all angles are in radians
    (2 pi = 360 degrees).

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.draw(sg.Circle(20)) is canvas
        True
        >>> canvas.active_page.sketches[0].radius
        20.0
    """

    def __init__(
        self,
        back_color: Color | None = None,
        border: float | None = None,
        page_size: VecType | PageSize | None = None,
        page_origin: PointType | None = (0, 0),
        page_orientation: PageOrientation = PageOrientation.PORTRAIT,
        **kwargs: object,
    ) -> None:
        """Create a canvas with optional background, border, and page size.

        If you use ``canvas = sg.Canvas()``, default settings are applied.

        Args:
            back_color: Background color of the canvas.
            border: Border width applied to all margins. Negative
                values clip the output.
            page_size: Page size in points, with ``page_origin`` at
                ``(0, 0)``. A ``PageSize`` member is converted to points.
                ``page_orientation`` swaps that pair for landscape.
                Calculated automatically unless specified.
            page_origin: Origin of the page coordinate system.
            page_orientation: ``PageOrientation.PORTRAIT`` or
                ``PageOrientation.LANDSCAPE``. Used when ``page_size``
                is a ``PageSize`` member.
            **kwargs: Style and positioning options such as ``fill``,
                ``line_width``, ``line_color``, and ``fill_color``.

        Returns:
            A ``Canvas`` instance.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.active_page is canvas.pages[0]
            True
            >>> canvas.active_page.sketches
            []
            >>> sg.Canvas(page_size=sg.PageSize.A3).page_size
            (841.89, 1190.55)
            >>> sg.Canvas(
            ...     page_size=sg.PageSize.A3,
            ...     page_orientation=sg.PageOrientation.LANDSCAPE,
            ... ).page_size
            (1190.55, 841.89)
        """
        validate_args(kwargs, canvas_args)
        _set_Nones(self, ["back_color", "border"], [back_color, border])
        self._origin = [0, 0]
        self._page_orientation = PageOrientation.PORTRAIT
        self._standard_page_size = None
        self._size = None
        self.page_orientation = page_orientation
        if page_size is not None:
            self.page_size = page_size
        self.border = border
        self.__dict__["margins"] = None
        self.__dict__["book_margins"] = None
        """This value is added to the bounding box of the canvas to expand the output size."""
        self.page_origin = page_origin
        self.type = Types.CANVAS
        """Used internally to identify the type of object. Do not change it!.
        Most objects in simetri has a type and subtype attribute. See ... for
        using this for user defined objects.
        """
        self.subtype = Types.CANVAS
        """Used internally to identify the type of object. Do not change it!.
        Most objects in simetri has a type and subtype attribute. See ... for
        using this for user defined objects.
        """
        self._code = []
        self._font_list = []
        self.preamble = defaults["preamble"]
        """Used for generating TikZ code."""
        self.back_color = back_color
        """Background color of the canvas."""
        self.pages = [
            Page(
                size=self.page_size,
                back_color=self.back_color,
                border=self.border,
                margins=self.margins,
                book_margins=self.book_margins,
            )
        ]
        """Each page of the canvas is a Page object.
        The canvas can have multiple pages that result in multi-page PDF output,
        or multiple images with image_name_1.svg, image_name_2.svg, etc."""
        self.active_page = self.pages[0]
        """canvas.draw() draws on the active page. If there are multiple pages,
        the active page is the last page created."""
        self._all_vertices = []
        self.drawn_entities = []
        self.draw_grid = False
        self.inset = 0

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._xform_matrix = identity_matrix()
        self._sketch_xform_matrix = identity_matrix()
        self.tex: Tex = Tex()
        self.render = defaults["render"]
        if self._size is not None:
            x, y = self.page_origin[:2]
            self._limits = [
                x,
                y,
                x + self.page_size[0],
                y + self.page_size[1],
            ]
        else:
            self._limits = None
        self.overlay = False  # used for inserting pdf pictures
        self.matrix_stack = []
        self.stack = self.matrix_stack
        self.style_stack = []
        self._style_overlay: dict[str, Any] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        """Set canvas attributes with special handling for layout properties.

        Args:
            name: Attribute name.
            value: Attribute value.

        Raises:
            ValueError: If ``border``, ``margins``, ``book_margins``, ``pos``,
                or ``angle`` is invalid.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.border = 5
            >>> canvas.border
            5
        """
        if name == "back_color":
            if hasattr(self, "active_page"):
                self.active_page.__setattr__(name, value)
            self.__dict__[name] = value
        elif name == "border":
            if value is None:
                border = None
            elif isinstance(value, (int, float)):
                border = value
            elif (
                isinstance(value, (list, tuple, np.ndarray)) and len(value) == 4
            ):
                border = tuple(value)
                if not all(isinstance(item, (int, float)) for item in border):
                    raise ValueError(
                        "Canvas.border must be a numeric value or a tuple of 4 numeric values."
                    )
            else:
                raise ValueError(
                    "Canvas.border must be a numeric value or a tuple of 4 numeric values."
                )

            if hasattr(self, "active_page"):
                self.active_page.__dict__["border"] = border
            self.__dict__["border"] = border
        elif name == "margins":
            if value is None:
                margins = (
                    defaults["margin_left"],
                    defaults["margin_bottom"],
                    defaults["margin_right"],
                    defaults["margin_top"],
                )
            elif isinstance(value, (int, float)):
                if value < 0:
                    raise ValueError(
                        "Canvas.margins must be a positive numeric value or a tuple of 4 positive numeric values."
                    )
                margins = (value, value, value, value)
            elif (
                isinstance(value, (list, tuple, np.ndarray)) and len(value) == 4
            ):
                margins = tuple(value)
                if not all(isinstance(item, (int, float)) for item in margins):
                    raise ValueError(
                        "Canvas.margins must be a positive numeric value or a tuple of 4 positive numeric values."
                    )
                if any(item < 0 for item in margins):
                    raise ValueError(
                        "Canvas.margins must be a positive numeric value or a tuple of 4 positive numeric values."
                    )
            else:
                raise ValueError(
                    "Canvas.margins must be a positive numeric value or a tuple of 4 positive numeric values."
                )

            if hasattr(self, "active_page"):
                self.active_page.margins = margins
                self.active_page.book_margins = None
            self.__dict__["margins"] = margins
            self.__dict__["book_margins"] = None
        elif name == "book_margins":
            if value is None:
                book_margins = (
                    defaults["margin_gutter"],
                    defaults["margin_footer"],
                    defaults["margin"],
                    defaults["margin_header"],
                )
            elif (
                isinstance(value, (list, tuple, np.ndarray)) and len(value) == 4
            ):
                book_margins = tuple(value)
                if not all(
                    isinstance(item, (int, float)) for item in book_margins
                ):
                    raise ValueError(
                        "Canvas.book_margins must be a tuple of 4 positive numeric values."
                    )
                if any(item < 0 for item in book_margins):
                    raise ValueError(
                        "Canvas.book_margins must be a tuple of 4 positive numeric values."
                    )
            else:
                raise ValueError(
                    "Canvas.book_margins must be a tuple of 4 positive numeric values."
                )

            recto = True
            if hasattr(self, "active_page"):
                recto = self.active_page.recto

            gutter, footer, margin, header = book_margins
            if recto:
                margins = (gutter, footer, margin, header)
            else:
                margins = (margin, footer, gutter, header)

            if hasattr(self, "active_page"):
                self.active_page.book_margins = book_margins
                self.active_page.margins = margins
            self.__dict__["book_margins"] = book_margins
            self.__dict__["margins"] = margins
        elif name in ["page_size", "page_origin", "page_orientation", "limits"]:
            if name == "page_size":
                type(self).page_size.fset(self, value)
            elif name == "page_origin":
                type(self).page_origin.fset(self, value)
            elif name == "page_orientation":
                type(self).page_orientation.fset(self, value)
            elif name == "limits":
                type(self).limits.fset(self, value)
        elif name == "size":
            raise AttributeError("Canvas.size was renamed to Canvas.page_size.")
        elif name == "origin":
            raise AttributeError(
                "Canvas.origin was renamed to Canvas.page_origin."
            )
        elif name == "scale":
            if isinstance(value, (list, tuple)):
                type(self).scale.fset(self, value[0], value[1])
            else:
                type(self).scale.fset(self, value)
        elif name == "pos":
            if isinstance(value, (list, tuple, np.ndarray)):
                type(self).pos.fset(self, value)
            else:
                raise ValueError("pos must be a list, tuple or np.ndarray.")
        elif name == "angle":
            if isinstance(value, (int, float)):
                type(self).angle.fset(self, value)
            else:
                raise ValueError("angle must be a number.")

        else:
            self.__dict__[name] = value

    def push_matrix(self) -> None:
        """Push the current transform matrix onto ``matrix_stack``.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.push_matrix()
            >>> canvas.matrix_stack[0].tolist()
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        """
        self.matrix_stack.append(self._xform_matrix.copy())

    def pop_matrix(self) -> None:
        """Pop the transform matrix from ``matrix_stack``.

        Warns if the stack is empty.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.push_matrix()
            >>> _ = canvas.translate(60, 80)
            >>> canvas.pop_matrix()
            >>> canvas.pos
            [0.0, 0.0]
        """
        if self.matrix_stack:
            self._xform_matrix = self.matrix_stack.pop()
        else:
            issue_warning(
                "Trying to pop from an empty stack!",
                warning_type=WarningType.canvas.empty_stack,
            )

    def push_style(self) -> None:
        """Push the current canvas style overlay onto ``style_stack``.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.style(line_width=2)
            >>> canvas.push_style()
            >>> canvas.style_stack
            [{'line_width': 2}]
        """
        self.style_stack.append(dict(self._style_overlay))

    def pop_style(self) -> None:
        """Pop the canvas style overlay from ``style_stack``.

        Warns if the stack is empty.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.style(line_width=2)
            >>> canvas.push_style()
            >>> _ = canvas.style(line_width=5)
            >>> canvas.pop_style()
            >>> canvas._style_overlay
            {'line_width': 2}
        """
        if self.style_stack:
            self._style_overlay = self.style_stack.pop()
        else:
            issue_warning(
                "Trying to pop from an empty stack!",
                warning_type=WarningType.canvas.empty_stack,
            )

    def style(self, mapping: Any = None, **kwargs: object) -> _CanvasScope:
        """Apply a canvas style overlay and return a restore-on-``with`` scope.

        Bare call leaves the overlay on. ``with canvas.style(...)`` restores
        the overlay from entry. Does not mutate drawables.

        Args:
            mapping: A ``Style``, a dict of draw aliases, or omitted.
            **kwargs: Draw-alias overlay; overwrites ``mapping`` for those keys.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> scope = canvas.style(line_width=2)
            >>> scope._kind
            'style'
            >>> canvas._style_overlay
            {'line_width': 2}
        """
        if mapping is None and not kwargs:
            raise TypeError(
                "canvas.style() requires a Style, a dict, or keyword arguments"
            )
        overlay = coerce_style_overlay(mapping, kwargs)
        saved = dict(self._style_overlay)
        for key, value in overlay.items():
            if value is None:
                if key in self._style_overlay:
                    del self._style_overlay[key]
            else:
                self._style_overlay[key] = value
        return _CanvasScope(self, "style", saved)

    def reset_style(self) -> Self:
        """Clear the current canvas style overlay. Does not pop ``style_stack``.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.style(line_width=2)
            >>> canvas.reset_style() is canvas
            True
            >>> canvas._style_overlay
            {}
        """
        self._style_overlay = {}
        return self

    def reset_line_style(self) -> Self:
        """Drop stroke keys from the current canvas overlay.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.style(line_width=2, fill_color='red')
            >>> canvas.reset_line_style() is canvas
            True
            >>> canvas._style_overlay
            {'fill_color': 'red'}
        """
        for key in LINE_STYLE_ATTRS:
            if key in self._style_overlay:
                del self._style_overlay[key]
        return self

    def reset_fill_style(self) -> Self:
        """Drop fill keys from the current canvas overlay.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.style(line_width=2, fill_color='red')
            >>> canvas.reset_fill_style() is canvas
            True
            >>> canvas._style_overlay
            {'line_width': 2}
        """
        for key in FILL_STYLE_ATTRS:
            if key in self._style_overlay:
                del self._style_overlay[key]
        return self

    def apply_mask(self, target: Shape | Group, mask: Mask) -> Self:
        """Apply a mask to a drawable target and append a masked sketch.

        Args:
            target: Shape or Group to mask.
            mask: Mask object applied to the target.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> from simetri.render.mask import Mask
            >>> canvas = sg.Canvas()
            >>> mask_shape = sg.Shape([(0, 0), (40, 0), (40, 40)], closed=True)
            >>> target = sg.Shape([(0, 0), (20, 0), (20, 20)], closed=True)
            >>> canvas.apply_mask(target, Mask(shape=mask_shape)) is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'MASKED_SKETCH'
        """
        if target.type == Types.GROUP:
            sketches = [draw.get_sketches(item, self) for item in target]
        else:
            sketches = [draw.get_sketches(target, self)]

        self.active_page.sketches.append(
            MaskedSketch(sketches=sketches, mask=mask)
        )
        self._all_vertices.extend(mask.shape.b_box.corners)

        return self

    def clip(
        self,
        target: Drawable,
        clipper: Shape,
        **kwargs: object,
    ) -> Self:
        """Clip a drawable target with a clipper shape.

        Args:
            target: Drawable content to clip.
            clipper: Shape used as the clipping path.
            **kwargs: Style overrides for the clipped sketch.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> target = sg.Rectangle(width=80, height=50)
            >>> clipper = sg.Rectangle(width=120, height=120)
            >>> canvas.clip(target, clipper) is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'CLIPPED_SKETCH'
        """
        # create a ClippedSketch
        # this replaces begin_clip and end_clip
        self._sketch_xform_matrix = (
            self._sketch_xform_matrix @ self._xform_matrix
        )
        self.active_page.sketches.append(
            draw.get_clipped_sketch(target, clipper, self, **kwargs)
        )
        draw.extend_vertices(self, clipper)
        self._sketch_xform_matrix = identity_matrix()

        return self

    def apply_filter(self, target: Drawable, filters: Any) -> Self:
        """Apply filters to a drawable target.

        Note:
            Currently a no-op placeholder that returns ``self``.

        Args:
            target: Drawable content to filter.
            filters: Filter specification.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.apply_filter(sg.Circle(40), None) is canvas
            True
            >>> canvas.active_page.sketches
            []
        """
        # createa FilteredSketch

        return self

    def display(self) -> None:
        """Show the canvas in a notebook cell.

        Side-effect only (returns ``None``). Do not add ``return self`` — Jupyter
        would show ``Canvas()`` after the figure. See ``agent_ground_rules.md``
        (protected notebook display).

        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.draw(sg.Circle(5))
            >>> canvas.display() is None
            <IPython.core.display.SVG object>
            True
        """
        display(self)

    @property
    def page_size(self) -> VecType:
        """
        The size of the page rectangle.

        Returns:
            VecType: The size of the page rectangle.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas(page_size=(100, 200))
            >>> canvas.page_size
            (100, 200)
        """
        return self._size

    @page_size.setter
    def page_size(self, value: VecType | PageSize) -> None:
        """
        Set the size of the page rectangle.

        Args:
            value (VecType): The size of the page rectangle.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.page_size = (100, 200)
            >>> canvas.page_size
            (100, 200)
            >>> canvas.page_size = sg.PageSize.A4
            >>> canvas.page_size
            (595.28, 841.89)
        """
        if isinstance(value, PageSize):
            self._standard_page_size = value
            value = _oriented_page_points(value, self._page_orientation)
        else:
            self._standard_page_size = None
        if len(value) == 2:
            self._size = value
            x, y = self.page_origin[:2]
            w, h = value
            self._limits = (x, y, x + w, y + h)
        else:
            raise ValueError("page_size must be a tuple of 2 values.")

    @property
    def page_orientation(self) -> PageOrientation:
        """Page orientation used for a standard ``page_size``.

        Returns:
            PageOrientation: Portrait or landscape.

        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas(page_size=sg.PageSize.A3)
            >>> canvas.page_orientation = sg.PageOrientation.LANDSCAPE
            >>> canvas.page_size
            (1190.55, 841.89)
            >>> canvas.page_orientation = sg.PageOrientation.PORTRAIT
            >>> canvas.page_size
            (841.89, 1190.55)
        """
        return self._page_orientation

    @page_orientation.setter
    def page_orientation(self, value: PageOrientation) -> None:
        """Set the orientation used for a standard ``page_size``.

        A numeric ``page_size`` is left unchanged.

        Args:
            value: ``PageOrientation.PORTRAIT`` or
                ``PageOrientation.LANDSCAPE``.

        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas(page_size=(100, 200))
            >>> canvas.page_orientation = sg.PageOrientation.LANDSCAPE
            >>> canvas.page_size
            (100, 200)
        """
        if (
            value is PageOrientation.PORTRAIT
            or value is PageOrientation.LANDSCAPE
        ):
            self._page_orientation = value
        else:
            raise ValueError(
                "page_orientation must be PageOrientation.PORTRAIT "
                "or PageOrientation.LANDSCAPE."
            )
        standard = self._standard_page_size
        if standard is None:
            return
        width, height = _oriented_page_points(standard, value)
        self._size = (width, height)
        x, y = self.page_origin[:2]
        self._limits = (x, y, x + width, y + height)

    @property
    def page_origin(self) -> VecType:
        """
        The lower-left corner of the page rectangle.

        Returns:
            VecType: The lower-left corner of the page rectangle.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas(page_origin=(60, 80))
            >>> canvas.page_origin
            (60, 80)
        """
        return self._origin[:2]

    @page_origin.setter
    def page_origin(self, value: VecType) -> None:
        """
        Set the lower-left corner of the page rectangle.

        Args:
            value (VecType): The lower-left corner of the page rectangle.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.page_origin = (40, 80)
            >>> canvas.page_origin
            (40, 80)
        """
        if len(value) == 2:
            self._origin = value
        else:
            raise ValueError("page_origin must be a tuple of 2 values.")

    @property
    def limits(self) -> VecType:
        """
        The limits of the canvas.
        [min_x, min_y, max_x, max_y]

        Returns:
            VecType: The limits of the canvas.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas(page_size=(40, 20))
            >>> canvas.limits
            (0, 0, 40, 20)
        """
        if self.page_size is None:
            res = None
        else:
            x, y = self.page_origin[:2]
            w, h = self.page_size
            res = (x, y, x + w, y + h)

        return res

    @limits.setter
    def limits(self, value: VecType) -> None:
        """
        Set the limits of the canvas.
        [min_x, min_y, max_x, max_y]

        Args:
            value (VecType): The limits of the canvas.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.limits = (0, 0, 40, 80)
            >>> canvas.page_size
            (40, 80)
        """
        if len(value) == 4:
            x1, y1, x2, y2 = value
            self._size = (x2 - x1, y2 - y1)
            self._origin = (x1, y1)
        else:
            raise ValueError("Limits must be a tuple of 4 values.")

    def b_box(self) -> BoundingBox:
        """Return the axis-aligned bounding box of drawn content.

        Returns:
            BoundingBox: All recorded vertices in canvas space.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.draw(sg.Circle(40))
            >>> canvas.b_box().width
            80.0
            >>> canvas.b_box().height
            80.0
        """
        xform = np.linalg.inv(self._xform_matrix)
        return bounding_box(homogenize(self._all_vertices) @ xform)

    def capture(self, format: str | None = None) -> Image:
        """Snapshot the canvas to an image.

        Default format is ``defaults["canvas_capture_format"]`` (``svg``).
        Native capture formats are the ``save`` formats except ``.tex``.
        Other extensions require a personal ``[converters.<ext>]`` entry.

        Args:
            format: Extension with or without a leading dot. ``None`` uses
                ``defaults["canvas_capture_format"]``.

        Returns:
            Image: Snapshot of the current drawing.

        Raises:
            ValueError: If ``format`` is ``.tex``.
            RuntimeError: If the format is not native and has no converter.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas(page_size=(80, 80), border=0)
            >>> _ = canvas.draw(sg.Circle(40))
            >>> canvas.capture().size
            (80, 80)
        """
        if format is None:
            capture_format = defaults["canvas_capture_format"]
        else:
            capture_format = format
        extension = capture_format.lower()
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension == ".tex":
            raise ValueError(
                "canvas.capture does not write .tex. Use canvas.save for TeX."
            )
        if (
            extension not in native_save_extensions()
            and not converter_supports_extension(extension)
        ):
            config_file = user_config_path()
            raise RuntimeError(
                f"Capture format {extension!r} is not supported.\n"
                "Native formats: "
                f"{', '.join(sorted(native_save_extensions()))}.\n"
                "For other formats, add a personal converter in "
                f"{config_file}, e.g.\n"
                f"[converters.{extension.lstrip('.')}]\n"
                'source = "svg"\n'
                'command = ["resvg", "{input}", "{output}"]'
            )

        handle, filepath = tempfile.mkstemp(suffix=extension)
        os.close(handle)
        keep_filepath = False
        try:
            self.save(
                filepath,
                overwrite=True,
                show=False,
                print_output=False,
                remove_aux=False,
            )
            wait_for_file_availability(filepath, timeout=5)
            if extension in native_save_extensions():
                document = fitz.open(filepath)
                page = document[0]
                pixmap = page.get_pixmap()
                if pixmap.alpha:
                    image_mode = ImageMode.RGBA
                else:
                    image_mode = ImageMode.RGB
                pil_img = PIL_Image.frombytes(
                    image_mode,
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                captured = Image(img=pil_img)
                if extension == ".svg":
                    captured.file_path = filepath
                    keep_filepath = True
                return captured
            opened = Image(img=filepath)
            return Image(img=opened.pil_img.copy())
        finally:
            if not keep_filepath and os.path.isfile(filepath):
                os.remove(filepath)

    def insert_svg(self, code: str, loc: SvgLoc = SvgLoc.PICTURE) -> Self:
        """
        Insert SVG markup into the canvas.

        Args:
            code (str): The SVG fragment to insert.
            loc (SvgLoc): The location to insert the markup.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.insert_svg('<circle cx="0" cy="0" r="10"/>') is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'SVG_SKETCH'
            >>> canvas.active_page.sketches[0].code
            '<circle cx="0" cy="0" r="10"/>'
        """
        draw.insert_svg(self, code, loc)
        return self

    def insert_tex(self, code: str, loc: TexLoc = TexLoc.PICTURE) -> Self:
        """
        Insert TeX code into the canvas.

        Args:
            code (str): The TeX to insert.
            loc (TexLoc): The location to insert the code.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.insert_tex('% note') is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'TEX_SKETCH'
            >>> canvas.active_page.sketches[0].code
            '% note'
        """
        draw.insert_tex(self, code, loc)
        return self

    @alias_argument({"radius_x": "rx", "radius_y": "ry"})
    def arc(
        self,
        center: PointType,
        radius_x: float,
        radius_y: float | None = None,
        start_angle: float = 0,
        span_angle: float | None = None,
        rot_angle: float = 0,
        *,
        end_angle: float | None = None,
        clockwise: bool = False,
        **kwargs: object,
    ) -> Self:
        """Draw an arc with the given center, radii, start angle, and sweep.

        Pass either ``span_angle`` or ``end_angle``, not both. Use
        ``clockwise=True`` or a negative ``span_angle`` for a clockwise
        arc.

        Args:
            center: The center of the arc.
            radius_x: Semi-axis along x.
            radius_y: Semi-axis along y; defaults to ``radius_x``.
            start_angle: The start angle of the arc in radians. Defaults to 0.
            span_angle: Sweep in radians. A negative value draws clockwise.
                Mutually exclusive with ``end_angle``. At least one of
                ``span_angle`` or ``end_angle`` is required.
            rot_angle: The rotation angle of the arc. Defaults to 0.
            end_angle: Ending angle in radians. Mutually exclusive with
                ``span_angle``. Pass as a keyword.
            clockwise: If True, the arc is drawn clockwise. Defaults to False.
            kwargs: Additional keyword arguments.

        Returns:
            Self: The canvas object.

        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.arc((0, 0), 40, 40, 0, sg.pi / 2, 0) is canvas
            True
            >>> sketch = canvas.active_page.sketches[-1]
            >>> sketch.subtype.name
            'ARC_SKETCH'
            >>> tuple(round(coord, 6) for coord in sketch.vertices[0][:2])
            (40.0, 0.0)
            >>> tuple(round(coord, 6) for coord in sketch.vertices[-1][:2])
            (0.0, 40.0)
        """
        draw.arc(
            self,
            center,
            radius_x,
            radius_y,
            start_angle,
            span_angle,
            rot_angle,
            end_angle=end_angle,
            clockwise=clockwise,
            **kwargs,
        )
        return self

    def bezier(
        self, control_points: Sequence[PointType], **kwargs: object
    ) -> Self:
        """
        Draw a bezier curve.

        Args:
            control_points (Sequence[PointType]): The control points of the bezier curve.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.bezier([(0, 0), (20, 40), (40, 0)]) is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'BEZIER_SKETCH'
            >>> canvas.active_page.sketches[0].control_points
            [(0.0, 0.0, 1.0), (20.0, 40.0, 1.0), (40.0, 0.0, 1.0)]
        """
        draw.bezier(self, control_points, **kwargs)
        return self

    def circle(
        self, radius: float, center: PointType = (0, 0), **kwargs: object
    ) -> Self:
        """
        Draw a circle with the given radius and optional center.

        Args:
            radius (float): The radius of the circle.
            center (PointType): The center of the circle. Defaults to (0, 0).
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.circle(40, (0, 0)) is canvas
            True
            >>> canvas.active_page.sketches[-1].subtype.name
            'CIRCLE_SKETCH'
            >>> canvas.active_page.sketches[-1].radius
            40
            >>> canvas.active_page.sketches[-1].center
            [0.0, 0.0]
        """
        draw.circle(self, radius, center, **kwargs)
        return self

    def ellipse(
        self,
        width: float,
        height: float,
        center: PointType = (0, 0),
        angle: float = 0,
        **kwargs: object,
    ) -> Self:
        """
        Draw an ellipse with the given width, height, and optional center.

        Args:
            width (float): The width of the ellipse.
            height (float): The height of the ellipse.
            center (PointType): The center of the ellipse. Defaults to (0, 0).
            angle (float, optional): The angle of the ellipse, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.ellipse(20, 40) is canvas
            True
            >>> canvas.active_page.sketches[-1].subtype.name
            'ELLIPSE_SKETCH'
            >>> canvas.active_page.sketches[-1].x_radius
            10.0
            >>> canvas.active_page.sketches[-1].y_radius
            20.0
        """
        draw.ellipse(self, width, height, center, angle, **kwargs)

        return self

    def draw_fragments(
        self,
        lace: Lace | None = None,
        palette: Sequence[Color] | None = None,
        **kwargs: object,
    ) -> Self:
        """Draw lace fragment regions, optionally colored by a palette.

        Args:
            lace: Lace object whose fragments are drawn.
            palette: Color palette applied to fragments.
            **kwargs: Style overrides forwarded to the draw helper.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> lace = sg.Lace(
            ... sg.Group(
            ... sg.Shape([(0, -70), (50, 70), (100, -70), (150, 70), (200, -70)]),
            ... sg.Line((-40, 0), (240, 0)),
            ... ).scale(2),
            ... offset=12,
            ... )
            >>> canvas.draw_fragments(lace) is canvas
            True
            >>> [sketch.subtype.name for sketch in canvas.active_page.sketches]
            ['SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH']
        """
        draw.draw_fragments(self, lace, palette, **kwargs)

        return self

    def draw_plaits(self, lace: Lace | None = None, **kwargs: object) -> Self:
        """Draw lace plaits.

        Args:
            lace: Lace object whose plaits are drawn.
            **kwargs: Style overrides forwarded to the draw helper.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> lace = sg.Lace(
            ... sg.Group(
            ... sg.Shape([(0, -70), (50, 70), (100, -70), (150, 70), (200, -70)]),
            ... sg.Line((-40, 0), (240, 0)),
            ... ).scale(2),
            ... offset=12,
            ... )
            >>> canvas.draw_plaits(lace) is canvas
            True
            >>> [sketch.subtype.name for sketch in canvas.active_page.sketches]
            ['SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH']
        """
        draw.draw_plaits(self, lace, **kwargs)

        return self

    def draw_lace_with_fillets(self, lace: Lace, **kwargs: object) -> Self:
        """Draw a lace with filleted plait geometry.

        Args:
            lace: Lace object to draw.
            **kwargs: Style overrides forwarded to the draw helper.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> lace = sg.Lace(
            ... sg.Group(
            ... sg.Shape([(0, -70), (50, 70), (100, -70), (150, 70), (200, -70)]),
            ... sg.Line((-40, 0), (240, 0)),
            ... ).scale(2),
            ... offset=12,
            ... )
            >>> canvas.draw_lace_with_fillets(lace, fillet_radii=(80, 80)) is canvas
            True
            >>> [sketch.subtype.name for sketch in canvas.active_page.sketches]
            []
        """
        draw.draw_lace_with_fillets(self, lace, **kwargs)

        return self

    def text(
        self,
        text: str,
        pos: PointType,
        font_family: str | None = None,
        font_size: int | None = None,
        font_color: Color | None = None,
        anchor: Anchor | None = None,
        align: Align | None = None,
        **kwargs: object,
    ) -> Self:
        """
        Draw text at the given point.

        Args:
            text (str): The text to draw.
            pos (PointType): The position to draw the text.
            font_family (str, optional): The font family of the text, defaults to None.
            font_size (int, optional): The font size of the text, defaults to None.
            anchor (Anchor, optional): The anchor of the text, defaults to None.
            anchor options: BASE, BASE_EAST, BASE_WEST, BOTTOM, CENTER, EAST, NORTH,
            NORTHEAST, NORTHWEST, SOUTH, SOUTHEAST, SOUTHWEST, WEST, MIDEAST, MIDWEST, RIGHT,
            LEFT, TOP
            align (Align, optional): The alignment of the text, defaults to Align.CENTER.
            align options: CENTER, FLUSH_CENTER, FLUSH_LEFT, FLUSH_RIGHT, JUSTIFY, LEFT, RIGHT
            kwargs (dict): Additional keyword arguments.
            common kwargs: fill_color, line_color, line_width, fill, line, alpha, font_color

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.text('A', (0, 0)) is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'TAG_SKETCH'
            >>> canvas.active_page.sketches[0].text
            'A'
            >>> canvas.active_page.sketches[0].pos
            [0.0, 0.0]
        """
        draw.text(
            self,
            txt=text,
            pos=pos,
            font_family=font_family,
            font_size=font_size,
            font_color=font_color,
            anchor=anchor,
            align=align,
            **kwargs,
        )
        return self

    def text_path(
        self,
        text: str,
        path: Path2D | Shape,
        font_family: str | None = None,
        font_size: int | None = None,
        font_color: Color | None = None,
        bold: bool = False,
        italic: bool = False,
        draw_path: bool = False,
        **kwargs: object,
    ) -> Self:
        """Draw text along a path.

        Args:
            text: The text to place on the path.
            path: A ``Path2D`` or ``Shape``.
            font_family: Font family. Defaults to None.
            font_size: Font size. Defaults to None.
            font_color: Text color. Defaults to None.
            bold: Bold type. Defaults to False.
            italic: Italic type. Defaults to False.
            draw_path: If True, also stroke the guide path. Defaults to False.
            kwargs: Extra attributes stored on the ``TextPath``.

        Returns:
            Self: The canvas object.

        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> curve = sg.Path2D((0, 0)).line_to((80, 0))
            >>> canvas.text_path("along", curve) is canvas
            True
            >>> canvas.active_page.sketches[-1].subtype.name
            'TEXT_PATH_SKETCH'
        """
        draw.text_path(
            self,
            txt=text,
            path=path,
            font_family=font_family,
            font_size=font_size,
            font_color=font_color,
            bold=bold,
            italic=italic,
            draw_path=draw_path,
            **kwargs,
        )
        return self

    def help_lines(
        self,
        pos: tuple[float, float] | None = None,
        width: float | None = None,
        height: float | None = None,
        spacing: float | None = None,
        cs_size: float | None = None,
        deferred: bool = True,
        **kwargs: object,
    ) -> Self:
        """
        Draw help lines on the canvas.

        Args:
            pos (tuple): The lower-left corner of the grid.
            width (float): The length of the help lines along the x-axis.
            height (float): The length of the help lines along the y-axis.
            spacing (int): The spacing between the help lines.
            cs_size (float): The size of the coordinate system.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.help_lines((0, 0), 20, 20, spacing=40, cs_size=0, deferred=False) is canvas
            True
            >>> [tuple(sketch.vertices) for sketch in canvas.active_page.sketches]
            [((0.0, 40.0), (20.0, 40.0)), ((40.0, 0.0), (40.0, 20.0))]
        """
        if spacing is None:
            spacing = defaults["help_lines_spacing"]
        if cs_size is None:
            cs_size = defaults["CS_size"]
        if pos is None:
            margin = defaults["help_lines_margin"]
            pos = (-margin, -margin)
        if width is None:
            width = defaults["help_lines_width"]
        if height is None:
            height = defaults["help_lines_height"]

        draw.help_lines(
            self, pos, width, height, spacing, cs_size, deferred, **kwargs
        )

        return self

    def grid(
        self,
        pos: PointType,
        width: float,
        height: float,
        spacing: float,
        **kwargs: object,
    ) -> Self:
        """
        Draw a grid with the given size and spacing.

        Args:
            pos (PointType): The position to start drawing the grid.
            width (float): The length of the grid along the x-axis.
            height (float): The length of the grid along the y-axis.
            spacing (float): The spacing between the grid lines.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.grid((0, 0), 40, 40, 20) is canvas
            True
            >>> [tuple(sketch.vertices) for sketch in canvas.active_page.sketches]
            [((0.0, 0.0), (40.0, 0.0)), ((0.0, 20.0), (40.0, 20.0)), ((0.0, 40.0), (40.0, 40.0)), ((0.0, 0.0), (0.0, 40.0)), ((20.0, 0.0), (20.0, 40.0)), ((40.0, 0.0), (40.0, 40.0))]
        """
        draw.grid(self, pos, width, height, spacing, **kwargs)
        return self

    def line(self, start: PointType, end: PointType, **kwargs: object) -> Self:
        """
        Draw a line from start to end.

        Args:
            start (PointType): The starting point of the line.
            end (PointType): The ending point of the line.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.line((0, 0), (40, 0)) is canvas
            True
            >>> canvas.active_page.sketches[0].vertices
            [(0.0, 0.0), (40.0, 0.0)]
        """
        draw.line(self, start, end, **kwargs)
        return self

    def rectangle(
        self,
        width: float | None = None,
        height: float | None = None,
        center: PointType = (0, 0),
        angle: float = 0,
        **kwargs: object,
    ) -> Self:
        """
        Draw a rectangle (width and height first, default center ``(0, 0)``).

        Args:
            width (float): The width of the rectangle. ``None`` uses
                ``defaults["rectangle_width_height"]``.
            height (float): The height of the rectangle. ``None`` uses
                ``defaults["rectangle_width_height"]``.
            center (PointType): The center of the rectangle. Defaults to (0, 0).
            angle (float, optional): The angle of the rectangle, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.rectangle(40, 60) is canvas
            True
            >>> canvas.active_page.sketches[0].vertices
            [(-20.0, 30.0), (-20.0, -30.0), (20.0, -30.0), (20.0, 30.0)]
        """
        if width is None or height is None:
            default_width, default_height = defaults["rectangle_width_height"]
            if width is None:
                width = default_width
            if height is None:
                height = default_height
        draw.rectangle(self, width, height, center, angle, **kwargs)
        return self

    def rectangle2(
        self,
        corner1: PointType = (0, 0),
        corner2: PointType = (0, 0),
        angle: float = 0,
        **kwargs: object,
    ) -> Self:
        """
        Draw a rectangle.

        Args:
            corner1 (PointType): The first corner of the rectangle.
            corner2 (PointType): The diagonally opposing corner.
            angle (float, optional): The angle of the rectangle, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.rectangle2((0, 0), (40, 60)) is canvas
            True
            >>> canvas.active_page.sketches[0].vertices
            [(0.0, 60.0), (0.0, 0.0), (40.0, 0.0), (40.0, 60.0)]
        """
        x1, y1 = corner1
        x2, y2 = corner2
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        draw.rectangle(self, width, height, center, angle, **kwargs)
        return self

    def rectangle3(
        self,
        upper_left: PointType,
        width: float | None = None,
        height: float | None = None,
        angle: float = 0,
        **kwargs: object,
    ) -> Self:
        """
        Draw a rectangle from the upper-left corner.

        Args:
            upper_left (PointType): The upper_left corner of the rectangle.
            width (float): The width of the rectangle. ``None`` uses
                ``defaults["rectangle_width_height"]``.
            height (float): The height of the rectangle. ``None`` uses
                ``defaults["rectangle_width_height"]``.
            angle (float, optional): The angle of the rectangle, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.rectangle3((0, 0), 80, 40) is canvas
            True
            >>> canvas.active_page.sketches[0].vertices
            [(0.0, 0.0), (0.0, -40.0), (80.0, -40.0), (80.0, 0.0)]
        """
        if width is None or height is None:
            default_width, default_height = defaults["rectangle_width_height"]
            if width is None:
                width = default_width
            if height is None:
                height = default_height
        x1, y1 = upper_left[:2]
        x2, y2 = x1 + width, y1 - height
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        draw.rectangle(self, width, height, center, angle, **kwargs)
        return self

    def square(
        self,
        size: float | None = None,
        center: PointType = (0, 0),
        angle: float = 0,
        **kwargs: object,
    ) -> Self:
        """
        Draw a square (side ``size`` first, default center ``(0, 0)``).

        Args:
            size (float): The size of the square. ``None`` uses
                ``defaults["square_size"]``.
            center (PointType): The center of the square. Defaults to (0, 0).
            angle (float, optional): The angle of the square, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.square(8) is canvas
            True
            >>> canvas.active_page.sketches[0].vertices
            [(-4.0, 4.0), (-4.0, -4.0), (4.0, -4.0), (4.0, 4.0)]
        """
        if size is None:
            size = defaults["square_size"]
        draw.rectangle(self, size, size, center, angle, **kwargs)
        return self

    def lines(self, points: Sequence[PointType], **kwargs: object) -> Self:
        """
        Draw a polyline through the given points.

        Args:
            points (Sequence[PointType]): The points to draw the polyline through.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.lines([(0, 0), (40, 0), (40, 20)]) is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'LINE_SKETCH'
            >>> canvas.active_page.sketches[0].vertices
            [(0.0, 0.0), (40.0, 0.0), (40.0, 20.0)]
        """
        draw.lines(self, points, **kwargs)
        return self

    def draw_lace(
        self,
        lace: Lace,
        *,
        fragment_coloring: FragmentColoring | None = None,
        plait_style: PlaitStyle | None = None,
        shade_plaits: bool | None = None,
        fillet_radii: tuple[float, float] | None = None,
        palette: Sequence[Color] | None = None,
        swatch: Sequence[Color] | None = None,
        plait_color: Color | None = None,
        draw_fragments: bool | None = None,
        draw_plaits: bool | None = None,
        percent_offsets: Sequence[float] | None = None,
        line_widths: Sequence[float] | None = None,
        **kwargs: object,
    ) -> Self:
        """Draw the lace object.

        Lace-specific options belong here, not on ``canvas.draw``. Generic
        shape styles in ``kwargs`` are forwarded to fragment and plait draws.

        Args:
            lace: Lace object to be drawn.
            fragment_coloring: Color equivalent fragments by area, or by
                distance from the lace center.
            plait_style: Plait drawing style. ``None`` uses filled plaits.
            shade_plaits: Shade embossed or diamond plaits.
            fillet_radii: ``(inner, outer)`` fillet radii. ``None`` does not
                fillet.
            palette: Fragment color palette.
            swatch: Overrides ``palette`` when given.
            plait_color: Fill color for plaits.
            draw_fragments: Draw fragment regions.
            draw_plaits: Draw plaits.
            percent_offsets: Inner-line positions for ``PlaitStyle.INNERLINES``.
            line_widths: Inner-line widths for ``PlaitStyle.INNERLINES``.
            **kwargs: Generic shape style overrides.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> lace = sg.Lace(
            ... sg.Group(
            ... sg.Shape([(0, -70), (50, 70), (100, -70), (150, 70), (200, -70)]),
            ... sg.Line((-40, 0), (240, 0)),
            ... ).scale(2),
            ... offset=12,
            ... )
            >>> canvas.draw_lace(lace) is canvas
            True
            >>> [sketch.subtype.name for sketch in canvas.active_page.sketches]
            ['SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH']
        """
        draw.draw_lace(
            self,
            lace,
            fragment_coloring=fragment_coloring,
            plait_style=plait_style,
            shade_plaits=shade_plaits,
            fillet_radii=fillet_radii,
            palette=palette,
            swatch=swatch,
            plait_color=plait_color,
            draw_fragments=draw_fragments,
            draw_plaits=draw_plaits,
            percent_offsets=percent_offsets,
            line_widths=line_widths,
            **kwargs,
        )
        return self

    def draw_dimension(self, dim: Shape, **kwargs: object) -> Self:
        """
        Draw the dimension.

        Args:
            dim (Shape): The dimension to draw.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> dim = sg.Dimension((0, 0), (40, 0), 'up', 20)
            >>> canvas.draw_dimension(dim) is canvas
            True
            >>> [sketch.subtype.name for sketch in canvas.active_page.sketches]
            ['LINE_SKETCH', 'LINE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'TAG_SKETCH']
            >>> canvas.active_page.sketches[-1].text
            '40.0'
        """
        draw.draw_dimension(self, dim, **kwargs)
        return self

    def draw_widget(self, item: Drawable, **kwargs: object) -> Self:
        """Draw an item by expanding ``item.draw_list`` into a composite sketch.

        Args:
            item (Drawable): Widget-like drawable with a ``draw_list``.
            **kwargs: Style overrides forwarded to the draw helper.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> widget = sg.Group()
            >>> widget.draw_list = [sg.Rectangle(width=56, height=24)]
            >>> canvas.draw_widget(widget) is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'COMPOSITE_SKETCH'
        """
        draw.draw_widget(self, item, **kwargs)
        return self

    def begin_style(self, style: str) -> Self:
        """Begin a TikZ scope that appends ``style`` to every path.

        Args:
            style: TikZ style fragment inserted into the scope options.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.begin_style('dashed') is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'TEX_SKETCH'
            >>> 'dashed' in canvas.active_page.sketches[0].code
            True
        """
        # code = rf'\begin{{scope}}[every path/.append style={{dashed, draw=green}}]'
        code = rf"\begin{{scope}}[every path/.append style={{ {style} }}]"
        code += "\n"
        sketch = TexSketch(code)
        self.active_page.sketches.append(sketch)

        return self

    def end_style(self) -> Self:
        """End the TikZ style scope started by ``begin_style``.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.begin_style('dashed')
            >>> canvas.end_style() is canvas
            True
            >>> print(canvas.active_page.sketches[-1].code)
            \end{scope}
            <BLANKLINE>
        """
        return self._end_scope()

    def _end_scope(self) -> Self:
        sketch = TexSketch("\\end{scope}\n")
        self.active_page.sketches.append(sketch)

        return self

    def draw(
        self,
        *item_s: "Drawable | Sequence",
        pos: PointType = None,
        angle: float = 0,
        rotocenter: PointType = (0, 0),
        scale: VecType | float = (1, 1),
        about: PointType = (0, 0),
        show: bool = False,
        **kwargs: object,
    ) -> Self:
        """
        Draw the item_s. Pass items individually or as one sequence.

        Args:
            *item_s (Drawable | Sequence): Item(s) to draw. Use
                ``draw(a, b)`` or ``draw([a, b])``. A ``Vector`` is
                drawn as a line with an arrow head from the origin to
                ``(x, y)``. ``vec_start`` places the tail; ``vec_end``
                places the tip. If both are omitted, the tail is at the
                origin. Only one is needed. If both are given and match
                the Vector, a warning is issued; if they disagree,
                ``ValueError`` is raised. Several Vectors in one call
                share the same ``vec_start`` or ``vec_end``.
                ``shaft_line_color`` / ``shaft_line_dash_array`` /
                ``shaft_line_width`` style the shaft;
                ``head_fill_color`` / ``head_line_color`` /
                ``head_line_width`` style the head; ``color`` and
                ``alpha`` style both. A ``Dimension`` uses those arrow
                names for its dimension line, ``ext_line_alpha`` /
                ``ext_line_color`` / ``ext_line_dash_array`` /
                ``ext_line_width`` for both extension lines, and
                ``tag_bold`` / ``tag_fill`` / ``tag_fill_color`` /
                ``tag_font_alpha`` / ``tag_font_color`` /
                ``tag_font_family`` / ``tag_font_size`` /
                ``tag_line_color`` / ``tag_line_width`` / ``tag_stroke``
                for the label.
                ``tag_stroke`` defaults to False. ``color`` and
                ``alpha`` set every part; a prefixed name wins.
                An ``AnnotationArrow`` uses the same shaft, head, and
                tag names. Shaft style covers the angled shaft and the
                landing. Its ``tag_stroke`` stays as it is unless
                given: False for a note, True for a balloon.
            pos (PointType, optional): Midpoint where the item is drawn.
                For a group, this is the group's midpoint; every member is
                shifted by the same ``(dx, dy)``. The item is not moved.
                Defaults to None.
            angle (float, optional): The angle to rotate the item(s), defaults to 0.
            rotocenter (PointType, optional): The point about which to rotate, defaults to (0, 0).
            scale (tuple, optional): The scale factors for the x and y axes, defaults to (1, 1).
            about (tuple, optional): The point about which to scale, defaults to (0, 0).
            show (bool, optional): If True, draws the canvas in a Jupyter cell.
            filter (SVG_Filter, optional): SVG filter object to apply to the drawn item(s).
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> triangle = sg.Shape([(0, 0), (40, 0), (40, 40)], closed=True)
            >>> canvas.draw(triangle, fill_color=sg.blue, pos=(40, 0))
            Canvas()
        """
        warn_unknown_kwargs(
            kwargs,
            get_draw_valid_kwargs(),
            context="canvas.draw",
            stacklevel=3,
        )
        if not item_s:
            raise TypeError("Canvas.draw() requires at least one item")
        if len(item_s) == 1 and isinstance(item_s[0], (list, tuple)):
            items = item_s[0]
        else:
            items = item_s

        vector_draw.prepare_vector_batch(items, kwargs, pos)

        base_sketch_xform = self._sketch_xform_matrix

        for item in items:
            if vector_draw.is_vector_marker_shape(item, kwargs):
                vector_draw.draw_vector_marker_shape(
                    self,
                    item,
                    pos=pos,
                    angle=angle,
                    rotocenter=rotocenter,
                    scale=scale,
                    about=about,
                    draw_kwargs=kwargs,
                )
                continue

            drawable, part_kwargs = vector_draw.drawable_for_item(
                item, kwargs
            )
            sketch_xform = base_sketch_xform
            if pos is not None:
                mid_x, mid_y = drawable.midpoint[:2]
                dest_x, dest_y = pos[:2]
                dx = dest_x - mid_x
                dy = dest_y - mid_y
                sketch_xform = translation_matrix(dx, dy) @ sketch_xform
            if scale[0] != 1 or scale[1] != 1:
                sketch_xform = (
                    scale_in_place_matrix(*scale[:2], about) @ sketch_xform
                )
            if angle != 0:
                sketch_xform = (
                    rotation_matrix(angle, rotocenter) @ sketch_xform
                )
            self._sketch_xform_matrix = self._xform_matrix @ sketch_xform
            draw.draw(self, drawable, **part_kwargs)

        self._sketch_xform_matrix = identity_matrix()
        if show:
            self.display()
        else:
            return self

    def draw_lines(
        self, lines: Sequence[tuple[float, float]], **kwargs: object
    ) -> Self:
        """These lines are drawn with the same style.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.draw_lines([((0, 0), (48, 0)), ((48, 0), (48, 24))]) is canvas
            True
            >>> [tuple(sketch.vertices) for sketch in canvas.active_page.sketches]
            [((0.0, 0.0), (48.0, 0.0)), ((48.0, 0.0), (48.0, 24.0))]
        """
        draw.draw_lines(self, lines, **kwargs)

        return self

    def draw_points(
        self,
        points: Sequence[PointType],
        marker_style: object | None = None,
        **kwargs: object,
    ) -> Self:
        """Draw markers at the given points.

        If ``marker_style`` is omitted, default marker values are used.
        A ``MarkerStyle``, style dict, or ``marker_*`` keyword arguments
        override those defaults.

        Args:
            points: Sequence of ``(x, y)`` positions.
            marker_style: Optional marker style. Defaults to None (library
                marker defaults).
            **kwargs: Extra style forwarded to ``draw``.

        Returns:
            Self: The canvas.

        Raises:
            ValueError: If ``points`` is empty.

        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.draw_points([(0, 0), (40, 0)]) is canvas
            True
            >>> sketch = canvas.active_page.sketches[0]
            >>> sketch.draw_markers
            True
            >>> sketch.markers_only
            True
            >>> sketch.marker_type == sg.defaults["marker_type"]
            True
            >>> tuple(sketch.vertices)
            ((0.0, 0.0), (40.0, 0.0))
        """
        return draw.draw_points(
            self, points, marker_style=marker_style, **kwargs
        )

    def draw_CS(self, size: float | None = None, **kwargs: object) -> Self:
        """
        Draw the Canvas coordinate system.

        Args:
            size (float, optional): The size of the coordinate system, defaults to None.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.draw_CS(40) is canvas
            True
            >>> [sketch.subtype.name for sketch in canvas.active_page.sketches]
            ['SHAPE_SKETCH', 'SHAPE_SKETCH', 'CIRCLE_SKETCH']
            >>> canvas.active_page.sketches[0].vertices
            [(0.0, 0.0), (40.0, 0.0)]
            >>> canvas.active_page.sketches[1].vertices
            [(0.0, 0.0), (0.0, 40.0)]
            >>> canvas.active_page.sketches[2].radius
            2
        """
        draw.draw_CS(self, size, **kwargs)
        return self

    def draw_pdf(
        self,
        pdf: str | Path | Any,
        pos: PointType,
        size: VecType | float | None = None,
        scale: float | VecType | None = None,
        angle: float = 0,
        **kwargs: object,
    ) -> Self:
        """Draw a PDF on the canvas.

        Args:
            pdf: PDF object or file path.
            pos: Upper-left position to draw the PDF at.
            size: Optional display size.
            scale: Optional scale factor.
            angle: Rotation angle in radians.
            **kwargs: Forwarded to the draw helper.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.draw_pdf('missing.pdf', (0, 0)) is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'PDF_SKETCH'
            >>> canvas.active_page.sketches[0].file_path
            'missing.pdf'
        """
        draw.draw_pdf(self, pdf, pos, size, scale, angle, **kwargs)
        return self

    def draw_image(
        self, image: Image, pos: PointType, **kwargs: object
    ) -> Self:
        """
        Draw an image on the canvas.

        Args:
            image (Image): The image to draw.
            pos (PointType): The position to draw the image at.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> image = sg.Image(size=(2, 2), mode='RGB')
            >>> canvas.draw_image(image, (0, 0)) is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'IMAGE_SKETCH'
            >>> canvas.active_page.sketches[0].pos
            [0.0, 0.0]
        """
        draw.draw_image(self, image, pos, **kwargs)
        return self

    def draw_on_image(
        self, item: Drawable, image: Image, **kwargs: object
    ) -> Image:
        """Draw an item on a copy of an image.

        Args:
            item: Shape or group to draw.
            image: Simetri image used as the pixel base.
            **kwargs: Style overrides applied while creating sketches.

        Returns:
            Image: New image with the item drawn on it.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> image = sg.Image(size=(4, 4), mode='RGB')
            >>> out = canvas.draw_on_image(sg.Circle(2), image)
            >>> out.size
            (4, 4)
        """
        warn_unknown_kwargs(
            kwargs,
            get_draw_valid_kwargs(),
            context="canvas.draw_on_image",
            stacklevel=3,
        )
        sketches = draw.get_sketches(item, self, **kwargs)

        return draw_sketches_on_image(sketches, image)

    def save_image(self, image: Image, filepath: Path, **params: Any) -> Self:
        """Save an image to a file.

        Args:
            image: Simetri image to save.
            filepath: Output file path.
            **params: Extra Pillow save parameters.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> import tempfile
            >>> from pathlib import Path
            >>> canvas = sg.Canvas()
            >>> image = sg.Image(size=(2, 2), mode='RGB')
            >>> path = Path(tempfile.mkstemp(suffix='.png')[1])
            >>> canvas.save_image(image, path) is canvas
            True
            >>> path.exists()
            True
        """
        image.save(filepath, **params)

        return self

    def draw_latex(
        self,
        formula: str,
        pos: PointType,
        font_size: int = 14,
        font_family: str | None = None,
        font_color: Color | None = None,
        bold: bool = False,
        anchor: Anchor | None = None,
        **kwargs: object,
    ) -> Self:
        """Draw a LaTeX math formula on the canvas using matplotlib mathtext (no TeX compiler needed).

        Args:
            formula (str): LaTeX math string without surrounding $. E.g. r'\\frac{a}{b}'.
                Text-mode commands are silently mapped to their math-mode equivalents:
                \\texttt → \\mathtt (monospace), \\textrm → \\mathrm, \\textbf → \\mathbf,
                \\textit → \\mathit, \\textsf → \\mathsf.
            pos (PointType): Canvas position for the formula anchor.
            font_size (int): Font size in points. Defaults to 14.
            font_family (str, optional): Mathtext fontset — 'computer modern'/'cm', 'stix',
                'stix sans'/'stixsans', 'dejavu sans'/'dejavusans', 'dejavu serif'/'dejavuserif'.
                If omitted and the formula contains \\mathbf{}, STIX is chosen automatically
                (closest to LaTeX output). Otherwise matplotlib's current default is used.
            font_color: Formula colour — simetri Color, (r,g,b) tuple, or matplotlib colour
                string (e.g. 'red', '#ff0000'). Defaults to black.
            bold (bool): Wrap the *entire* formula in \\mathbf{}. For partial bold, write
                \\mathbf{} directly in the formula string — STIX is still selected automatically.
                Defaults to False.
            anchor: Anchor point for the formula box. Defaults to Anchor.SOUTHWEST.
            **kwargs: Additional keyword arguments.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.draw_latex('x', (0, 0)) is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'LATEX_SKETCH'
            >>> canvas.active_page.sketches[0].formula
            'x'
            >>> canvas.active_page.sketches[0].pos
            [0.0, 0.0]
        """
        draw.draw_latex(
            self,
            formula,
            pos,
            font_size=font_size,
            font_family=font_family,
            font_color=font_color,
            bold=bold,
            anchor=anchor,
            **kwargs,
        )
        return self

    def reset(self) -> Self:
        """
        Reset the canvas to its initial state.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.draw(sg.Circle(40))
            >>> canvas.reset() is canvas
            True
            >>> canvas.active_page.sketches
            []
            >>> canvas.pos
            [0.0, 0.0]
        """
        self._code = []
        self.preamble = defaults["preamble"]
        self.back_color = defaults["back_color"]
        self.border = defaults["canvas_border"]
        page_margins = self.margins
        if self.book_margins is not None:
            gutter, footer, margin, header = self.book_margins
            page_margins = (gutter, footer, margin, header)
        self.pages = [
            Page(
                size=self.page_size,
                back_color=self.back_color,
                border=self.border,
                margins=page_margins,
                book_margins=self.book_margins,
            )
        ]
        self.active_page = self.pages[0]
        self._all_vertices = []
        self.tex: Tex = Tex()
        self._xform_matrix = identity_matrix()
        self._sketch_xform_matrix = identity_matrix()
        self.active_page = self.pages[0]
        self._all_vertices = []

        return self

    def __str__(self) -> str:
        """
        Return a string representation of the canvas.

        Returns:
            str: The string representation of the canvas.
        Examples:
            >>> import simetri.graphics as sg
            >>> str(sg.Canvas())
            'Canvas()'
        """
        return "Canvas()"

    def __repr__(self) -> str:
        """
        Return a string representation of the canvas.

        Returns:
            str: The string representation of the canvas.
        Examples:
            >>> import simetri.graphics as sg
            >>> repr(sg.Canvas())
            'Canvas()'
        """
        return "Canvas()"

    @property
    def pos(self) -> PointType:
        """
        The position of the canvas.

        Args:
            point (PointType, optional): The point to set the position to.

        Returns:
            PointType: The position of the canvas.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.translate(60, 80)
            >>> canvas.pos
            [60.0, 80.0]
        """

        return self._xform_matrix[2, :2].tolist()[:2]

    @pos.setter
    def pos(self, point: PointType) -> None:
        """
        Set the position of the canvas.

        Args:
            point (PointType): The point to set the position to.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.pos = (20, 60)
            >>> canvas.pos
            [20.0, 60.0]
        """
        self._xform_matrix[2, :2] = point[:2]

    @property
    def angle(self) -> float:
        """
        The angle of the canvas.

        Returns:
            float: The angle of the canvas.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.angle = sg.pi / 4
            >>> round(canvas.angle, 4)
            0.7854
        """
        xform = self._xform_matrix

        return np.arctan2(xform[0, 1], xform[0, 0])

    @angle.setter
    def angle(self, angle: float) -> None:
        """
        Set the angle of the canvas.

        Args:
            angle (float): The angle to set the canvas to.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.rotate(sg.pi / 2)
            >>> round(canvas.angle, 4)
            1.5708
        """
        self._xform_matrix = rotation_matrix(angle) @ self._xform_matrix

    @property
    def scale_xy(self) -> VecType:
        """
        The scale of the canvas.

        Returns:
            VecType: The scale of the canvas.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.scale(2, 3)
            >>> canvas.scale_xy
            (2.0, 3.0)
        """
        xform = self._xform_matrix

        return np.linalg.norm(xform[:2, 0]), np.linalg.norm(xform[:2, 1])

    @scale_xy.setter
    def scale_xy(
        self,
        scale_x: float = 1,
        scale_y: float | None = None,
        about: PointType = (0, 0),
    ) -> None:
        """
        Set the scale of the canvas.

        Args:
            scale_x (float): The x-scale to set the canvas to.
            scale_y (float): The y-scale to set the canvas to.
            about (PointType): The point about which to scale the canvas.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.scale_xy = (2, 2)
            >>> canvas.scale_xy
            (80.0, 80.0)
        """
        if scale_y is None:
            scale_y = scale_x

        self._xform_matrix = self._xform_matrix @ scale_in_place_matrix(
            scale_x, scale_y, about=about
        )

    @property
    def xform_matrix(self) -> NDArray:
        """
        The transformation matrix of the canvas.

        Returns:
            np.ndarray: The transformation matrix of the canvas.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.xform_matrix.tolist()
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        """
        return self._xform_matrix.copy()

    def transform(self, transform_matrix: NDArray) -> Self:
        """
        Transforms the canvas by the given transformation matrix.

        Args:
            transform_matrix (np.ndarray): The transformation matrix.

        Returns:
            Self: The Canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> import numpy as np
            >>> canvas = sg.Canvas()
            >>> canvas.transform(np.eye(3)) is canvas
            True
            >>> canvas.xform_matrix.tolist()
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            >>> _ = canvas.transform(sg.translation_matrix(60, 0))
            >>> canvas.pos
            [60.0, 0.0]
        """
        self._xform_matrix = transform_matrix @ self._xform_matrix

        return self

    def reset_transform(self) -> Self:
        """
        Reset the transformation matrix of the canvas.
        The canvas origin is at (0, 0) and the orientation angle is 0.
        Transformation matrix is the identity matrix.

        Does not pop ``matrix_stack``. Does not touch ``style_stack``.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> _ = canvas.translate(40, 80)
            >>> canvas.reset_transform() is canvas
            True
            >>> canvas.pos
            [0.0, 0.0]
        """
        self._xform_matrix = identity_matrix()

        return self

    def translate(self, dx: float, dy: float) -> _CanvasScope:
        """
        Translate the canvas by dx and dy.

        Bare call leaves the translation on. ``with canvas.translate(dx, dy)``
        restores the matrix from entry.

        Args:
            dx (float): The translation distance along the x-axis.
            dy (float): The translation distance along the y-axis.

        Returns:
            A scope that restores the matrix when used as a context manager.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> scope = canvas.translate(60, 80)
            >>> scope._kind
            'matrix'
            >>> canvas.pos
            [60.0, 80.0]
        """
        saved = self._xform_matrix.copy()
        self._xform_matrix = translation_matrix(dx, dy) @ self._xform_matrix
        return _CanvasScope(self, "matrix", saved)

    def rotate(self, angle: float, about: PointType = (0, 0)) -> _CanvasScope:
        """
        Rotate the canvas by angle in radians about the given point.

        Bare call leaves the rotation on. ``with canvas.rotate(angle)``
        restores the matrix from entry.

        Args:
            angle (float): The rotation angle in radians.
            about (tuple): The point about which to rotate the canvas.

        Returns:
            A scope that restores the matrix when used as a context manager.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> with canvas.rotate(sg.pi / 2):
            ...     round(canvas.angle, 4)
            1.5708
            >>> round(canvas.angle, 4)
            0.0
        """
        saved = self._xform_matrix.copy()
        self._xform_matrix = rotation_matrix(angle, about) @ self._xform_matrix
        return _CanvasScope(self, "matrix", saved)

    @alias_argument({"scale_x": "sx", "scale_y": "sy"})
    def scale(
        self,
        scale_x: float,
        scale_y: float | None = None,
        about: PointType = (0, 0),
    ) -> _CanvasScope:
        """
        Scale the canvas by scale_x and scale_y about the given point.
        If scale_y is not given then scale_y = scale_x.

        Bare call leaves the scale on. ``with canvas.scale(...)`` restores
        the matrix from entry.

        Args:
            scale_x (float): The scale factor in x direction.
            scale_y (float): The scale factor in y direction.

        Returns:
            A scope that restores the matrix when used as a context manager.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> with canvas.scale(2, 2):
            ...     canvas.scale_xy
            (2.0, 2.0)
            >>> canvas.scale_xy
            (1.0, 1.0)
        """
        if scale_y is None:
            scale_y = scale_x
        saved = self._xform_matrix.copy()
        self._xform_matrix = (
            scale_in_place_matrix(scale_x, scale_y, about) @ self._xform_matrix
        )
        return _CanvasScope(self, "matrix", saved)

    def _flip(self, axis: Axis) -> Self:
        """
        Flip the canvas along the specified axis.

        Args:
            axis (str): The axis to flip the canvas along ('x' or 'y').

        Returns:
            Self: The canvas object.
        """
        if axis == Axis.X:
            sx = -self.scale[0]
            sy = 1
        elif axis == Axis.Y:
            sx = 1
            sy = -self.scale[1]

        self._xform_matrix = scale_matrix(sx, sy) @ self._xform_matrix

        return self

    def flip_x_axis(self) -> Self:
        """
        Flip the x-axis direction. Warning: This will reverse the positive rotation direction.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.flip_x_axis()
            >>> canvas.xform_matrix.tolist()
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        """
        issue_warning(
            "Flipping the x-axis will change the positive rotation direction.",
            warning_type=WarningType.canvas.flip_x,
        )
        saved = self._xform_matrix.copy()
        self._flip(Axis.X)
        return _CanvasScope(self, "matrix", saved)

    def flip_y_axis(self) -> Self:
        """
        Flip the y-axis direction.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.flip_y_axis()
            >>> canvas.xform_matrix.tolist()
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        """
        issue_warning(
            "Flipping the y-axis will reverse the positive rotation direction.",
            warning_type=WarningType.canvas.flip_y,
        )
        saved = self._xform_matrix.copy()
        self._flip(Axis.Y)
        return _CanvasScope(self, "matrix", saved)

    @property
    def x(self) -> float:
        """
        The x coordinate of the canvas origin.

        Returns:
            float: The x coordinate of the canvas origin.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.x
            0.0
        """
        return self.pos[0]

    @x.setter
    def x(self, value: float) -> None:
        """
        Set the x coordinate of the canvas origin.

        Args:
            value (float): The x coordinate to set.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.x = 7
            >>> canvas.x
            7
        """
        self.pos = [value, self.pos[1]]

    @property
    def y(self) -> float:
        """
        The y coordinate of the canvas origin.

        Returns:
            float: The y coordinate of the canvas origin.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.y
            0.0
        """
        return self.pos[1]

    @y.setter
    def y(self, value: float) -> None:
        """
        Set the y coordinate of the canvas origin.

        Args:
            value (float): The y coordinate to set.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.y = 8
            >>> canvas.y
            8
        """
        self.pos = [self.pos[0], value]

    def group_graph(self, group: "Group") -> nx.DiGraph:
        """
        Return a directed graph of the group and its elements.
        Canvas is the root of the graph.
        Graph nodes are the ids of the elements.

        Args:
            group (Group): The group to create the graph from.

        Returns:
            nx.DiGraph: The directed graph of the group and its elements.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> group = sg.Group([sg.Circle(40)])
            >>> graph = canvas.group_graph(group)
            >>> graph.number_of_nodes()
            3
            >>> graph.number_of_edges()
            2
            >>> list(graph.successors(canvas.id))
            [group.id]
            >>> list(graph.successors(group.id))
            [group.elements[0].id]
        """

        def add_group(group: Group, graph: nx.DiGraph) -> nx.DiGraph:
            """Recursively register ``group`` and its elements on ``graph``.

            Examples:
                Called from ``group_graph`` for nested groups.
            """
            graph.add_node(group.id)
            for item in group.elements:
                graph.add_edge(group.id, item.id)
                if item.subtype == Types.GROUP:
                    add_group(item, graph)
            return graph

        di_graph = nx.DiGraph()
        di_graph.add_edge(self.id, group.id)
        for item in group.elements:
            if item.subtype == Types.GROUP:
                di_graph.add_edge(group.id, item.id)
                add_group(item, di_graph)
            else:
                di_graph.add_edge(group.id, item.id)

        return di_graph

    def resolve_property(self, item: Drawable, property_name: str) -> Any:
        """Resolve a property from the item, then ``defaults``.

        Does **not** apply ``canvas.draw`` kwargs or the canvas style overlay.
        For full draw-time precedence (kwargs → overlay → item → defaults),
        use ``resolve_style_properties``.

        Args:
            item: Drawable whose attribute is read when not ``None``.
            property_name: Style field name.

        Returns:
            Any: ``getattr(item, name)`` or the configured default when unset.

        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> circle = sg.Circle(40)
            >>> circle.line_width = 3
            >>> canvas.resolve_property(circle, 'line_width')
            3
        """
        value = getattr(item, property_name, None)
        if value is None:
            value = runtime_defaults.get(property_name, VOID)
            if value == VOID and property_name not in ("color", "alpha"):
                issue_warning(
                    f"Property {property_name} is not in defaults.",
                    warning_type=WarningType.canvas.missing_default,
                )
                value = None
        return value

    def resolve_style_properties(
        self,
        item: Drawable,
        style_map: dict[str, str],
        **draw_kwargs: object,
    ) -> dict[str, Any]:
        """Resolve style values for sketch creation in one place.

        Precedence per key (see ``ground_rules.md``): draw kwargs, then
        canvas style overlay (``layered``), then ``resolve_property`` (item,
        then defaults). Color and alpha fan-out are handled before the
        style-map loop.

        Examples:
            >>> import simetri.graphics as sg
            >>> from simetri.render.style_map import shape_style_map
            >>> canvas = sg.Canvas()
            >>> shape = sg.Shape([(0, 0), (40, 0), (40, 40)], closed=True)
            >>> shape.line_width = 3
            >>> resolved = canvas.resolve_style_properties(shape, shape_style_map)
            >>> resolved['line_width']
            3
        """
        overlay = self._style_overlay

        def layered(name: str) -> tuple[bool, Any]:
            """Return whether ``name`` is set in kwargs or canvas style overlay.

            Examples:
                Used by ``resolve_style_properties`` for kwargs and overlay.
            """
            if name in draw_kwargs:
                return True, draw_kwargs[name]
            if name in overlay:
                return True, overlay[name]
            return False, None

        d_resolved = {}
        resolved = []
        # handle color
        color = None
        found_color, color_value = layered("color")
        found_line_color, line_color_value = layered("line_color")
        found_fill_color, fill_color_value = layered("fill_color")
        if found_color:
            if check_color(color_value):
                d_resolved["color"] = color_value
                resolved.append("color")
                if not found_line_color:
                    d_resolved["line_color"] = color_value
                    resolved.append("line_color")
                if not found_fill_color:
                    d_resolved["fill_color"] = color_value
                    resolved.append("fill_color")
            else:
                raise ValueError(f"Invalid color value: {color_value}")
        else:
            item_color = getattr(item, "color", None)
            if item_color is not None:
                if check_color(item_color):
                    color = item_color
                else:
                    raise ValueError(f"Invalid color value: {item_color}")

        if found_line_color:
            d_resolved["line_color"] = line_color_value
            if "line_color" not in resolved:
                resolved.append("line_color")
        if found_fill_color:
            d_resolved["fill_color"] = fill_color_value
            if "fill_color" not in resolved:
                resolved.append("fill_color")

        if color is not None:
            if "line_color" not in resolved:
                if item.line_color is None:
                    d_resolved["line_color"] = color
                else:
                    d_resolved["line_color"] = item.line_color
                resolved.append("line_color")
            if "fill_color" not in resolved:
                if item.fill_color is None:
                    d_resolved["fill_color"] = color
                else:
                    d_resolved["fill_color"] = item.fill_color
                resolved.append("fill_color")
            if "color" not in resolved:
                resolved.append("color")

        # handle alpha
        alpha = None
        found_alpha, alpha_value = layered("alpha")
        found_line_alpha, line_alpha_value = layered("line_alpha")
        found_fill_alpha, fill_alpha_value = layered("fill_alpha")
        if found_alpha:
            if check_alpha(alpha_value):
                d_resolved["alpha"] = alpha_value
                resolved.append("alpha")
                if not found_line_alpha:
                    d_resolved["line_alpha"] = alpha_value
                    resolved.append("line_alpha")
                if not found_fill_alpha:
                    d_resolved["fill_alpha"] = alpha_value
                    resolved.append("fill_alpha")
            else:
                raise ValueError(f"Invalid alpha value: {alpha_value}")
        else:
            item_alpha = getattr(item, "alpha", None)
            if item_alpha is not None:
                if check_alpha(item_alpha):
                    alpha = item_alpha
                else:
                    raise ValueError(f"Invalid alpha value: {item_alpha}")

        if found_line_alpha:
            d_resolved["line_alpha"] = line_alpha_value
            if "line_alpha" not in resolved:
                resolved.append("line_alpha")
        if found_fill_alpha:
            d_resolved["fill_alpha"] = fill_alpha_value
            if "fill_alpha" not in resolved:
                resolved.append("fill_alpha")

        if alpha is not None:
            if "line_alpha" not in resolved:
                if item.line_alpha is None:
                    d_resolved["line_alpha"] = alpha
                else:
                    d_resolved["line_alpha"] = item.line_alpha
                resolved.append("line_alpha")
            if "fill_alpha" not in resolved:
                if item.fill_alpha is None:
                    d_resolved["fill_alpha"] = alpha
                else:
                    d_resolved["fill_alpha"] = item.fill_alpha
                resolved.append("fill_alpha")
            if "alpha" not in resolved:
                resolved.append("alpha")

        for attrib_name in style_map:
            if attrib_name in resolved:
                continue
            found_layer, layer_value = layered(attrib_name)
            if found_layer:
                d_resolved[attrib_name] = layer_value
            else:
                d_resolved[attrib_name] = self.resolve_property(
                    item, attrib_name
                )
            resolved.append(attrib_name)

        return d_resolved

    def draw_all_segments(
        self,
        item: Shape | Group,
        vert_indices: bool = False,
        **kwargs: object,
    ) -> Self:
        """Split edges into segments and draw them with indices.

        Used with ``get_loop``-style workflows.

        Args:
            item: Shape or group to annotate.
            vert_indices: If True, label vertices; otherwise label edges.
            **kwargs: Style overrides forwarded to the draw helper.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> shape = sg.Shape([(0, 0), (40, 0), (40, 40), (0, 40)])
            >>> canvas.draw_all_segments(shape) is canvas
            True
            >>> [sketch.subtype.name for sketch in canvas.active_page.sketches]
            ['SHAPE_SKETCH', 'TAG_SKETCH', 'SHAPE_SKETCH', 'TAG_SKETCH']
            >>> [sketch.text for sketch in canvas.active_page.sketches if sketch.subtype.name == 'TAG_SKETCH']
            ['0', '1']
        """

        return draw.draw_all_segments(self, item, vert_indices, **kwargs)

    def draw_bbox(
        self,
        bbox: BoundingBox,
        border: bool | dict[str, object] = False,
        centerlines: bool | dict[str, object] = False,
        diagonals: bool | dict[str, object] = False,
        **kwargs: object,
    ) -> Self:
        """Draw a bounding box.

        If ``border``, ``centerlines``, and ``diagonals`` are all False, the
        box is drawn as a bounding-box sketch (same as ``canvas.draw(bbox)``).
        If a flag is True, those lines use the default line color and width.
        If a flag is a style dict, those lines use that style.

        Args:
            bbox: Bounding box to draw.
            border: Rectangle outline. Defaults to False.
            centerlines: Horizontal and vertical centerlines. Defaults to False.
            diagonals: Both diagonals. Defaults to False.
            **kwargs: Extra style forwarded to each drawn part, or to the
                bounding-box sketch when all flags are False.

        Returns:
            Self: The canvas.

        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> box = sg.BoundingBox((0, 0), (40, 20))
            >>> canvas.draw_bbox(box) is canvas
            True
            >>> canvas.active_page.sketches[0].subtype.name
            'BBOX_SKETCH'
            >>> len(canvas._all_vertices)
            4
        """
        return draw.draw_bbox(
            self,
            bbox,
            border=border,
            centerlines=centerlines,
            diagonals=diagonals,
            **kwargs,
        )

    def get_fonts_list(self) -> list[str]:
        """
        Get the list of fonts used in the canvas.

        Returns:
            list[str]: The list of fonts used in the canvas.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.get_fonts_list()
            []
        """
        user_fonts = set(self._font_list)

        latex_fonts = {
            defaults["main_font"],
            defaults["sans_font"],
            defaults["mono_font"],
            "serif",
            "sansserif",
            "monospace",
        }

        for sketch in self.active_page.sketches:
            if sketch.subtype == Types.TAG_SKETCH:
                name = sketch.font_family
                if name is not None and name not in latex_fonts:
                    user_fonts.add(name)
        return list(user_fonts.difference(latex_fonts))

    def set_page_size(self, width: float, height: float) -> None:
        """Set the active page size.

        Args:
            width: Page width.
            height: Page height.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> canvas.set_page_size(100, 200)
            >>> canvas.page_size
            (100, 200)
        """
        self.page_size = (width, height)

    def _calculate_size(
        self,
        border: float | Sequence[float] | np.ndarray | None = None,
        b_box: BoundingBox | None = None,
    ) -> tuple[float, float, float, float] | None:
        """Calculate canvas size from content bounding box and border.

        Args:
            border: Border width or four-side tuple. ``None`` uses ``self.border``.
            b_box: Bounding box. ``None`` is computed from ``self._all_vertices``.

        Returns:
            tuple[float, float] | None: ``(width, height, offset_x, offset_y)`` or
            ``None`` when there are no vertices.
        """
        vertices = self._all_vertices
        if vertices:
            if b_box is None:
                b_box = bounding_box(vertices)

            if border is None:
                if self.border is None:
                    border = defaults["border"]
                else:
                    border = self.border
            if isinstance(border, (int, float)):
                border_left = border
                border_bottom = border
                border_right = border
                border_top = border
            elif (
                isinstance(border, (list, tuple, np.ndarray))
                and len(border) == 4
            ):
                border_left, border_bottom, border_right, border_top = border
            else:
                raise ValueError(
                    "Canvas.border must be a numeric value or a tuple of 4 numeric values."
                )
            w = b_box.width + border_left + border_right
            h = b_box.height + border_bottom + border_top
            offset_x, offset_y = b_box.southwest
            res = w, h, offset_x - border_left, offset_y - border_bottom
        else:
            res = None
        return res

    def _sketch_bbox(
        self, sketch: Any
    ) -> tuple[float, float, float, float] | None:
        """Return axis-aligned bbox ``(xmin, ymin, xmax, ymax)`` for a sketch."""
        sketch_data = sketch.__dict__

        if "vertices" in sketch_data and sketch.vertices:
            sketch_bbox = bounding_box(sketch.vertices)
            min_x, min_y = sketch_bbox.southwest[:2]
            max_x, max_y = sketch_bbox.northeast[:2]
            return min_x, min_y, max_x, max_y

        if sketch.subtype == Types.CIRCLE_SKETCH:
            center_x, center_y = sketch.center[:2]
            radius = sketch.radius
            return (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            )

        if sketch.subtype == Types.ELLIPSE_SKETCH:
            center_x, center_y = sketch.center[:2]
            return (
                center_x - sketch.x_radius,
                center_y - sketch.y_radius,
                center_x + sketch.x_radius,
                center_y + sketch.y_radius,
            )

        if sketch.subtype == Types.RECTANGLE_SKETCH:
            min_x, min_y = sketch.lower_left[:2]
            return min_x, min_y, min_x + sketch.width, min_y + sketch.height

        return None

    def _warn_sketches_outside_page(self) -> None:
        """Warn when a sketch is completely outside page limits."""
        if self.page_size is None:
            return

        page_limits = self.limits
        page_min_x, page_min_y, page_max_x, page_max_y = page_limits

        for page_index, page in enumerate(self.pages, start=1):
            for sketch in page.sketches:
                sketch_bbox = self._sketch_bbox(sketch)
                if sketch_bbox is None:
                    continue

                sketch_min_x, sketch_min_y, sketch_max_x, sketch_max_y = (
                    sketch_bbox
                )
                is_outside = (
                    sketch_max_x < page_min_x
                    or sketch_min_x > page_max_x
                    or sketch_max_y < page_min_y
                    or sketch_min_y > page_max_y
                )
                if is_outside:
                    issue_warning(
                        "Sketch is completely outside page limits: "
                        f"page={page_index}, subtype={sketch.subtype}, id={sketch.id}, "
                        f"bbox={sketch_bbox}, limits={page_limits}.",
                        warning_type=WarningType.canvas.outside_page,
                    )

    def _show_browser(
        self, filepath: Path, show_browser: bool, multi_page_svg: bool
    ) -> None:
        """
        Open the saved file with the personal ``[viewer]`` setting.

        Args:
            filepath (Path): The path to the file.
            show_browser (bool): Whether to open the file.
            multi_page_svg (bool): Whether the file is a multi-page SVG.
        """
        if show_browser is None:
            show_browser = defaults["show_browser"]
        if show_browser:
            if multi_page_svg:
                root, extension = os.path.splitext(str(filepath))
                for i, _ in enumerate(self.pages):
                    open_saved_file(f"{root}_{i + 1}{extension}")
            else:
                open_saved_file(filepath)

    def save(
        self,
        filepath: Path,
        overwrite: bool | None = None,
        show: bool | None = None,
        print_output: bool = False,
        remove_aux: bool = True,
        inset: float | None = None,
        display: bool = False,
    ) -> Self:
        """Save the canvas to a file.

        Args:
            filepath: Output path.
            overwrite: Whether to overwrite an existing file.
            show: Whether to open the file after save (uses ``[viewer]`` config).
            print_output: Print compiler output when saving TeX-backed formats.
            remove_aux: Remove auxiliary TeX files after PDF generation.
            inset: Clip this margin from all sides before export.
            display: Show the canvas in a notebook after save.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> import tempfile
            >>> from pathlib import Path
            >>> canvas = sg.Canvas()
            >>> _ = canvas.draw(sg.Circle(40))
            >>> path = Path(tempfile.mkstemp(suffix='.svg')[1])
            >>> canvas.save(path, overwrite=True, show=False) is canvas
            True
            >>> path.exists()
            True
        """

        if inset is not None:
            self.inset = inset

        self._vertex_label_sizing_warned = False
        self._warn_sketches_outside_page()

        filepath = resolve_save_filepath(filepath)
        parent_dir, file_name, extension = validate_output_filepath(
            filepath, overwrite
        )

        if (
            extension not in native_save_extensions()
            and converter_supports_extension(extension)
        ):
            converter = get_converter_for_extension(extension)
            source_extension = f".{converter['source']}"
            intermediate_path = os.path.join(
                parent_dir, f".{file_name}.simetri_src{source_extension}"
            )
            try:
                self.save(
                    intermediate_path,
                    overwrite=True,
                    show=False,
                    print_output=print_output,
                    remove_aux=remove_aux,
                    inset=None,
                    display=False,
                )
                run_external_converter(
                    input_path=intermediate_path,
                    output_path=filepath,
                    extension=extension,
                )
            finally:
                if os.path.isfile(intermediate_path):
                    os.remove(intermediate_path)
            self._show_browser(
                filepath=filepath, show_browser=show, multi_page_svg=False
            )
            return self

        renderer = _save_renderer(extension)
        multi_page_svg = False
        if renderer == Renderer.SVG:
            from simetri.render.render_svg.svg import get_svg_code

            if len(self.pages) > 1:
                multi_page_svg = True
                active_page = self.active_page
                for i, page in enumerate(self.pages):
                    self.active_page = page
                    page_filepath = os.path.join(
                        parent_dir, f"{file_name}_{i + 1}{extension}"
                    )
                    validate_output_filepath(page_filepath, overwrite)
                    svg_code = get_svg_code(self)
                    with open(page_filepath, "w", encoding="utf-8") as f:
                        f.write(svg_code)
                self.active_page = active_page
            else:
                svg_code = get_svg_code(self)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(svg_code)
        else:
            tex_code = get_tex_code(self)
            tex_path = os.path.join(parent_dir, file_name + ".tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tex_code)
            if extension == ".tex":
                return self

            run_job(parent_dir, file_name, extension, tex_path)
            if remove_aux:
                remove_aux_files(filepath)

        self._show_browser(
            filepath=filepath, show_browser=show, multi_page_svg=multi_page_svg
        )
        return self

    def new_page(self, **kwargs: object) -> Self:
        """Create a new page and append it to ``pages``.

        Args:
            **kwargs: Attributes set on the new ``Page`` instance.

        Returns:
            Self: The canvas object.
        Examples:
            >>> import simetri.graphics as sg
            >>> canvas = sg.Canvas()
            >>> first = canvas.active_page
            >>> canvas.new_page() is canvas
            True
            >>> canvas.active_page is canvas.pages[1]
            True
            >>> canvas.active_page is first
            False
        """
        recto = not self.active_page.recto
        page_margins = self.margins
        if self.book_margins is not None:
            gutter, footer, margin, header = self.book_margins
            if recto:
                page_margins = (gutter, footer, margin, header)
            else:
                page_margins = (margin, footer, gutter, header)

        page = Page(
            size=self.page_size,
            back_color=self.back_color,
            border=self.border,
            margins=page_margins,
            book_margins=self.book_margins,
            recto=recto,
        )
        self.pages.append(page)
        self.active_page = page
        for k, v in kwargs.items():
            setattr(page, k, v)
        if page.book_margins is not None:
            gutter, footer, margin, header = page.book_margins
            if page.recto:
                page.margins = (gutter, footer, margin, header)
            else:
                page.margins = (margin, footer, gutter, header)
        self.__dict__["margins"] = page.margins
        self.__dict__["book_margins"] = page.book_margins
        return self


@dataclass
class PageGrid:
    """
    Grid class for drawing grids on a page.

    Args:
        spacing (float, optional): The spacing between grid lines.
        back_color (Color, optional): The background color of the grid.
        line_color (Color, optional): The color of the grid lines.
        line_width (float, optional): The width of the grid lines.
        line_dash_array (Sequence[float], optional): The dash array for the grid lines.
        x_shift (float, optional): The x-axis shift of the grid.
        y_shift (float, optional): The y-axis shift of the grid.
    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.canvas import PageGrid
        >>> PageGrid().spacing
        18
    """

    spacing: float | None = None
    back_color: "Color" = None
    line_color: "Color" = None
    line_width: float | None = None
    line_dash_array: Sequence[float] = None
    x_shift: float | None = None
    y_shift: float | None = None

    def __post_init__(self) -> None:
        """Initialize page-grid defaults from settings.
        Examples:
            >>> import simetri.graphics as sg
            >>> from simetri.render.canvas import PageGrid
            >>> pg = PageGrid()
            >>> pg.type.name
            'PAGE_GRID'
        """
        self.type = Types.PAGE_GRID
        self.subtype = Types.RECTANGULAR
        self.spacing = defaults["page_grid_spacing"]
        self.back_color = defaults["page_grid_back_color"]
        self.line_color = defaults["page_grid_line_color"]
        self.line_width = defaults["page_grid_line_width"]
        self.line_dash_array = defaults["page_grid_line_dash_array"]
        self.x_shift = defaults["page_grid_x_shift"]
        self.y_shift = defaults["page_grid_y_shift"]


@dataclass
class Page:
    """
    Page class for drawing sketches and text on a page. All drawing
    operations result as sketches on the canvas.active_page.

    Args:
        size (VecType, optional): The size of the page.
        back_color (Color, optional): The background color of the page.
        mask (Any, optional): The mask of the page.
        margins (Any, optional): The margins of the page (left, bottom, right, top).
        recto (bool, optional): Whether the page is recto (True) or verso (False).
        grid (PageGrid, optional): The grid of the page.
        kwargs (dict, optional): Additional keyword arguments.
    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.canvas import Page
        >>> Page().type.name
        'PAGE'
    """

    size: VecType = None
    back_color: "Color" = None
    border: Any = None
    mask: Any = None
    margins: Any = None  # left, bottom, right, top
    book_margins: Any = None  # gutter, footer, margin, header
    recto: bool = True  # True if page is recto, False if verso
    grid: PageGrid = None
    kwargs: dict | None = None

    def __post_init__(self) -> None:
        """Initialize page metadata and an empty sketch list.
        Examples:
            >>> import simetri.graphics as sg
            >>> from simetri.render.canvas import Page
            >>> Page().sketches
            []
        """
        self.type = Types.PAGE
        self.sketches = []
        self.scope_groups = []
        if self.grid is None:
            self.grid = PageGrid()
        if self.kwargs:
            for k, v in self.kwargs.items():
                setattr(self, k, v)


def hello() -> None:
    """
    Show a hello message.
    Used for testing an installation of simetri.
    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.canvas import hello
        >>> hello()  # doctest: +SKIP
    """
    canvas = Canvas()
    import simetri.graphics as sg

    canvas.text(
        f"Hello from simetri.graphics version Alpha {sg.__version__}!",
        (0, -130),
        bold=True,
        font_size=20,
    )
    canvas.draw(logo())

    d_path = os.path.dirname(os.path.abspath(__file__))
    f_path = os.path.join(d_path, "hello.svg")

    canvas.save(f_path, overwrite=True)
