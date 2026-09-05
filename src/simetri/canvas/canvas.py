"""Canvas class for drawing shapes and text on a page. All drawing
operations are handled by the Canvas class. Canvas class can draw all
graphics objects and text objects. It also provides methods for
drawing basic shapes like lines, circles, and polygons.
"""

import os
import shutil
import sys
import tempfile
import time
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any, Self

import fitz
import networkx as nx
import numpy as np

from simetri.canvas import draw
from simetri.canvas.style_map import canvas_args, get_draw_valid_kwargs
from simetri.colors.colors import Color
from simetri.geometry.homogenize import homogenize
from simetri.geometry.affine import (
    identity_matrix,
    rotation_matrix,
    scale_in_place_matrix,
    scale_matrix,
    translation_matrix,
)
from simetri.core.all_enums import (
    Align,
    Anchor,
    Axis,
    Drawable,
    Renderer,
    TexLoc,
    Types,
)
from simetri.group.batch import Group
from simetri.geometry.bbox import bounding_box
from simetri.core.common import (
    VOID,
    PointType,
    VecType,
    _set_Nones,
)
from simetri.graphics.shape import Shape
from simetri.canvas.sketch import MaskedSketch
from simetri.helpers.file_operations import validate_filepath
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
from simetri.image.image import Image, create_image_from_data
from simetri.notebook import display
from simetri.settings.settings import defaults, issue_warning
from simetri.tex.tex import Tex, remove_aux_files, run_job
from simetri.tikz.tikz import get_tex_code
from simetri.tikz.tikz_sketch import TexSketch


def _save_renderer(extension: str) -> Renderer:
    """Return the renderer family for the output extension."""
    if extension in (".svg", ".png"):
        return Renderer.SVG
    return Renderer.TEX


def save_svg_png(svg_code: str, filepath: Path) -> None:
    """Rasterize SVG code to a PNG file.

    Args:
        svg_code (str): SVG document text.
        filepath (Path): Destination PNG path.
    """
    document = fitz.open(stream=svg_code.encode("utf-8"), filetype="svg")
    page = document[0]
    pixmap = page.get_pixmap()
    pixmap.save(filepath)
    document.close()


def canvas_has_vertex_coord_labels(canvas) -> bool:
    """Return True if any sketch on the canvas shows vertex coordinate labels.

    Args:
        canvas: Canvas instance to inspect.

    Returns:
        bool: True if at least one sketch has ``show_vertex_coords``.
    """
    for page in canvas.pages:
        for sketch in page.sketches:
            if getattr(sketch, "show_vertex_coords", False):
                return True
    return False


def normalize_canvas_border(border) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, right, top)`` border values.

    Args:
        border: Scalar border, 4-tuple, or ``None`` (uses defaults).

    Returns:
        tuple[float, float, float, float]: Normalized border sides.
    """
    if border is None:
        border = defaults["border"]
    if isinstance(border, (int, float)):
        return (border, border, border, border)
    if isinstance(border, (list, tuple, np.ndarray)) and len(border) == 4:
        return tuple(border)
    raise ValueError(
        "Canvas.border must be a positive numeric value or a tuple of 4 "
        "positive numeric values."
    )


def effective_border_for_export(canvas) -> tuple[float, float, float, float]:
    """Return export border, optionally expanded for vertex labels.

    Args:
        canvas: Canvas instance whose border is used.

    Returns:
        tuple[float, float, float, float]: ``(left, bottom, right, top)``.
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


def warn_vertex_coord_label_sizing(canvas) -> None:
    """Warn once per export about vertex label sizing behavior.

    Args:
        canvas: Canvas instance being exported.
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
            "vertices_canvas_expand or canvas.border if labels are still clipped."
        )
    else:
        issue_warning(
            "Vertex coordinate labels are not included in canvas size. "
            "Increase canvas.border (e.g. 40) if labels are clipped."
        )
    canvas._vertex_label_sizing_warned = True


class Canvas:
    """Main drawing surface for shapes, text, and pages.

    All drawing operations go through ``Canvas``. It can draw graphics and
    text objects and provides helpers for lines, circles, polygons, and more.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.draw(sg.Circle((0, 0), 20))
        >>> canvas.save('out.pdf')
    """

    def __init__(
        self,
        back_color: Color | None = None,
        border: float | None = None,
        page_size: VecType | None = None,
        page_origin: PointType | None = (0, 0),
        **kwargs,
    ):
        """Create a canvas with optional background, border, and page size.

        If you use ``canvas = sg.Canvas()``, default settings are applied.

        Args:
            back_color: Background color of the canvas.
            border: Border width applied to all margins.
            page_size: Page size with ``page_origin`` at ``(0, 0)``.
                Calculated automatically unless specified.
            page_origin: Origin of the page coordinate system.
            **kwargs: Style and positioning options such as ``fill``,
                ``line_width``, ``line_color``, and ``fill_color``.

        Returns:
            A ``Canvas`` instance.
        """
        validate_args(kwargs, canvas_args)
        _set_Nones(self, ["back_color", "border"], [back_color, border])
        self._size = page_size
        self._origin = [0, 0]
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
        self.stack = []

    def __setattr__(self, name, value):
        """Set canvas attributes with special handling for layout properties.

        Args:
            name: Attribute name.
            value: Attribute value.

        Raises:
            ValueError: If ``border``, ``margins``, ``book_margins``, ``pos``,
                or ``angle`` is invalid.
        """
        if name == "back_color":
            if hasattr(self, "active_page"):
                self.active_page.__setattr__(name, value)
            self.__dict__[name] = value
        elif name == "border":
            if value is None:
                border = None
            elif isinstance(value, (int, float)):
                if value < 0:
                    raise ValueError(
                        "Canvas.border must be a positive numeric value."
                    )
                border = value
            elif (
                isinstance(value, (list, tuple, np.ndarray)) and len(value) == 4
            ):
                border = tuple(value)
                if not all(isinstance(item, (int, float)) for item in border):
                    raise ValueError(
                        "Canvas.border must be a positive numeric value or a tuple of 4 positive numeric values."
                    )
                if any(item < 0 for item in border):
                    raise ValueError(
                        "Canvas.border must be a positive numeric value or a tuple of 4 positive numeric values."
                    )
            else:
                raise ValueError(
                    "Canvas.border must be a positive numeric value or a tuple of 4 positive numeric values."
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
        elif name in ["page_size", "page_origin", "limits"]:
            if name == "page_size":
                type(self).page_size.fset(self, value)
            elif name == "page_origin":
                type(self).page_origin.fset(self, value)
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

    def push_matrix(self):
        """Push the current transform matrix onto the stack."""
        self.stack.append(self._xform_matrix)

    def pop_matrix(self):
        """Pop the transform matrix from the stack.

        Warns if the stack is empty.
        """
        if self.stack:
            self._xform_matrix = self.stack.pop()
        else:
            issue_warning("Trying to pop from an empty stack!")

    def apply_mask(self, target, mask):
        """Apply a mask to a drawable target and append a masked sketch.

        Args:
            target: Shape or Group to mask.
            mask: Mask object applied to the target.

        Returns:
            Self: The canvas object.
        """
        sketches = []
        if target.type == Types.GROUP:
            for item in target:
                sketches.append(draw.get_sketches(item, self))
        else:
            sketches.append(draw.get_sketches(target, self))

        self.active_page.sketches.append(
            MaskedSketch(sketches=sketches, mask=mask)
        )
        self._all_vertices.extend(mask.shape.b_box.corners)

        return self

    def clip(self, target, clipper, **kwargs):
        """Clip a drawable target with a clipper shape.

        Args:
            target: Drawable content to clip.
            clipper: Shape used as the clipping path.
            **kwargs: Style overrides for the clipped sketch.

        Returns:
            Self: The canvas object.
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

    def apply_filter(self, target, filters):
        """Apply filters to a drawable target.

        Note:
            Currently a no-op placeholder that returns ``self``.

        Args:
            target: Drawable content to filter.
            filters: Filter specification.

        Returns:
            Self: The canvas object.
        """
        # createa FilteredSketch

        return self

    def display(self) -> Self:
        """Show the canvas in a notebook cell."""
        display(self)

    @property
    def page_size(self) -> VecType:
        """
        The size of the page rectangle.

        Returns:
            VecType: The size of the page rectangle.
        """
        return self._size

    @page_size.setter
    def page_size(self, value: VecType) -> None:
        """
        Set the size of the page rectangle.

        Args:
            value (VecType): The size of the page rectangle.
        """
        if len(value) == 2:
            self._size = value
            x, y = self.page_origin[:2]
            w, h = value
            self._limits = (x, y, x + w, y + h)
        else:
            raise ValueError("page_size must be a tuple of 2 values.")

    @property
    def page_origin(self) -> VecType:
        """
        The lower-left corner of the page rectangle.

        Returns:
            VecType: The lower-left corner of the page rectangle.
        """
        return self._origin[:2]

    @page_origin.setter
    def page_origin(self, value: VecType) -> None:
        """
        Set the lower-left corner of the page rectangle.

        Args:
            value (VecType): The lower-left corner of the page rectangle.
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
        """
        if len(value) == 4:
            x1, y1, x2, y2 = value
            self._size = (x2 - x1, y2 - y1)
            self._origin = (x1, y1)
        else:
            raise ValueError("Limits must be a tuple of 4 values.")

    def b_box(self):
        """Return the axis-aligned bounding box of drawn content.

        Returns:
            Bounding box of all recorded vertices in canvas space.
        """
        xform = np.linalg.inv(self._xform_matrix)
        return bounding_box(homogenize(self._all_vertices) @ xform)

    def capture(self, **kwargs) -> Image:
        """
        Create an image from the canvas.

        Returns:
            Image: The Image object.
        """
        tmpdirname = tempfile.TemporaryDirectory().name
        os.makedirs(tmpdirname, exist_ok=True)
        file_name = next(tempfile._get_candidate_names())
        filepath = os.path.join(tmpdirname, file_name + ".ps")

        self.save(filepath, show=False, print_output=False, remove_aux=False)
        wait_for_file_availability(filepath, timeout=5)
        temp_img = create_image_from_data(filepath)

        time.sleep(1)
        shutil.rmtree(tmpdirname, ignore_errors=True)
        return temp_img

    def insert_code(self, code, loc: TexLoc = TexLoc.PICTURE) -> Self:
        """
        Insert code into the canvas.

        Args:
            code (str): The code to insert.
            loc (TexLoc): The location to insert the code.

        Returns:
            Self: The canvas object.
        """
        draw.insert_code(self, code, loc)
        return self

    def arc(
        self,
        center: PointType,
        radius_x: float,
        radius_y: float | None = None,
        start_angle: float = 0,
        span_angle: float = pi / 2,
        rot_angle: float = 0,
        **kwargs,
    ) -> Self:
        """
        Draw an arc with the given center, radius, start angle and end angle.

        Args:
            center (PointType): The center of the arc.
            radius_x (float): The radius of the arc.
            radius_y (float, optional): The second radius of the arc, defaults to None.
            start_angle (float): The start angle of the arc.
            end_angle (float): The end angle of the arc.
            rot_angle (float, optional): The rotation angle of the arc, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        if radius_y is None:
            radius_y = radius_x
        draw.arc(
            self,
            center,
            radius_x,
            radius_y,
            start_angle,
            span_angle,
            rot_angle,
            **kwargs,
        )
        return self

    def bezier(self, control_points: Sequence[PointType], **kwargs) -> Self:
        """
        Draw a bezier curve.

        Args:
            control_points (Sequence[PointType]): The control points of the bezier curve.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.bezier(self, control_points, **kwargs)
        return self

    def circle(
        self, radius: float, center: PointType = (0, 0), **kwargs
    ) -> Self:
        """
        Draw a circle with the given center and radius.

        Args:
            center (PointType): The center of the circle.
            radius (float): The radius of the circle.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.circle(self, radius, center, **kwargs)
        return self

    def ellipse(
        self,
        center: PointType,
        width: float,
        height: float,
        angle: float = 0,
        **kwargs,
    ) -> Self:
        """
        Draw an ellipse with the given center and radius.

        Args:
            center (PointType): The center of the ellipse.
            width (float): The width of the ellipse.
            height (float): The height of the ellipse.
            angle (float, optional): The angle of the ellipse, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.ellipse(self, center, width, height, angle, **kwargs)

        return self

    def draw_fragments(self, lace=None, palette=None, **kwargs):
        """Draw lace fragment regions, optionally colored by a palette.

        Args:
            lace (optional): Lace object whose fragments are drawn.
            palette (optional): Color palette applied to fragments.
            **kwargs: Style overrides forwarded to the draw helper.

        Returns:
            Self: The canvas object.
        """
        draw.draw_fragments(self, lace, palette, **kwargs)

        return self

    def draw_plaits(self, lace=None, **kwargs):
        """Draw lace plaits.

        Args:
            lace (optional): Lace object whose plaits are drawn.
            **kwargs: Style overrides forwarded to the draw helper.

        Returns:
            Self: The canvas object.
        """
        draw.draw_plaits(self, lace, **kwargs)

        return self

    def draw_lace_with_fillets(self, lace, **kwargs):
        """Draw a lace with filleted plait geometry.

        Args:
            lace: Lace object to draw.
            **kwargs: Style overrides forwarded to the draw helper.

        Returns:
            Self: The canvas object.
        """
        draw.draw_lace_with_fillets(self, lace, **kwargs)

        return self

    def text(
        self,
        text: str,
        pos: PointType,
        font_family: str | None = None,
        font_size: int | None = None,
        font_color: Color = None,
        anchor: Anchor = None,
        align: Align = None,
        **kwargs,
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

    def help_lines(
        self,
        pos: tuple[float, float] | None = None,
        width: float | None = None,
        height: float | None = None,
        spacing=None,
        cs_size: float | None = None,
        deferred: bool = True,
        **kwargs,
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
        **kwargs,
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
        """
        draw.grid(self, pos, width, height, spacing, **kwargs)
        return self

    def line(self, start: PointType, end: PointType, **kwargs) -> Self:
        """
        Draw a line from start to end.

        Args:
            start (PointType): The starting point of the line.
            end (PointType): The ending point of the line.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.line(self, start, end, **kwargs)
        return self

    def rectangle(
        self,
        center: PointType = (0, 0),
        width: float = 100,
        height: float = 100,
        angle: float = 0,
        **kwargs,
    ) -> Self:
        """
        Draw a rectangle.

        Args:
            center (PointType): The center of the rectangle.
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.
            angle (float, optional): The angle of the rectangle, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.rectangle(self, center, width, height, angle, **kwargs)
        return self

    def rectangle2(
        self,
        corner1: PointType = (0, 0),
        corner2: PointType = (0, 0),
        angle: float = 0,
        **kwargs,
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
        """
        x1, y1 = corner1
        x2, y2 = corner2
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        draw.rectangle(self, center, width, height, angle, **kwargs)
        return self

    def rectangle3(
        self,
        upper_left: PointType,
        width: float = 100,
        height: float = 100,
        angle: float = 0,
        **kwargs,
    ) -> Self:
        """
        Draw a rectangle.

        Args:
            upper_left (PointType): The upper_left corner of the rectangle.
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.
            angle (float, optional): The angle of the rectangle, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        x1, y1 = upper_left[:2]
        x2, y2 = x1 + width, y1 - height
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        draw.rectangle(self, center, width, height, angle, **kwargs)
        return self

    def square(
        self,
        center: PointType = (0, 0),
        size: float = 100,
        angle: float = 0,
        **kwargs,
    ) -> Self:
        """
        Draw a square with the given center and size.

        Args:
            center (PointType): The center of the square.
            size (float): The size of the square.
            angle (float, optional): The angle of the square, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.rectangle(self, center, size, size, angle, **kwargs)
        return self

    def lines(self, points: Sequence[PointType], **kwargs) -> Self:
        """
        Draw a polyline through the given points.

        Args:
            points (Sequence[PointType]): The points to draw the polyline through.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.lines(self, points, **kwargs)
        return self

    def draw_lace(self, lace: Group, **kwargs) -> Self:
        """
        Draw the lace.

        Args:
            lace (Group): The lace to draw.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.draw_lace(self, lace, **kwargs)
        return self

    def draw_dimension(self, dim: Shape, **kwargs) -> Self:
        """
        Draw the dimension.

        Args:
            dim (Shape): The dimension to draw.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.draw_dimension(self, dim, **kwargs)
        return self

    def draw_widget(self, item: Drawable, **kwargs) -> Self:
        """Draw an item by expanding ``item.draw_list`` into a composite sketch.

        Args:
            item (Drawable): Widget-like drawable with a ``draw_list``.
            **kwargs: Style overrides forwarded to the draw helper.

        Returns:
            Self: The canvas object.
        """
        draw.draw_widget(self, item, **kwargs)
        return self

    def begin_style(self, style: str):
        """Begin a TikZ scope that appends ``style`` to every path.

        Args:
            style (str): TikZ style fragment inserted into the scope options.

        Returns:
            Self: The canvas object.
        """
        # code = rf'\begin{{scope}}[every path/.append style={{dashed, draw=green}}]'
        code = rf"\begin{{scope}}[every path/.append style={{ {style} }}]"
        code += "\n"
        sketch = TexSketch(code)
        self.active_page.sketches.append(sketch)

        return self

    def end_style(self):
        """End the TikZ style scope started by ``begin_style``.

        Returns:
            Self: The canvas object.
        """
        return self._end_scope()

    def _end_scope(self):
        sketch = TexSketch("\\end{scope}\n")
        self.active_page.sketches.append(sketch)

        return self

    def draw(
        self,
        item_s: Shape | Group | Sequence,
        pos: PointType = None,
        angle: float = 0,
        rotocenter: PointType = (0, 0),
        scale=(1, 1),
        about=(0, 0),
        show: bool = False,
        **kwargs,
    ) -> Self:
        """
        Draw the item_s. item_s can be a single item or a list of items.

        Args:
            item_s (Group | Shape | Sequence): The item(s) to draw.
            pos (PointType, optional): The position to draw the item(s), defaults to None.
            angle (float, optional): The angle to rotate the item(s), defaults to 0.
            rotocenter (PointType, optional): The point about which to rotate, defaults to (0, 0).
            scale (tuple, optional): The scale factors for the x and y axes, defaults to (1, 1).
            about (tuple, optional): The point about which to scale, defaults to (0, 0).
            show (bool, optional): If True, draws the canvas in a Jupyter cell.
            filter (SVG_Filter, optional): SVG filter object to apply to the drawn item(s).
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        warn_unknown_kwargs(
            kwargs,
            get_draw_valid_kwargs(),
            context="canvas.draw",
            stacklevel=3,
        )
        sketch_xform = self._sketch_xform_matrix

        if pos is not None:
            sketch_xform = translation_matrix(*pos[:2]) @ sketch_xform
        if scale[0] != 1 or scale[1] != 1:
            if pos is None:
                pos = (0, 0)
            sketch_xform = (
                scale_in_place_matrix(*scale[:2], about) @ sketch_xform
            )
        if angle != 0:
            sketch_xform = rotation_matrix(angle, rotocenter) @ sketch_xform
        # self._sketch_xform_matrix = sketch_xform @ self._xform_matrix
        self._sketch_xform_matrix = self._xform_matrix @ sketch_xform

        if isinstance(item_s, (list, tuple)):
            for item in item_s:
                draw.draw(self, item, **kwargs)
        else:
            draw.draw(self, item_s, **kwargs)

        self._sketch_xform_matrix = identity_matrix()
        if show:
            self.display()
        else:
            return self

    def draw_lines(
        self, lines: Sequence[tuple[float, float]], **kwargs
    ) -> Self:
        """These lines are drawn with the same style."""
        draw.draw_lines(self, lines, **kwargs)

        return self

    def draw_CS(self, size: float | None = None, **kwargs) -> Self:
        """
        Draw the Canvas coordinate system.

        Args:
            size (float, optional): The size of the coordinate system, defaults to None.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.draw_CS(self, size, **kwargs)
        return self

    def draw_pdf(
        self, pdf, pos: PointType, size=None, scale=None, angle=0, **kwargs
    ) -> Self:
        """
        Draw a PDF on the canvas.

        Args:
            pdf (PDF): The PDF object to draw or file path.
            pos (PointType): Upper-left position to draw the PDF at.

        Returns:
            Self: The canvas object.
        """
        draw.draw_pdf(self, pdf, pos, size, scale, angle, **kwargs)
        return self

    def draw_image(self, image: Image, pos: PointType, **kwargs) -> Self:
        """
        Draw an image on the canvas.

        Args:
            image (Image): The image to draw.
            pos (PointType): The position to draw the image at.

        Returns:
            Self: The canvas object.
        """
        draw.draw_image(self, image, pos, **kwargs)
        return self

    def draw_latex(
        self,
        formula: str,
        pos: PointType,
        font_size: int = 14,
        font_family: str | None = None,
        font_color=None,
        bold: bool = False,
        anchor=None,
        **kwargs,
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
        """
        return "Canvas()"

    def __repr__(self) -> str:
        """
        Return a string representation of the canvas.

        Returns:
            str: The string representation of the canvas.
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
        """

        return self._xform_matrix[2, :2].tolist()[:2]

    @pos.setter
    def pos(self, point: PointType) -> None:
        """
        Set the position of the canvas.

        Args:
            point (PointType): The point to set the position to.
        """
        self._xform_matrix[2, :2] = point[:2]

    @property
    def angle(self) -> float:
        """
        The angle of the canvas.

        Returns:
            float: The angle of the canvas.
        """
        xform = self._xform_matrix

        return np.arctan2(xform[0, 1], xform[0, 0])

    @angle.setter
    def angle(self, angle: float) -> None:
        """
        Set the angle of the canvas.

        Args:
            angle (float): The angle to set the canvas to.
        """
        self._xform_matrix = rotation_matrix(angle) @ self._xform_matrix

    @property
    def scale_xy(self) -> VecType:
        """
        The scale of the canvas.

        Returns:
            VecType: The scale of the canvas.
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
        """
        if scale_y is None:
            scale_y = scale_x

        self._xform_matrix = self._xform_matrix @ scale_in_place_matrix(
            scale_x, scale_y, about=about
        )

    @property
    def xform_matrix(self) -> "np.ndarray":
        """
        The transformation matrix of the canvas.

        Returns:
            np.ndarray: The transformation matrix of the canvas.
        """
        return self._xform_matrix.copy()

    def transform(self, transform_matrix: "np.ndarray") -> Self:
        """
        Transforms the canvas by the given transformation matrix.

        Args:
            transform_matrix (np.ndarray): The transformation matrix.

        Returns:
            Self: The Canvas object.
        """
        self._xform_matrix = transform_matrix @ self._xform_matrix

        return self

    def reset_transform(self) -> Self:
        """
        Reset the transformation matrix of the canvas.
        The canvas origin is at (0, 0) and the orientation angle is 0.
        Transformation matrix is the identity matrix.

        Returns:
            Self: The canvas object.
        """
        self._xform_matrix = identity_matrix()

        return self

    def translate(self, dx: float, dy: float) -> Self:
        """
        Translate the canvas by dx and dy.

        Args:
            dx (float): The translation distance along the x-axis.
            dy (float): The translation distance along the y-axis.

        Returns:
            Self: The canvas object.
        """

        self._xform_matrix = translation_matrix(dx, dy) @ self._xform_matrix

        return self

    def rotate(self, angle: float, about=(0, 0)) -> Self:
        """
        Rotate the canvas by angle in radians about the given point.

        Args:
            angle (float): The rotation angle in radians.
            about (tuple): The point about which to rotate the canvas.

        Returns:
            Self: The canvas object.
        """

        self._xform_matrix = rotation_matrix(angle, about) @ self._xform_matrix

        return self

    def scale(
        self,
        scale_x: float,
        scale_y: float | None = None,
        about: PointType = (0, 0),
    ) -> Self:
        """
        Scale the canvas by scale_x and scale_y about the given point.
        If scale_y is not given then scale_y = scale_x.

        Args:
            scale_x (float): The scale factor in x direction.
            scale_y (float): The scale factor in y direction.

        Returns:
            Self: The canvas object.
        """
        if scale_y is None:
            scale_y = scale_x
        self._xform_matrix = (
            scale_in_place_matrix(scale_x, scale_y, about) @ self._xform_matrix
        )

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
        """
        issue_warning(
            "Flipping the x-axis will change the positive rotation direction."
        )
        return self._flip(Axis.X)

    def flip_y_axis(self) -> Self:
        """
        Flip the y-axis direction.

        Returns:
            Self: The canvas object.
        """
        issue_warning(
            "Flipping the y-axis will reverse the positive rotation direction."
        )

        return self._flip(Axis.Y)

    @property
    def x(self) -> float:
        """
        The x coordinate of the canvas origin.

        Returns:
            float: The x coordinate of the canvas origin.
        """
        return self.pos[0]

    @x.setter
    def x(self, value: float) -> None:
        """
        Set the x coordinate of the canvas origin.

        Args:
            value (float): The x coordinate to set.
        """
        self.pos = [value, self.pos[1]]

    @property
    def y(self) -> float:
        """
        The y coordinate of the canvas origin.

        Returns:
            float: The y coordinate of the canvas origin.
        """
        return self.pos[1]

    @y.setter
    def y(self, value: float) -> None:
        """
        Set the y coordinate of the canvas origin.

        Args:
            value (float): The y coordinate to set.
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
        """

        def add_group(group, graph):
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
        """
        Handles None values for properties.
        try item.property_name first,
        then use the default value.

        Args:
            item (Drawable): The item to resolve the property for.
            property_name (str): The name of the property to resolve.

        Returns:
            Any: The resolved property value.
        """
        value = getattr(item, property_name, None)
        if value is None:
            value = defaults.get(property_name, VOID)
            if value == VOID and property_name not in ("color", "alpha"):
                issue_warning(f"Property {property_name} is not in defaults.")
                value = None
        return value

    def resolve_style_properties(
        self, item: Drawable, style_map, **draw_kwargs
    ) -> dict[str, Any]:
        """Resolve style values for sketch creation in one place.

        1. Handle color and alpha
        2. Handle kwargs
        """
        d_resolved = {}
        resolved = []
        # handle color
        color = None
        if "color" in draw_kwargs:
            draw_color = draw_kwargs["color"]
            if check_color(draw_color):
                color = draw_color
            else:
                raise ValueError(f"Invalid color value: {draw_color}")
        else:
            item_color = getattr(item, "color", None)
            if item_color is not None:
                if check_color(item_color):
                    color = item_color
                else:
                    raise ValueError(f"Invalid color value: {draw_color}")

        if color is not None:
            d_resolved["line_color"] = color
            d_resolved["fill_color"] = color
            resolved.extend(["color", "line_color", "fill_color"])

        # handle alpha
        alpha = None
        if "alpha" in draw_kwargs:
            draw_alpha = draw_kwargs["alpha"]
            if check_alpha(draw_alpha):
                alpha = draw_alpha
            else:
                raise ValueError(f"Invalid alpha value: {draw_alpha}")
        else:
            item_alpha = getattr(item, "alpha", None)
            if item_alpha is not None:
                if check_alpha(item_alpha):
                    alpha = item_alpha
                else:
                    raise ValueError(f"Invalid alpha value: {draw_alpha}")

        if alpha is not None:
            d_resolved["line_alpha"] = alpha
            d_resolved["fill_alpha"] = alpha
            resolved.extend(["alpha", "line_alpha", "fill_alpha"])

        for attrib_name in style_map:
            if attrib_name in draw_kwargs:
                d_resolved[attrib_name] = draw_kwargs[attrib_name]
            elif attrib_name in resolved:
                continue
            else:
                d_resolved[attrib_name] = self.resolve_property(
                    item, attrib_name
                )
            resolved.append(attrib_name)

        # for attrib_name in style_map:
        #     if attrib_name in resolved:
        #         continue
        #     # we need to validate input here!!!!
        #     if attrib_name in draw_kwargs:
        #         d_resolved[attrib_name] = draw_kwargs[attrib_name]
        #     else:
        #         d_resolved[attrib_name] = self.resolve_property(
        #             item, attrib_name
        #         )
        #     resolved.append(attrib_name)

        return d_resolved

    def draw_all_segments(
        self, item: Shape | Group, vert_indices=False, **kwargs
    ) -> Self:
        """
        Using intersections, splits edges of the item into separate segments and
        draws them with their indices. This is usually used for the "get_loop"
        function.

        Args:
            item: A shape or a group.
            vert_indices: If True, vertex indices are shown.
                          Default is False, edge indices are shown.
        Returns:
            The canvas object.
        """

        return draw.draw_all_segments(self, item, vert_indices, **kwargs)

    def get_fonts_list(self) -> list[str]:
        """
        Get the list of fonts used in the canvas.

        Returns:
            list[str]: The list of fonts used in the canvas.
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

    def set_page_size(self, width, height):
        """Set the active page size.

        Args:
            width: Page width.
            height: Page height.
        """
        self.page_size = (width, height)

    def _calculate_size(self, border=None, b_box=None) -> tuple[float, float]:
        """
        Calculate the size of the canvas based on the bounding box and border.

        Args:
            border (float, optional): The border of the canvas, defaults to None.
            b_box (Any, optional): The bounding box of the canvas, defaults to None.

        Returns:
            tuple[float, float]: The size of the canvas.
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
                    "Canvas.border must be a positive numeric value or a tuple of 4 positive numeric values."
                )
            w = b_box.width + border_left + border_right
            h = b_box.height + border_bottom + border_top
            offset_x, offset_y = b_box.southwest
            res = w, h, offset_x - border_left, offset_y - border_bottom
        else:
            res = None
        return res

    def _sketch_bbox(self, sketch):
        """Return axis-aligned bbox tuple (xmin, ymin, xmax, ymax) for a sketch."""
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
                        f"bbox={sketch_bbox}, limits={page_limits}."
                    )

    def _show_browser(
        self, filepath: Path, show_browser: bool, multi_page_svg: bool
    ) -> None:
        """
        Show the file in the browser.

        Args:
            filepath (Path): The path to the file.
            show_browser (bool): Whether to show the file in the browser.
            multi_page_svg (bool): Whether the file is a multi-page SVG.
        """
        if show_browser is None:
            show_browser = defaults["show_browser"]
        if show_browser:
            filepath = "file:///" + filepath
            if multi_page_svg:
                root, extension = os.path.splitext(filepath)
                for i, _ in enumerate(self.pages):
                    f_path = f"{root}_{i + 1}{extension}"
                    webbrowser.open(f_path)
            else:
                webbrowser.open(filepath)

    def save(
        self,
        filepath: Path,
        overwrite: bool | None = None,
        show: bool | None = None,
        print_output=False,
        remove_aux=True,
        inset=None,
        display=False,
    ) -> Self:
        """
        Save the canvas to a file.

        Args:
            filepath (Path, optional): The path to save the file.
            overwrite (bool, optional): Whether to overwrite the file if it exists.
            show (bool, optional): Whether to show the file in the browser.
            inset (float, optional): The inset value will be clipped from all sides, defaults to None.
        Returns:
            Self: The canvas object.
        """

        if inset is not None:
            self.inset = inset

        self._vertex_label_sizing_warned = False
        self._warn_sketches_outside_page()

        try:
            parent_dir, file_name, extension = validate_filepath(
                filepath, overwrite
            )
        except NotADirectoryError:
            parent_dir = Path(sys.path[0])
            filepath = str(parent_dir / filepath)
            parent_dir, file_name, extension = validate_filepath(
                filepath, overwrite
            )

            issue_warning(f"Unspecified filepath, using {filepath}.")

        renderer = _save_renderer(extension)
        multi_page_svg = False
        if renderer == Renderer.SVG:
            from simetri.svg.svg import get_svg_code

            if extension == ".png":
                svg_code = get_svg_code(self)
                save_svg_png(svg_code, filepath)
            else:
                if len(self.pages) > 1:
                    multi_page_svg = True
                    active_page = self.active_page
                    for i, page in enumerate(self.pages):
                        self.active_page = page
                        page_filepath = os.path.join(
                            parent_dir, f"{file_name}_{i + 1}{extension}"
                        )
                        validate_filepath(page_filepath, overwrite)
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

    def new_page(self, **kwargs) -> Self:
        """
        Create a new page and add it to the canvas.pages.

        Args:
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
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
    """

    spacing: float = None
    back_color: "Color" = None
    line_color: "Color" = None
    line_width: float = None
    line_dash_array: Sequence[float] = None
    x_shift: float = None
    y_shift: float = None

    def __post_init__(self):
        """Initialize page-grid defaults from settings."""
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
    """

    size: VecType = None
    back_color: "Color" = None
    border: Any = None
    mask: Any = None
    margins: Any = None  # left, bottom, right, top
    book_margins: Any = None  # gutter, footer, margin, header
    recto: bool = True  # True if page is recto, False if verso
    grid: PageGrid = None
    kwargs: dict = None

    def __post_init__(self):
        """Initialize page metadata and an empty sketch list."""
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
