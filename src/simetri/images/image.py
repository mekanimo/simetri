"""Raster image and PDF drawable objects for Simetri canvases.

Wraps Pillow images as canvas items and provides helpers for opening
and constructing ``Image`` / ``PDF`` objects.

Examples:
    >>> import simetri.graphics as sg
    >>> sg.Image(size=(1, 1), mode="RGB").height
    1
"""

from __future__ import annotations

import io
import os
from collections.abc import Callable, Iterable, Sequence
from math import degrees
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray
from PIL import Image as PIL_Image
from PIL import ImageDraw, ImageFilter

if TYPE_CHECKING:
    from ..render.sketch import Sketch

from ..base.all_enums import (
    Anchor,
    ImageMode,
    InPlace,
    TransformationType,
    Types,
    get_enum_value,
)
from ..base.common import PointType
from ..base.core import _update_inplace
from ..coloring.colors import ColorLike, check_color
from ..geom.affine import (
    rotation_matrix,
    scale_in_place_matrix,
)
from ..geom.matrices import identity_matrix
from ..group.batch import Group
from ..helpers.utilities import decompose_transformations
from ..shapes.geom_items import Rectangle


class PDF(Rectangle):
    """Drawable placeholder for a PDF file on the canvas.

    Examples:
        >>> import os
        >>> import tempfile
        >>> import simetri.graphics as sg
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = os.path.join(tmp, "sample.pdf")
        ...     with open(path, "wb") as handle:
        ...         _ = handle.write(b"%PDF-1.0\\n%%EOF\\n")
        ...     isinstance(PDF(path), PDF)
        True
    """

    def __init__(
        self,
        pdf_path: str,
        pos: PointType = (0, 0),
        size: Sequence[int] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize a PDF drawable.

        Args:
            pdf_path: Path to the PDF file.
            pos: Placement anchor point on the canvas.
            size: Width and height when known; defaults to ``(100, 100)``.
            **kwargs: Forwarded to :class:`~simetri.shapes.geom_items.Rectangle`.

        Raises:
            FileNotFoundError: If ``pdf_path`` does not exist.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import simetri.graphics as sg
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     path = os.path.join(tmp, "sample.pdf")
            ...     with open(path, "wb") as handle:
            ...         _ = handle.write(b"%PDF-1.0\\n%%EOF\\n")
            ...     pdf = PDF(path)
            ...     pdf.pdf_path == path
            True
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File {pdf_path} not found.")
        self.pdf_path = pdf_path
        self.type = Types.PDF_SKETCH
        self.subtype = Types.PDF_SKETCH
        self.anchor = kwargs.get("anchor", Anchor.CENTER)
        if "xform_matrix" in kwargs:
            self.xform_matrix = kwargs["xform_matrix"]
        else:
            self.xform_matrix = identity_matrix()

        # For simplicity, we will not load the actual PDF content here.
        # In a full implementation, you might want to use a library like PyMuPDF or pdf2image
        # to render the PDF pages into images.

        # Placeholder width and height; in a real implementation, extract from PDF metadata
        width, height = (100, 100) if size is None else size
        kwargs["fill"] = False
        kwargs["stroke"] = False
        super().__init__(width, height, center=pos, **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the PDF object.

        Returns:
            str: A string representation of the PDF object.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import simetri.graphics as sg
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     path = os.path.join(tmp, "sample.pdf")
            ...     with open(path, "wb") as handle:
            ...         _ = handle.write(b"%PDF-1.0\\n%%EOF\\n")
            ...     repr(PDF(path)).startswith("PDF(")
            True
        """
        return f"PDF({self.pdf_path})"

    def __str__(self) -> str:
        """Return a human-readable string representation of the PDF object.

        Returns:
            str: A human-readable string representation of the PDF object.

        Examples:
            >>> import os
            >>> import tempfile
            >>> import simetri.graphics as sg
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     path = os.path.join(tmp, "sample.pdf")
            ...     with open(path, "wb") as handle:
            ...         _ = handle.write(b"%PDF-1.0\\n%%EOF\\n")
            ...     str(PDF(path)).startswith("PDF file at")
            True
        """
        return f"PDF file at {self.pdf_path}"


class Image(Rectangle):
    """Simetri drawable backed by a Pillow image (``pil_img``).

    Pillow method names are available via attribute delegation. See the
    `Pillow docs <https://pillow.readthedocs.io/en/stable/>`_ for raster ops.

    Examples:
        >>> import simetri.graphics as sg
        >>> isinstance(sg.Image(size=(2, 2), mode="RGB"), sg.Image)
        True
    """

    def __init__(
        self,
        img: str | PIL_Image.Image | None = None,
        pos: PointType = (0, 0),
        size: Sequence[int] | None = None,
        mode: ImageMode | str = ImageMode.RGB,
        **kwargs: object,
    ) -> None:
        """Initialize an image drawable.

        Args:
            img: File path, Pillow image, or ``None`` to create a blank image.
            pos: Canvas point for the image ``anchor``.
            size: Required when ``img`` is ``None`` (new image dimensions).
            mode: Pillow mode when creating a blank image.
            **kwargs: Forwarded to :class:`~simetri.shapes.geom_items.Rectangle`
                and Pillow ``Image.new`` when applicable.
                ``anchor`` chooses which point of the image sits on ``pos``.

        Raises:
            FileNotFoundError: If ``img`` is a path that does not exist.
            TypeError: If ``img`` is not a path, Pillow image, or ``None``.

        Examples:
            >>> import simetri.graphics as sg
            >>> im = sg.Image(size=(10, 10), mode="RGB")
            >>> im.width
            10
            >>> im.mode
            'RGB'
            >>> im = sg.Image(
            ...     size=(40, 20),
            ...     mode="RGB",
            ...     pos=(20, 60),
            ...     anchor=sg.Anchor.SOUTHWEST,
            ... )
            >>> im.anchor == sg.Anchor.SOUTHWEST
            True
            >>> im.pos[0], im.pos[1]
            (20.0, 60.0)
            >>> im.southwest[0], im.southwest[1]
            (20.0, 60.0)
        """
        anchor = kwargs.pop("anchor", Anchor.CENTER)
        file_path = None
        if img is None:
            img = PIL_Image.new(mode=mode, size=size, **kwargs)
            width, height = size
        elif isinstance(img, str):
            if os.path.exists(img):
                file_path = img
                img = PIL_Image.open(img)
                width, height = img.size
            else:
                raise FileNotFoundError(f"File {img} not found.")
        elif isinstance(img, PIL_Image.Image):
            width, height = img.size
        elif not isinstance(img, PIL_Image.Image):
            raise TypeError("img must be a PIL Image object or a file path.")
        self.__dict__["pil_img"] = img
        kwargs["fill"] = False
        kwargs["stroke"] = False
        left, bottom, right, top = _anchor_bounds(
            anchor, pos, width, height
        )
        center = ((left + right) / 2, (bottom + top) / 2)
        super().__init__(width, height, center=center, **kwargs)
        self.file_path = file_path
        self.type = Types.IMAGE
        self.subtype = Types.IMAGE
        self.__dict__["anchor"] = anchor
        if "xform_matrix" in kwargs:
            self.xform_matrix = kwargs["xform_matrix"]
        else:
            self.xform_matrix = identity_matrix()

    def __repr__(self) -> str:
        """Return a string representation of the Image object.

        Returns:
            str: A string representation of the Image object.

        Examples:
            >>> import simetri.graphics as sg
            >>> repr(sg.Image(size=(2, 3), mode="RGB"))
            'Image((2, 3), RGB)'
        """
        return f"Image({self.pil_img.size}, {self.pil_img.mode})"

    def __str__(self) -> str:
        """Return a human-readable string representation of the Image object.

        Returns:
            str: A human-readable string representation of the Image object.

        Examples:
            >>> import simetri.graphics as sg
            >>> str(sg.Image(size=(2, 3), mode="RGB"))
            'Image of size (2, 3) and mode RGB'
        """
        return f"Image of size {self.pil_img.size} and mode {self.pil_img.mode}"

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the underlying PIL Image object.

        Args:
            name (str): The name of the attribute to get.

        Returns:
            Any: The value of the requested attribute.

        Examples:
            >>> import simetri.graphics as sg
            >>> im = sg.Image(size=(2, 2), mode="RGB")
            >>> im.size
            (2, 2)
        """
        if name in self.__dict__:
            res = self.__dict__[name]
        else:
            # Check if the attribute exists in the PIL Image object
            if hasattr(self.pil_img, name):
                res = getattr(self.pil_img, name)
            else:
                try:
                    res = super().__getattr__(name)
                except AttributeError:
                    # If the attribute doesn't exist, raise an AttributeError
                    raise AttributeError(
                        f"'{self.__class__.__name__}' object has no attribute '{name}'"
                    )

        return res

    def _update(
        self,
        xform_matrix: "array",
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | NDArray
        | Sequence[Sequence[float]]
        | None = None,
        dyn_ref: bool | None = None,
        merge: bool = False,
        xform_type: TransformationType | None = None,
    ) -> Group | Image:
        """Used internally. Update the shape with a transformation matrix.

        Args:
            xform_matrix (array): The transformation matrix.
            reps (int, optional): The number of repetitions, defaults to 0.
            take: Not supported; must be ``None``.
            incr: Increment applied between repetitions when ``reps > 0``.
            merge: If True and ``reps > 0``, merge the copies.
            xform_type: Transform kind used with ``incr``.

        Returns:
            Group: The updated shape or a group of shapes.

        Raises:
            ValueError: If ``take`` is set, or if ``dyn_ref`` is used.
        """
        if dyn_ref:
            raise ValueError(
                "Image does not support dynamic references. Only Shape and "
                "Group resolve dyn_ref."
            )
        if take is not None:
            raise ValueError(
                "Image._update does not support take=; transform the whole image."
            )
        if reps == 0:
            self.xform_matrix = self.xform_matrix @ xform_matrix
            if "_final_coords" in self.__dict__:
                delattr(self, "_final_coords")
            if "_vertices" in self.__dict__:
                delattr(self, "_vertices")
            return self
        images = [self]
        image = self
        for i in range(reps):
            if incr is not None and i > 0:
                xform_matrix = _update_inplace(xform_matrix, xform_type, incr)
            image = image.copy()
            image._update(xform_matrix)
            images.append(image)
        res = Group(images)
        if merge:
            return res.merge_images()
        return res

    def _drawn_extent(self, pixels: int, axis: int) -> int | float:
        """Drawn length on one axis.

        ``axis`` 0 is width and 1 is height. A scale of 1 returns the
        pixel count unchanged.
        """
        _translation, _rotation, scale = decompose_transformations(
            self.xform_matrix
        )
        factor = float(scale[axis])
        if factor == 1:
            return pixels
        return pixels * factor

    @property
    def anchor(self) -> Anchor:
        """Which point of the image ``pos`` refers to.

        Changing the anchor keeps the current ``pos`` and moves the
        image so the new anchor sits on that point.

        Examples:
            >>> import simetri.graphics as sg
            >>> im = sg.Image(size=(10, 20), mode="RGB", pos=(0, 0))
            >>> im.anchor = sg.Anchor.SOUTHWEST
            >>> im.pos[0], im.pos[1]
            (0.0, 0.0)
            >>> im.southwest[0], im.southwest[1]
            (0.0, 0.0)
        """
        return self.__dict__["anchor"]

    @anchor.setter
    def anchor(self, value: Anchor) -> None:
        """Set the image anchor, keeping ``pos`` fixed."""
        point = self.pos
        self.__dict__["anchor"] = value
        self.move_to(point, anchor=value)

    @property
    def pos(self) -> PointType:
        """Canvas point named by ``anchor``.

        The default anchor is ``Anchor.CENTER``. With
        ``Anchor.SOUTHWEST``, ``pos`` is the lower-left corner.

        Examples:
            >>> import simetri.graphics as sg
            >>> im = sg.Image(size=(4, 4), mode="RGB", pos=(5, 6))
            >>> im.pos[0], im.pos[1]
            (5.0, 6.0)
            >>> im = sg.Image(
            ...     size=(40, 20),
            ...     mode="RGB",
            ...     pos=(20, 60),
            ...     anchor=sg.Anchor.SOUTHWEST,
            ... )
            >>> im.pos[0], im.pos[1]
            (20.0, 60.0)
        """
        name = get_enum_value(Anchor, self.__dict__["anchor"])
        if name == "center":
            name = "midpoint"
        return getattr(self.b_box, name)

    @pos.setter
    def pos(self, point: PointType) -> None:
        """Move the image so ``anchor`` lands on ``point``.

        This is the same move as ``move_to(point, anchor=self.anchor)``.

        Args:
            point (PointType): The new canvas point for ``anchor``.

        Examples:
            >>> import simetri.graphics as sg
            >>> im = sg.Image(size=(4, 4), mode="RGB")
            >>> im.pos = (40, 20)
            >>> im.pos[0], im.pos[1]
            (40.0, 20.0)
            >>> im = sg.Image(
            ...     size=(40, 20),
            ...     mode="RGB",
            ...     anchor=sg.Anchor.SOUTHWEST,
            ... )
            >>> _ = im.scale(0.5)
            >>> im.pos = (30, 40)
            >>> im.pos[0], im.pos[1]
            (30.0, 40.0)
            >>> im.southwest[0], im.southwest[1]
            (30.0, 40.0)
        """
        self.move_to(point, anchor=self.__dict__["anchor"])

    @property
    def pil_img(self) -> PIL_Image.Image:
        """The underlying PIL Image object.

        Returns:
            PIL_Image.Image: The PIL Image object.

        Examples:
            >>> import simetri.graphics as sg
            >>> im = sg.Image(size=(3, 4), mode="RGB")
            >>> im.pil_img.size
            (3, 4)
        """
        return self.__dict__["pil_img"]

    @property
    def filename(self) -> str | None:
        """Filename metadata from Pillow, if set.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(2, 2), mode="RGB").filename is None
            True
        """
        return self.pil_img.info.get("filename", None)

    @property
    def format(self) -> str | None:
        """Pillow format string (e.g. ``JPEG``, ``PNG``), if known.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(2, 2), mode="RGB").format is None
            True
        """
        return self.pil_img.format

    @property
    def mode(self) -> str:
        """The mode of the image.

        Returns:
            str: The mode of the image (e.g., "RGB", "L").

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(80, 80), mode="L").mode
            'L'
        """
        return self.pil_img.mode

    @property
    def size(self) -> tuple[int | float, int | float]:
        """Drawn width and height.

        Unscaled, this is the pixel size. ``scale`` multiplies both axes.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(5, 7), mode="RGB").size
            (5, 7)
            >>> im = sg.Image(size=(10, 20), mode="RGB")
            >>> _ = im.scale(0.6)
            >>> im.size
            (6.0, 12.0)
        """
        return (self.width, self.height)

    @property
    def width(self) -> int | float:
        """Drawn width of the image.

        Unscaled, this is the pixel width. ``scale`` multiplies it.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(5, 7), mode="RGB").width
            5
        """
        return self._drawn_extent(self.pil_img.size[0], 0)

    @property
    def height(self) -> int | float:
        """Drawn height of the image.

        Unscaled, this is the pixel height. ``scale`` multiplies it.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(5, 7), mode="RGB").height
            7
        """
        return self._drawn_extent(self.pil_img.size[1], 1)

    @property
    def info(self) -> dict[str, object]:
        """Pillow image metadata dictionary.

        Examples:
            >>> import simetri.graphics as sg
            >>> isinstance(sg.Image(size=(2, 2), mode="RGB").info, dict)
            True
        """
        return self.pil_img.info

    @property
    def palette(self) -> object | None:
        """Pillow palette for ``P`` mode images, or ``None``.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(2, 2), mode="RGB").palette is None
            True
        """
        return self.pil_img.palette

    @property
    def category(self) -> str:
        """The category of the image.

        Returns:
            str: The category of the image (e.g., "image").

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(2, 2), mode="RGB").category
            'image'
        """
        return "image"

    @property
    def readonly(self) -> bool:
        """Whether the image is read-only.

        Returns:
            bool: True if the image is read-only, False otherwise.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(2, 2), mode="RGB").readonly
            0
        """
        return self.pil_img.readonly

    @property
    def decoderconfig(self) -> tuple[object, ...]:
        """Pillow decoder configuration tuple.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(2, 2), mode="RGB").decoderconfig
            ()
        """
        try:
            return self.pil_img.decoderconfig
        except AttributeError:
            return ()

    @property
    def decodermaxblock(self) -> int:
        """The maximum block size used by the decoder.

        Returns:
            int: The maximum block size in bytes.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(2, 2), mode="RGB").decodermaxblock
            65536
        """
        try:
            return self.pil_img.decodermaxblock
        except AttributeError:
            return 65536

    def alpha_composite(
        self,
        im: Image,
        dest: Sequence[int] = (0, 0),
        source: Sequence[int] = (0, 0),
    ) -> Image:
        """
        Blend two images together using alpha compositing.
        This method is a wrapper around the PIL alpha_composite method.

        Args:
            im (Image): The source image to composite with.
            dest (Sequence[int], optional): The destination coordinates. Defaults to (0, 0).
            source (Sequence[int], optional): The source coordinates. Defaults to (0, 0).

        Returns:
            Image: This image after alpha compositing (in place).

        Examples:
            >>> import simetri.graphics as sg
            >>> base = sg.Image(size=(4, 4), mode="RGBA")
            >>> over = sg.Image(size=(4, 4), mode="RGBA")
            >>> base.alpha_composite(over).size
            (4, 4)
        """
        self.pil_img.alpha_composite(im.pil_img, dest, source)
        return self

    def apply_transparency(self) -> None:
        """Apply transparency to the image.

        This method is a wrapper around the PIL apply_transparency method.

        Examples:
            >>> import simetri.graphics as sg
            >>> im = sg.Image(size=(80, 80), mode="P")
            >>> callable(im.apply_transparency)
            True
        """
        return self.pil_img.apply_transparency()

    def convert(
        self,
        mode: str | None = None,
        matrix: Sequence[float] | None = None,
        dither: int | None = None,
        palette: int = 0,
        colors: int = 256,
    ) -> PIL_Image.Image:
        """Convert the Pillow image to another mode (delegates to ``pil_img``).

        Args:
            mode: Target mode; see Pillow documentation.
            matrix: Optional conversion matrix.
            dither: Dithering method for palette or bilevel conversion.
            palette: Palette selector when converting to ``P``.
            colors: Palette size when converting to ``P``.

        Returns:
            PIL.Image.Image: Converted Pillow image.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(2, 2), mode="RGB").convert("L").mode
            'L'
        """
        return self.pil_img.convert(mode, matrix, dither, palette, colors)

    def copy(self, **kwargs: object) -> Image:
        """Copy this Simetri image and its transform metadata.

        Args:
            **kwargs: Attributes set on the copy after duplication.

        Returns:
            Image: New Simetri image sharing the same pixel data.

        Examples:
            >>> import simetri.graphics as sg
            >>> im = sg.Image(size=(4, 4), mode="RGB")
            >>> copy = im.copy()
            >>> copy.width, copy is not im
            (4, True)
        """
        img = Image(
            pos=self.pos, img=self.pil_img.copy(), anchor=self.anchor
        )
        img.primary_points = self.primary_points.copy()
        img.xform_matrix = self.xform_matrix
        img.file_path = self.file_path

        for k, v in kwargs.items():
            setattr(img, k, v)

        return img

    def crop(
        self, box: tuple[int, int, int, int] | None = None
    ) -> PIL_Image.Image:
        """Crop ``pil_img`` to ``(left, upper, right, lower)``.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(4, 4), mode="RGB").crop((0, 0, 2, 2)).size
            (2, 2)
        """
        return self.pil_img.crop(box)

    def draft(self, mode: str, size: tuple[int, int]) -> None:
        """Configure the loader for a matching mode and size (Pillow ``draft``).

        Examples:
            >>> import simetri.graphics as sg
            >>> im = sg.Image(size=(8, 8), mode="RGB")
            >>> im.draft("RGB", (4, 4)) is None
            True
        """
        self.pil_img.draft(mode, size)

    def effect_spread(self, distance: int) -> PIL_Image.Image:
        """Randomly spread pixels (Pillow ``effect_spread``).

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(4, 4), mode="RGB").effect_spread(1).size
            (4, 4)
        """
        return self.pil_img.effect_spread(distance)

    def filter(self, filter: ImageFilter.Filter) -> PIL_Image.Image:
        """Apply a Pillow filter kernel to ``pil_img``.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(4, 4), mode="RGB").filter(
            ...     ImageFilter.BLUR
            ... ).size
            (4, 4)
        """
        return self.pil_img.filter(filter)

    def getbands(self) -> tuple[str, ...]:
        """Band names for this image (e.g. ``("R", "G", "B")``).

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(2, 2), mode="RGB").getbands()
            ('R', 'G', 'B')
        """
        return self.pil_img.getbands()

    def _getbbox(self) -> tuple[int, int, int, int] | None:
        """Non-zero bounding box of ``pil_img``, or ``None``."""
        return self.pil_img.getbbox()

    def getcolors(
        self, maxcolors: int = 256
    ) -> list[tuple[int, int | tuple[int, ...]]] | None:
        """Color usage counts, or ``None`` if ``maxcolors`` is exceeded.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(80, 80), mode="L").getcolors()
            [(6400, 0)]
        """
        return self.pil_img.getcolors(maxcolors)

    def getdata(self, band: int | None = None) -> Iterable[int]:
        """Pixel access sequence from ``pil_img.getdata``.

        Examples:
            >>> import simetri.graphics as sg
            >>> len(list(sg.Image(size=(80, 80), mode="L").getdata()))
            6400
        """
        return self.pil_img.getdata(band)

    def getextrema(self) -> tuple[tuple[int, int], ...]:
        """Per-band minimum and maximum pixel values.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(80, 80), mode="L").getextrema()
            (0, 0)
        """
        return self.pil_img.getextrema()

    def getpixel(self, xy: tuple[int, int]) -> int | tuple[int, ...]:
        """Pixel value at ``(x, y)``.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(80, 80), mode="L").getpixel((0, 0))
            0
        """
        return self.pil_img.getpixel(xy)

    def histogram(
        self,
        mask: PIL_Image.Image | None = None,
        extrema: Sequence[int] | None = None,
    ) -> list[int]:
        """Histogram of pixel values.

        Examples:
            >>> import simetri.graphics as sg
            >>> len(sg.Image(size=(80, 80), mode="L").histogram())
            256
        """
        return self.pil_img.histogram(mask, extrema)

    def paste(
        self,
        im: PIL_Image.Image | int | tuple[int, ...],
        box: tuple[int, ...] | None = None,
        mask: PIL_Image.Image | None = None,
    ) -> None:
        """Paste ``im`` into ``pil_img`` (mutates pixels).

        Examples:
            >>> import simetri.graphics as sg
            >>> im = sg.Image(size=(2, 2), mode="RGB")
            >>> im.paste((255, 0, 0), (0, 0, 1, 1))
            >>> im.getpixel((0, 0))
            (255, 0, 0)
        """
        self.pil_img.paste(im, box, mask)

    def resize(
        self,
        size: tuple[int, int],
        resample: int | None = None,
        box: tuple[int, int, int, int] | None = None,
        reducing_gap: float | None = None,
    ) -> PIL_Image.Image:
        """Return a resized copy of ``pil_img``.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(4, 4), mode="RGB").resize((2, 3)).size
            (2, 3)
        """
        return self.pil_img.resize(size, resample, box, reducing_gap)

    # def translate(self, dx: float=0, dy: float=0, reps: int=0, merge: bool=False, **kwargs)  -> "Group | Image":
    #     """
    #     Returns a translated copy of this image or a group of translated copies of this image.

    #     Args:
    #         dx (float): The x-coordinate translation.
    #         dy (float): The y-coordinate translation.
    #         reps (int, optional): The number of repetitions.
    #         merge (bool, optional): Whether to merge the images.

    #     Returns:
    #         Image: An Image object or a Group of images.
    #     """
    #     transform = translation_matrix(dx, dy)
    #     kwargs = {'transform': Transformation.TRANSLATE}
    #     return self._update(transform, reps=reps, merge=merge, kwargs=kwargs)

    # def rotate(self, angle: float, about: PointType=None, reps: int=0, merge: bool=False,
    #            resample=0, expand=0, translate=None, fillcolor=None) -> "Group | Image":
    #     """
    #     Returns a rotated copy of this image or a group of rotated Image objects.

    #     Args:
    #         angle (float): The angle to rotate the image.
    #         about (tuple, optional): Optional center of rotation. Origin is the lower left corner.
    #             Default is the center of the image.
    #         resample (int, optional): An optional resampling filter. This can be one of
    #             Resampling.NEAREST (use nearest neighbour), Resampling.BILINEAR
    #             (linear interpolation in a 2x2 environment), or Resampling.BICUBIC
    #             (cubic spline interpolation in a 4x4 environment).
    #             If omitted, or if the image has mode “1” or “P”, it is set to
    #             Resampling.NEAREST. See Filters.
    #         expand (int, optional): Optional expansion flag. If true, expands the output
    #             image to make it large enough to hold the entire rotated image. If false
    #             or omitted, make the output image the same size as the input image. Note
    #             that the expand flag assumes rotation around the center and no translation.
    #         translate (tuple, optional): An optional post-rotate translation.
    #         fillcolor (tuple, optional): Optional fill color for the area outside the rotated image.

    #     Returns:
    #         Image or Group: A group of images or an Image object.
    #     """
    #     if about is None:
    #         width, height = self.pil_img.size
    #         about = (width / 2, height / 2)
    #     transform = rotation_matrix(angle, about=about)
    #     angle = degrees(angle)
    #     x, y = about[:2]
    #     center = int(x), int(y)
    #     kwargs = {'transform': Transformation.ROTATE, 'angle': angle, 'resample': resample,
    #               'expand': expand, 'center': center, 'translate': translate,
    #               'fillcolor': fillcolor}

    #     return self._update(transform, reps=reps, merge=merge, **kwargs)

    # def scale(self, scale_x: float=1, scale_y: float=None, about: PointType=(0, 0),
    #                                 reps: int=0, merge: bool=False) -> "Group | Image":
    #     """
    #     Scales this image by the given scale factors about the given point.
    #         Args:
    #             scale_x (float): The x-coordinate scale factor.
    #             scale_y (float): The y-coordinate scale factor.
    #             about (PointType, optional): The point about which to scale. Default is the center of the image.
    #             reps (int, optional): The number of repetitions.
    #             merge (bool, optional): Whether to merge the images.

    #         Returns:
    #             Image: An Image object or a Group of images.
    #     """
    #     if scale_y is None:
    #         scale_y = scale_x
    #     transform = scale_in_place_matrix(scale_x, scale_y, about=about)
    #     kwargs = {'transform': Transformation.SCALE, 'sx': scale_x, 'sy': scale_y,
    #                                                                 'about': about}
    #     return self._update(transform, reps=reps, merge=merge, kwargs=kwargs)

    def save(
        self,
        fp: str | os.PathLike[str] | io.BufferedIOBase,
        format: str | None = None,
        **params: object,
    ) -> None:
        """Save ``pil_img`` to ``fp`` (Pillow ``save``).

        Examples:
            >>> import io
            >>> import simetri.graphics as sg
            >>> buf = io.BytesIO()
            >>> sg.Image(size=(2, 2), mode="RGB").save(buf, format="PNG")
            >>> buf.tell() > 0
            True
        """
        self.pil_img.save(fp, format, **params)

    def show(self, title: str | None = None, command: str | None = None) -> None:
        """Display ``pil_img`` with the system viewer (Pillow ``show``).

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(2, 2), mode="RGB").show()  # doctest: +SKIP
        """
        self.pil_img.show(title, command)

    def split(self) -> tuple[PIL_Image.Image, ...]:
        """Split ``pil_img`` into individual band images.

        Examples:
            >>> import simetri.graphics as sg
            >>> len(sg.Image(size=(2, 2), mode="RGB").split())
            3
        """
        return self.pil_img.split()

    def transpose(self, method: int) -> PIL_Image.Image:
        """Transpose ``pil_img`` (Pillow ``transpose``).

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Image(size=(4, 2), mode="RGB").transpose(
            ...     PIL_Image.Transpose.ROTATE_90
            ... ).size
            (2, 4)
        """
        return self.pil_img.transpose(method)


def open_img(file_path: str | os.PathLike[str]) -> Image:
    """Open an image file and wrap it as a Simetri ``Image``.

    Args:
        file_path: Path to a raster image file supported by Pillow.

    Returns:
        Image: Simetri image object backed by the opened Pillow image.

    Examples:
        >>> import os
        >>> import tempfile
        >>> import simetri.graphics as sg
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = os.path.join(tmp, "sample.png")
        ...     sg.Image(size=(3, 2), mode="RGB").pil_img.save(path)
        ...     opened = open_img(path)
        ...     width = opened.width
        ...     opened.pil_img.close()
        ...     width
        3
    """
    img = PIL_Image.open(file_path)

    return Image(img=img)


alpha_composite = PIL_Image.alpha_composite
blend = PIL_Image.blend
composite = PIL_Image.composite
eval = PIL_Image.eval
merge_images = PIL_Image.merge
new = PIL_Image.new
# fromarrow = PIL_Image.fromarrow
frombytes = PIL_Image.frombytes
frombuffer = PIL_Image.frombuffer
fromarray = PIL_Image.fromarray
effect_mandelbrot = PIL_Image.effect_mandelbrot
effect_noise = PIL_Image.effect_noise
linear_gradient = PIL_Image.linear_gradient
radial_gradient = PIL_Image.radial_gradient
register_open = PIL_Image.register_open
register_mime = PIL_Image.register_mime
register_save = PIL_Image.register_save
register_save_all = PIL_Image.register_save_all
register_extension = PIL_Image.register_extension
register_extensions = PIL_Image.register_extensions
registered_extensions = PIL_Image.registered_extensions
register_decoder = PIL_Image.register_decoder
register_encoder = PIL_Image.register_encoder


def is_pil_image(obj: object) -> bool:
    """Return whether ``obj`` is a Pillow ``Image`` instance.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.images.image import is_pil_image
        >>> im = sg.Image(size=(1, 1), mode="RGB")
        >>> is_pil_image(im.pil_img)
        True
        >>> is_pil_image(im)
        False
    """
    return isinstance(obj, PIL_Image.Image)


def supported_formats() -> list[str]:
    """List file extensions for formats Pillow can open on this system.

    Returns:
        list[str]: Sorted extension strings (e.g. ``".png"``).

    Examples:
        >>> from simetri.images.image import supported_formats
        >>> ".png" in supported_formats()
        True
    """

    exts = PIL_Image.registered_extensions()
    supported = {ex for ex, f in exts.items() if f in PIL_Image.OPEN}

    return sorted(supported)


def create_image_from_data(
    image_path: str | os.PathLike[str],
) -> Image | None:
    """Load image bytes from disk into a new Simetri ``Image``.

    Args:
        image_path: Path to the image file.

    Returns:
        Image on success, or ``None`` if loading fails (errors are printed).

    Examples:
        >>> import os
        >>> import tempfile
        >>> import simetri.graphics as sg
        >>> from simetri.images.image import create_image_from_data
        >>> fd, path = tempfile.mkstemp(suffix=".png")
        >>> os.close(fd)
        >>> sg.Image(size=(3, 2), mode="RGB").pil_img.save(path)
        >>> loaded = create_image_from_data(path)
        >>> loaded.width
        3
        >>> os.unlink(path)
    """
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        image = PIL_Image.open(io.BytesIO(image_data))

        new_image = PIL_Image.new(image.mode, image.size)
        new_image.paste(image)

        return Image(new_image)

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except (OSError, ValueError) as e:
        print(f"An error occurred: {e}")
        return None


def _normalize_target_image(
    image: Image | PIL_Image.Image,
) -> Image:
    """Return a Simetri ``Image`` wrapper for image drawing."""
    if isinstance(image, Image):
        target = image
    elif isinstance(image, PIL_Image.Image):
        target = Image(img=image)
    else:
        raise TypeError("image must be a Simetri Image or a PIL Image.")

    return target


def _anchor_bounds(
    anchor: Anchor,
    pos: PointType,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """Return left, bottom, right, top bounds for an anchored image."""
    pos_x, pos_y = pos[:2]
    if anchor == Anchor.CENTER:
        left = pos_x - width / 2
        bottom = pos_y - height / 2
    elif anchor == Anchor.NORTH:
        left = pos_x - width / 2
        bottom = pos_y - height
    elif anchor == Anchor.SOUTH:
        left = pos_x - width / 2
        bottom = pos_y
    elif anchor == Anchor.EAST:
        left = pos_x - width
        bottom = pos_y - height / 2
    elif anchor == Anchor.WEST:
        left = pos_x
        bottom = pos_y - height / 2
    elif anchor == Anchor.NORTHEAST:
        left = pos_x - width
        bottom = pos_y - height
    elif anchor == Anchor.NORTHWEST:
        left = pos_x
        bottom = pos_y - height
    elif anchor == Anchor.SOUTHEAST:
        left = pos_x - width
        bottom = pos_y
    elif anchor == Anchor.SOUTHWEST:
        left = pos_x
        bottom = pos_y
    else:
        raise ValueError(f"Unsupported image anchor: {anchor!r}")

    return left, bottom, left + width, bottom + height


def _image_bounds(image: Image) -> tuple[float, float, float, float]:
    """Return image bounds in Simetri canvas coordinates."""
    width, height = image.pil_img.size
    return _anchor_bounds(image.anchor, image.pos, width, height)


def _canvas_to_pixel(
    point: PointType,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Map a Simetri canvas point to Pillow pixel coordinates."""
    left, _, _, top = bounds
    point_x, point_y = point[:2]

    return point_x - left, top - point_y


def _points_to_pixels(
    points: Sequence[PointType],
    bounds: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Map Simetri canvas points to Pillow pixel coordinates."""
    return [_canvas_to_pixel(point, bounds) for point in points]


def _style_rgba(color: ColorLike | None, alpha: float | None) -> tuple[int, int, int, int]:
    """Return a Pillow RGBA tuple from a Simetri color and alpha."""
    if color is None:
        raise ValueError("Color must be resolved before drawing on an image.")
    if alpha is None:
        raise ValueError("Alpha must be resolved before drawing on an image.")

    color_value = check_color(color)
    red, green, blue = color_value.rgb255
    _, _, _, color_alpha = color_value.rgba255
    combined_alpha = round(color_alpha * alpha)

    return red, green, blue, combined_alpha


def _fill_rgba(sketch: Sketch) -> tuple[int, int, int, int] | None:
    """Return Pillow fill color for a sketch."""
    fill = sketch.fill
    if fill:
        fill_color = _style_rgba(sketch.fill_color, sketch.fill_alpha)
    else:
        fill_color = None

    return fill_color


def _stroke_rgba(sketch: Sketch) -> tuple[int, int, int, int] | None:
    """Return Pillow stroke color for a sketch."""
    stroke = sketch.stroke
    if stroke:
        line_color = _style_rgba(sketch.line_color, sketch.line_alpha)
    else:
        line_color = None

    return line_color


def _line_width(sketch: Sketch) -> int:
    """Return Pillow line width for a sketch."""
    line_width = sketch.line_width
    if line_width is None:
        raise ValueError("Line width must be resolved before drawing on an image.")
    pixel_width = round(line_width)
    if pixel_width < 1:
        raise ValueError("Line width must be at least one pixel.")

    return pixel_width


def _flatten_sketches(
    sketches: Sketch | Sequence[Sketch | Sequence[Sketch]],
) -> list[Sketch]:
    """Flatten sketch lists and composite sketches."""
    if isinstance(sketches, Sequence):
        candidates = sketches
    else:
        candidates = (sketches,)

    flat_sketches = []
    for sketch in candidates:
        if isinstance(sketch, Sequence):
            flat_sketches.extend(_flatten_sketches(sketch))
        elif sketch.subtype == Types.COMPOSITE_SKETCH:
            flat_sketches.extend(_flatten_sketches(sketch.sketches))
        else:
            flat_sketches.append(sketch)

    return flat_sketches


def _draw_shape_sketch(
    drawer: ImageDraw.ImageDraw,
    sketch: Sketch,
    bounds: tuple[float, float, float, float],
) -> None:
    """Draw a polygon/polyline sketch on a Pillow drawer."""
    vertices = _points_to_pixels(sketch.vertices, bounds)
    fill_color = _fill_rgba(sketch)
    line_color = _stroke_rgba(sketch)

    if sketch.closed:
        if fill_color is not None:
            drawer.polygon(vertices, fill=fill_color)
        if line_color is not None:
            line_width = _line_width(sketch)
            outline_vertices = vertices + [vertices[0]]
            drawer.line(outline_vertices, fill=line_color, width=line_width)
    elif line_color is not None:
        line_width = _line_width(sketch)
        drawer.line(vertices, fill=line_color, width=line_width)


def _draw_line_sketch(
    drawer: ImageDraw.ImageDraw,
    sketch: Sketch,
    bounds: tuple[float, float, float, float],
) -> None:
    """Draw a line sketch on a Pillow drawer."""
    vertices = _points_to_pixels(sketch.vertices, bounds)
    line_color = _stroke_rgba(sketch)
    if line_color is not None:
        line_width = _line_width(sketch)
        drawer.line(vertices, fill=line_color, width=line_width)


def _draw_circle_sketch(
    drawer: ImageDraw.ImageDraw,
    sketch: Sketch,
    bounds: tuple[float, float, float, float],
) -> None:
    """Draw a circle sketch on a Pillow drawer."""
    center_x, center_y = sketch.center[:2]
    radius = sketch.radius
    left = center_x - radius
    right = center_x + radius
    bottom = center_y - radius
    top = center_y + radius
    bbox_left, bbox_top = _canvas_to_pixel((left, top), bounds)
    bbox_right, bbox_bottom = _canvas_to_pixel((right, bottom), bounds)
    fill_color = _fill_rgba(sketch)
    line_color = _stroke_rgba(sketch)
    if line_color is None:
        drawer.ellipse(
            (bbox_left, bbox_top, bbox_right, bbox_bottom),
            fill=fill_color,
            outline=line_color,
        )
    else:
        line_width = _line_width(sketch)
        drawer.ellipse(
            (bbox_left, bbox_top, bbox_right, bbox_bottom),
            fill=fill_color,
            outline=line_color,
            width=line_width,
        )


def _draw_ellipse_sketch(
    drawer: ImageDraw.ImageDraw,
    sketch: Sketch,
    bounds: tuple[float, float, float, float],
) -> None:
    """Draw an unrotated ellipse sketch on a Pillow drawer."""
    if sketch.angle != 0:
        raise NotImplementedError(
            "draw_on_image does not support rotated ellipse sketches."
        )

    center_x, center_y = sketch.center[:2]
    left = center_x - sketch.x_radius
    right = center_x + sketch.x_radius
    bottom = center_y - sketch.y_radius
    top = center_y + sketch.y_radius
    bbox_left, bbox_top = _canvas_to_pixel((left, top), bounds)
    bbox_right, bbox_bottom = _canvas_to_pixel((right, bottom), bounds)
    fill_color = _fill_rgba(sketch)
    line_color = _stroke_rgba(sketch)
    if line_color is None:
        drawer.ellipse(
            (bbox_left, bbox_top, bbox_right, bbox_bottom),
            fill=fill_color,
            outline=line_color,
        )
    else:
        line_width = _line_width(sketch)
        drawer.ellipse(
            (bbox_left, bbox_top, bbox_right, bbox_bottom),
            fill=fill_color,
            outline=line_color,
            width=line_width,
        )


def _draw_sketch_on_image(
    drawer: ImageDraw.ImageDraw,
    sketch: Sketch,
    bounds: tuple[float, float, float, float],
) -> None:
    """Draw one supported sketch on a Pillow drawer."""
    subtype = sketch.subtype
    if subtype == Types.SHAPE_SKETCH:
        _draw_shape_sketch(drawer, sketch, bounds)
    elif subtype == Types.LINE_SKETCH:
        _draw_line_sketch(drawer, sketch, bounds)
    elif subtype == Types.CIRCLE_SKETCH:
        _draw_circle_sketch(drawer, sketch, bounds)
    elif subtype == Types.ELLIPSE_SKETCH:
        _draw_ellipse_sketch(drawer, sketch, bounds)
    else:
        raise NotImplementedError(
            f"draw_on_image does not support {subtype!r} sketches."
        )


def draw_on_image(
    sketches: Sketch | Sequence[Sketch | Sequence[Sketch]],
    image: Image | PIL_Image.Image,
) -> Image:
    """Draw sketch snapshots on a copy of the given image.

    Args:
        sketches: Sketch or nested sequence of sketches to draw.
        image: Simetri ``Image`` or Pillow image used as the pixel base.

    Returns:
        Image: New Simetri image containing the drawn sketches.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.images.image import draw_on_image
        >>> base = sg.Image(size=(20, 20), mode="RGB")
        >>> canvas = sg.Canvas()
        >>> _ = canvas.line((0, 0), (40, 40))
        >>> out = draw_on_image(canvas.active_page.sketches[0], base)
        >>> out.width
        20
    """
    target = _normalize_target_image(image)
    source_mode = target.pil_img.mode
    base_rgba = target.pil_img.convert(ImageMode.RGBA)
    overlay = PIL_Image.new(ImageMode.RGBA, base_rgba.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(overlay, ImageMode.RGBA)
    bounds = _image_bounds(target)

    for sketch in _flatten_sketches(sketches):
        _draw_sketch_on_image(drawer, sketch, bounds)

    drawn_rgba = PIL_Image.alpha_composite(base_rgba, overlay)
    if source_mode == ImageMode.RGBA:
        drawn_pil = drawn_rgba
    else:
        drawn_pil = drawn_rgba.convert(source_mode)

    result = Image(img=drawn_pil, pos=target.pos)
    result.primary_points = target.primary_points.copy()
    result.xform_matrix = target.xform_matrix
    result.file_path = target.file_path
    result.anchor = target.anchor

    return result
