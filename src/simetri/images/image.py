"""Raster image and PDF drawable objects for Simetri canvases.

Wraps Pillow images as canvas items and provides helpers for opening
and constructing ``Image`` / ``PDF`` objects.
"""

import io
import os
from collections.abc import Callable, Sequence
from math import degrees
from typing import Any

from PIL import Image as PIL_Image
from PIL import ImageDraw

from ..base.all_enums import Anchor, ImageMode, TransformationType, Types
from ..base.common import PointType
from ..base.core import _update_inplace
from ..coloring.colors import check_color
from ..geom.affine import (
    rotation_matrix,
    scale_in_place_matrix,
    translation_matrix,
)
from ..geom.matrices import identity_matrix
from ..group.batch import Group
from ..shapes.geom_items import Rectangle


class PDF(Rectangle):
    """
    A class to represent a PDF file as a drawable object.
    """

    def __init__(
        self,
        pdf_path: str,
        pos: PointType = (0, 0),
        size: Sequence[int] | None = None,
        **kwargs,
    ):
        """
        Initialize a PDF object.

        Args:
            pdf_path (str): The path to the PDF file.
            pos (PointType, optional): The position of the PDF on the canvas. Defaults to (0, 0).
            size (Sequence[int], optional): The size of the PDF. If None, uses the original size. Defaults to None.
            **kwargs: Additional keyword arguments for the Rectangle base class.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File {pdf_path} not found.")
        self.pdf_path = pdf_path
        self.type = Types.PDF
        self.subtype = Types.PDF
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

    def __repr__(self):
        """
        Return a string representation of the PDF object.

        Returns:
            str: A string representation of the PDF object.
        """
        return f"PDF({self.pdf_path})"

    def __str__(self):
        """
        Return a human-readable string representation of the PDF object.

        Returns:
            str: A human-readable string representation of the PDF object.
        """
        return f"PDF file at {self.pdf_path}"


class Image(Rectangle):
    """
    A class that extends the PIL Image class to add additional functionality.
    Image.pil_img is the PIL Image object. It behaves like a TikZ node.
    For documentation see https://pillow.readthedocs.io/en/stable/.
    """

    def __init__(
        self,
        img: str | None = None,
        pos: PointType = (0, 0),
        size: Sequence[int] | None = None,
        mode=ImageMode.RGB,
        **kwargs,
    ):
        """
        Initialize an Image object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
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
        super().__init__(width, height, center=pos, **kwargs)
        self.file_path = file_path
        self.type = Types.IMAGE
        self.subtype = Types.IMAGE
        self.anchor = kwargs.get("anchor", Anchor.CENTER)
        if "xform_matrix" in kwargs:
            self.xform_matrix = kwargs["xform_matrix"]
        else:
            self.xform_matrix = identity_matrix()

    def __repr__(self):
        """
        Return a string representation of the Image object.

        Returns:
            str: A string representation of the Image object.
        """
        return f"Image({self.pil_img.size}, {self.pil_img.mode})"

    def __str__(self):
        """
        Return a human-readable string representation of the Image object.

        Returns:
            str: A human-readable string representation of the Image object.
        """
        return f"Image of size {self.pil_img.size} and mode {self.pil_img.mode}"

    def __getattr__(self, name: str) -> Any:
        """
        Get an attribute from the underlying PIL Image object.

        Args:
            name (str): The name of the attribute to get.

        Returns:
            Any: The value of the requested attribute.
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
        incr=None,
        dyn_ref: bool | None = None,
        merge: bool = False,
        xform_type: TransformationType = None,
    ) -> "Group | Image":
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

    @property
    def pos(self) -> PointType:
        """
        The position of the image.

        Returns:
            PointType: The position of the image.
        """
        return self.midpoint

    @pos.setter
    def pos(self, point: PointType) -> None:
        """
        Set the position of the image.

        Args:
            point (PointType): The new position of the image.
        """
        x, y = self.pos[:2]
        dx = point[0] - x
        dy = point[1] - y
        self.xform_matrix = translation_matrix(dx, dy) @ self.xform_matrix

    @property
    def pil_img(self) -> PIL_Image.Image:
        """
        The underlying PIL Image object.

        Returns:
            PIL_Image.Image: The PIL Image object.
        """
        return self.__dict__["pil_img"]

    @property
    def filename(self) -> str:
        """
        The filename of the image, if available.

        Returns:
            str: The filename of the image or None if not set.
        """
        return self.pil_img.info.get("filename", None)

    @property
    def format(self) -> str:
        """
        The format of the image, if available.

        Returns:
            str: The format of the image (e.g., "JPEG", "PNG") or None if not set.
        """
        return self.pil_img.format

    @property
    def mode(self) -> str:
        """
        The mode of the image.

        Returns:
            str: The mode of the image (e.g., "RGB", "L").
        """
        return self.pil_img.mode

    @property
    def size(self) -> tuple:
        """
        The size of the image.

        Returns:
            tuple: A tuple (width, height) representing the dimensions of the image in pixels.
        """
        return self.pil_img.size

    @property
    def width(self) -> int:
        """
        The width of the image.

        Returns:
            int: The width of the image in pixels.
        """
        return self.pil_img.size[0]

    @property
    def height(self) -> int:
        """
        The height of the image.

        Returns:
            int: The height of the image in pixels.
        """
        return self.pil_img.size[1]

    @property
    def info(self) -> dict:
        """
        A dictionary containing miscellaneous information about the image.

        Returns:
            dict: A dictionary of metadata associated with the image.
        """
        # todo: we can add more info here

        return self.pil_img.info

    @property
    def palette(self):
        """
        The palette of the image, if available.

        Returns:
            ImagePalette: The palette of the image or None if not applicable.
        """
        return self.pil_img.palette

    @property
    def category(self) -> str:
        """
        The category of the image.

        Returns:
            str: The category of the image (e.g., "image").
        """
        return self.pil_img.category

    @property
    def readonly(self) -> bool:
        """
        Whether the image is read-only.

        Returns:
            bool: True if the image is read-only, False otherwise.
        """
        return self.pil_img.readonly

    @property
    def decoderconfig(self) -> tuple:
        """
        The decoder configuration of the image.

        Returns:
            tuple: A tuple containing the decoder configuration.
        """
        return self.pil_img.decoderconfig

    @property
    def decodermaxblock(self) -> int:
        """
        The maximum block size used by the decoder.

        Returns:
            int: The maximum block size in bytes.
        """
        return self.pil_img.decodermaxblock

    def alpha_composite(
        self,
        im: "Image",
        dest: Sequence[int] = (0, 0),
        source: Sequence[int] = (0, 0),
    ) -> "Image":
        """
        Blend two images together using alpha compositing.
        This method is a wrapper around the PIL alpha_composite method.

        Args:
            im (Image): The source image to composite with.
            dest (Sequence[int], optional): The destination coordinates. Defaults to (0, 0).
            source (Sequence[int], optional): The source coordinates. Defaults to (0, 0).

        Returns:
            Image: The resulting image after alpha compositing.
        """
        return self.pil_img.alpha_composite(im, dest, source)

    def apply_transparency(self) -> None:
        """
        Apply transparency to the image.

        This method is a wrapper around the PIL apply_transparency method.
        """
        return self.pil_img.apply_transparency()

    def convert(
        self, mode=None, matrix=None, dither=None, palette=0, colors=256
    ):
        """
        Converts an image to a different mode.

        Args:
            mode (str, optional): The requested mode. See: :ref:`concept-modes`.
            matrix (list, optional): An optional conversion matrix.
            dither (int, optional): Dithering method, used when converting from mode "RGB" to "P" or from "RGB" or "L" to "1".
            palette (int, optional): Palette to use when converting from mode "RGB" to "P".
            colors (int, optional): Number of colors to use for the palette.

        Returns:
            Image: An Image object.
        """
        return self.pil_img.convert(mode, matrix, dither, palette, colors)

    def copy(self, **kwargs):
        """
        Copies this image. Use this method if you wish to paste things into an image, but still retain the original.

        Returns:
            Image: An Image object.
        """
        img = Image(pos=self.pos, img=self.pil_img.copy())
        img.primary_points = self.primary_points.copy()
        img.xform_matrix = self.xform_matrix
        img.file_path = self.file_path
        img.anchor = self.anchor

        for k, v in kwargs.items():
            setattr(img, k, v)

        return img

    def crop(self, box=None):
        """
        Returns a rectangular region from this image. The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.

        Args:
            box (tuple, optional): The crop rectangle, as a (left, upper, right, lower)-tuple.

        Returns:
            Image: An Image object.
        """
        return self.pil_img.crop(box)

    def draft(self, mode, size):
        """
        Configures the image file loader so it returns a version of the image that as closely as possible matches the given mode and size.

        Args:
            mode (str): The requested mode.
            size (tuple): The requested size.

        Returns:
            None
        """
        return self.pil_img.draft(mode, size)

    def effect_spread(self, distance):
        """
        Randomly spreads pixels in an image.

        Args:
            distance (int): Distance to spread pixels.

        Returns:
            Image: An Image object.
        """
        return self.pil_img.effect_spread(distance)

    def filter(self, filter):
        """
        Applies the given filter to this image.

        Args:
            filter (Filter): Filter kernel.

        Returns:
            Image: An Image object.
        """
        return self.pil_img.filter(filter)

    def getbands(self):
        """
        Returns a tuple containing the name of each band in this image. For example, "RGB" returns ("R", "G", "B").

        Returns:
            tuple: A tuple containing band names.
        """
        return self.pil_img.getbands()

    def _getbbox(self):
        """
        Calculates the bounding box of the non-zero regions in the image.

        Returns:
            tuple: The bounding box is returned as a 4-tuple defining the left, upper, right, and lower pixel coordinate.
        """
        return self.pil_img.getbbox()

    def getcolors(self, maxcolors=256):
        """
        Returns a list of colors used in this image.

        Args:
            maxcolors (int, optional): Maximum number of colors. If this number is exceeded, this method stops counting and returns None.

        Returns:
            list: A list of (count, pixel) values.
        """
        return self.pil_img.getcolors(maxcolors)

    def getdata(self, band=None):
        """
        Returns the contents of this image as a sequence object containing pixel values.

        Args:
            band (int, optional): What band to return. Default is None.

        Returns:
            Sequence: Pixel values.
        """
        return self.pil_img.getdata(band)

    def getextrema(self):
        """
        Gets the minimum and maximum pixel values for each band in the image.

        Returns:
            tuple: A tuple containing one (min, max) tuple for each band.
        """
        return self.pil_img.getextrema()

    def getpixel(self, xy):
        """
        Returns the pixel value at a given position.

        Args:
            xy (tuple): The coordinate, given as (x, y).

        Returns:
            Pixel: The pixel value.
        """
        return self.pil_img.getpixel(xy)

    def histogram(self, mask=None, extrema=None):
        """
        Returns a histogram for the image.

        Args:
            mask (Image, optional): A mask image.
            extrema (tuple, optional): A tuple of manually-specified extrema.

        Returns:
            list: A list containing pixel counts.
        """
        return self.pil_img.histogram(mask, extrema)

    def paste(self, im, box=None, mask=None):
        """
        Pastes another image into this image.

        Args:
            im (Image or tuple): The source image or pixel value.
            box (tuple, optional): A 2-tuple giving the upper left corner, or a 4-tuple defining the left, upper, right, and lower pixel coordinate.
            mask (Image, optional): A mask image.

        Returns:
            None
        """
        return self.pil_img.paste(im, box, mask)

    def resize(self, size, resample=None, box=None, reducing_gap=None):
        """
        Returns a resized copy of this image.

        Args:
            size (tuple): The requested size in pixels, as a 2-tuple.
            resample (int, optional): An optional resampling filter.
            box (tuple, optional): A box to define the region to resize.
            reducing_gap (float, optional): Apply optimization by resizing the image in two steps.

        Returns:
            Image: An Image object.
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

    def save(self, fp, format=None, **params):
        """
        Saves this image under the given filename.

        Args:
            fp (str or file object): A filename (string) or file object.
            format (str, optional): Optional format override.
            **params: Extra parameters to the image writer.

        Returns:
            None
        """
        return self.pil_img.save(fp, format, **params)

    def show(self, title=None, command=None):
        """
        Displays this image.

        Args:
            title (str, optional): Optional title for the image window.
            command (str, optional): Command used to show the image.

        Returns:
            None
        """
        return self.pil_img.show(title, command)

    def split(self):
        """
        Splits this image into individual bands.

        Returns:
            tuple: A tuple containing individual bands as Image objects.
        """
        return self.pil_img.split()

    def transpose(self, method):
        """
        Transposes this image.

        Args:
            method (int): One of the transpose methods.

        Returns:
            Image: An Image object.
        """
        return self.pil_img.transpose(method)


def open_img(file_path):
    """Open an image file and wrap it as a Simetri ``Image``.

    Args:
        file_path: Path to a raster image file supported by Pillow.

    Returns:
        Image: Simetri image object backed by the opened Pillow image.
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


def is_pil_image(obj):
    """
    Checks if an object is a PIL Image object.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a PIL Image object, False otherwise.
    """
    return isinstance(obj, PIL_Image.Image)


def supported_formats() -> list[str]:
    """Generates a list of supported image formats available in your system.
    Returns:
        list[str]: A list of supported image formats available in your system.
    """

    exts = PIL_Image.registered_extensions()
    supported = {ex for ex, f in exts.items() if f in PIL_Image.OPEN}

    return sorted(supported)


def create_image_from_data(image_path):
    """Load image bytes from disk into a new Simetri ``Image``.

    Args:
        image_path: Path to the image file.

    Returns:
        Image: Wrapped image on success, or None if loading fails.
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


def _normalize_target_image(image):
    """Return a Simetri Image wrapper for image drawing."""
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


def _style_rgba(color, alpha: float) -> tuple[int, int, int, int]:
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


def _fill_rgba(sketch):
    """Return Pillow fill color for a sketch."""
    fill = sketch.fill
    if fill:
        fill_color = _style_rgba(sketch.fill_color, sketch.fill_alpha)
    else:
        fill_color = None

    return fill_color


def _stroke_rgba(sketch):
    """Return Pillow stroke color for a sketch."""
    stroke = sketch.stroke
    if stroke:
        line_color = _style_rgba(sketch.line_color, sketch.line_alpha)
    else:
        line_color = None

    return line_color


def _line_width(sketch) -> int:
    """Return Pillow line width for a sketch."""
    line_width = sketch.line_width
    if line_width is None:
        raise ValueError("Line width must be resolved before drawing on an image.")
    pixel_width = round(line_width)
    if pixel_width < 1:
        raise ValueError("Line width must be at least one pixel.")

    return pixel_width


def _flatten_sketches(sketches) -> list:
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


def _draw_shape_sketch(drawer, sketch, bounds) -> None:
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


def _draw_line_sketch(drawer, sketch, bounds) -> None:
    """Draw a line sketch on a Pillow drawer."""
    vertices = _points_to_pixels(sketch.vertices, bounds)
    line_color = _stroke_rgba(sketch)
    if line_color is not None:
        line_width = _line_width(sketch)
        drawer.line(vertices, fill=line_color, width=line_width)


def _draw_circle_sketch(drawer, sketch, bounds) -> None:
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


def _draw_ellipse_sketch(drawer, sketch, bounds) -> None:
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


def _draw_sketch_on_image(drawer, sketch, bounds) -> None:
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


def draw_on_image(sketches, image):
    """Draw sketch snapshots on a copy of the given image.

    Args:
        sketches: Sketch or sequence of sketches to draw.
        image: Simetri ``Image`` or Pillow image used as the pixel base.

    Returns:
        Image: New Simetri image containing the drawn sketches.
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
