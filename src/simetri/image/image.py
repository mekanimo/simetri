import os
from typing import Any, Optional, Union, Callable, Tuple, Dict, List, Sequence

from PIL import Image as PIL_Image


from ..graphics.core import Base, StyleMixin
from ..graphics.affine import identity_matrix
from ..graphics.common import common_properties
from ..graphics.all_enums import Types
from ..graphics.bbox import BoundingBox, bounding_box
from ..canvas.style_map import TagStyle, tag_style_map

class Image(Base, StyleMixin):
    """
    A class that extends the PIL Image class to add additional functionality.
    Image.pil_img is the PIL Image object.
    For documentation see https://pillow.readthedocs.io/en/stable/.
    """

    def __init__(self, img=None, pos=(0, 0), angle=0, scale=1, **kwargs):
        """
        Initialize an Image object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if img is None:
            # todo: add mode and size to kwargs
            img = PIL_Image.new(**kwargs)
        elif isinstance(img, str):
            if os.path.exists(img):
                img = PIL_Image.open(img)
            else:
                raise FileNotFoundError(f"File {img} not found.")
        elif not isinstance(img, PIL_Image.Image):
            raise TypeError("img must be a PIL Image object or a file path.")
        self.__dict__['pil_img'] = img
        self.__dict__["style"] = TagStyle()
        self.__dict__["_style_map"] = tag_style_map
        self._set_aliases()
        self.pos = pos
        self.type = Types.IMAGE
        self.subtype = Types.IMAGE
        common_properties(self)
        if 'xform_matrix' in kwargs:
            self.xform_matrix = kwargs['xform_matrix']
        else:
            self.xform_matrix = identity_matrix()
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 1.0
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

    @property
    def filename(self) -> str:
        """
        The filename of the image, if available.

        Returns:
            str: The filename of the image or None if not set.
        """
        return self.pil_img.info.get("filename", None)

    @property
    def b_box(self) -> BoundingBox:
        """Return the bounding box of the shape.

        Returns:
            BoundingBox: The bounding box of the shape.
        """
        if self.pil_img:
            extremes = self.pil_img.getbbox()
            if extremes:
                p1 = extremes[0], extremes[1]
                p2 = extremes[2], extremes[3]
                self._b_box = bounding_box([p1, p2])
        else:
            self._b_box = bounding_box([(0, 0)])
        return self._b_box

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

    def alpha_composite(self, im: 'Image', dest: Sequence[int] = (0, 0),
                                    source: Sequence[int] = (0, 0)) -> 'Image':
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

    def convert(self, mode=None, matrix=None, dither=None, palette=0, colors=256):
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

    def copy(self):
        """
        Copies this image. Use this method if you wish to paste things into an image, but still retain the original.

        Returns:
            Image: An Image object.
        """
        img = Image(pos=self.pos, img=self.pil_img.copy())
        img.xform_matrix = self.xform_matrix.copy()

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

    def rotate(self, angle, resample=0, expand=0, center=None, translate=None, fillcolor=None):
        """
        Returns a rotated copy of this image.

        Args:
            angle (float): The angle to rotate the image.
            resample (int, optional): An optional resampling filter.
            expand (int, optional): Optional expansion flag.
            center (tuple, optional): Optional center of rotation.
            translate (tuple, optional): An optional post-rotate translation.
            fillcolor (tuple, optional): Optional fill color for the area outside the rotated image.

        Returns:
            Image: An Image object.
        """
        return self.pil_img.rotate(angle, resample, expand, center, translate, fillcolor)

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

def open(file_path):
    img = PIL_Image.open(file_path)

    return Image(img=img)

alpha_composite = PIL_Image.alpha_composite
blend = PIL_Image.blend
composite = PIL_Image.composite
eval = PIL_Image.eval
merge = PIL_Image.merge
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

def convert_png_to_ico(png_path, ico_path, sizes=None):
    """Converts a PNG image to an ICO file.

    Args:
        png_path: Path to the input PNG image.
        ico_path: Path to save the output ICO file.
        sizes: A list of tuples specifying the sizes to include in the ICO file,
               e.g., [(16, 16), (32, 32), (48, 48)]. If None, defaults to [(32, 32)].
    """
    if sizes is None:
      sizes = [(32, 32)]

    img = Image.open(png_path)

    icon_sizes = []
    for size in sizes:
      icon_sizes.append(size)

    img.save(ico_path, sizes=icon_sizes)

    # Example usage:
    convert_png_to_ico("input.png", "output.ico", sizes=[(16, 16), (32, 32)])

def supported_formats() -> List[str]:
    """Generates a list of supported image formats available in your system.
        Returns:
            List[str]: A list of supported image formats available in your system.
    """

    exts = PIL_Image.registered_extensions()
    supported = {ex for ex, f in exts.items() if f in PIL_Image.OPEN}

    return sorted(supported)