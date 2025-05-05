from typing import List, Optional, Union


class Image:
    def __init__(self, *args, **kwargs):
        pass


def supported_formats() -> List[str]:
    """Generates a list of supported image formats available in your system.
        Returns:
            List[str]: A list of supported image formats available in your system.
    """

def open(self, fp, mode='r', formats=None):
    """
    Open an image file and return an Image object.

    Args:
        fp: A file-like object or a string path to the image file.
        mode: The mode to open the image in. Default is 'r' (read).
        formats: A list of formats to try when opening the image.

    Returns:
        Image object.
    """


def alpha_composite(self, im, dest=None):
    """
    Blend two images together using alpha compositing.

    Args:
        im: The image to blend with.
        dest: The destination image. If None, a new image is created.

    Returns:
        The blended image.
    """

def composite(self, im, dest=None):
    """
    Composite two images together.

    Args:
        im: The image to composite with.
        dest: The destination image. If None, a new image is created.

    Returns:
        The composited image.
    """

def eval(self, expr, channel_order=None):
    """
    Evaluate an expression on the image.

    Args:
        expr: The expression to evaluate.
        channel_order: The order of the channels in the image.

    Returns:
        The evaluated image.
    """

def merge_images(self, mode, *args):
    """
    Merge multiple images into one.

    Args:
        mode: The mode to merge the images in.
        *args: The images to merge.

    Returns:
        The merged image.
    """

def new(self, mode, size, color=0):
    """
    Create a new image with the specified mode and size.

    Args:
        mode: The mode of the image.
        size: The size of the image.
        color: The color to fill the image with. Default is 0 (black).

    Returns:
        A new Image object.
    """

def fromarray(self, obj, mode=None):
    """
    Create an image from a numpy array.

    Args:
        obj: The numpy array to create the image from.
        mode: The mode of the image. If None, inferred from the array.

    Returns:
        A new Image object.
    """

def fromarrow(self, obj):
    """
    Create an image from an Arrow table.

    Args:
        obj: The Arrow table to create the image from.

    Returns:
        A new Image object.
    """

def frombytes(self, mode, size, data, decoder_name='raw', *args):
    """
    Create an image from bytes.

    Args:
        mode: The mode of the image.
        size: The size of the image.
        data: The byte data to create the image from.
        decoder_name: The name of the decoder to use. Default is 'raw'.

    Returns:
        A new Image object.
    """

def frombuffer(self, mode, size, data, decoder_name='raw', *args):
    """
    Create an image from a buffer.

    Args:
        mode: The mode of the image.
        size: The size of the image.
        data: The buffer data to create the image from.
        decoder_name: The name of the decoder to use. Default is 'raw'.

    Returns:
        A new Image object.
    """

def effect_mandelbrot(self, **kwargs):
    """
    Apply the Mandelbrot effect to the image.

    Args:
        **kwargs: Additional arguments for the effect.

    Returns:
        The image with the Mandelbrot effect applied.
    """

def effect_noise(self, **kwargs):
    """
    Apply noise to the image.

    Args:
        **kwargs: Additional arguments for the effect.

    Returns:
        The image with noise applied.
    """

def linear_gradient(self, *args, **kwargs):
    """
    Create a linear gradient image.

    Args:
        *args: Arguments for the gradient.
        **kwargs: Additional keyword arguments for the gradient.

    Returns:
        A new Image object with the linear gradient.
    """

def radial_gradient(self, *args, **kwargs):
    """
    Create a radial gradient image.

    Args:
        *args: Arguments for the gradient.
        **kwargs: Additional keyword arguments for the gradient.

    Returns:
        A new Image object with the radial gradient.
    """


def preinit(self, mode, size):
    """
    Preinitialize the image with the specified mode and size.

    Args:
        mode: The mode of the image.
        size: The size of the image.

    Returns:
        None
    """

def init(self, mode, size):
    """
    Initialize the image with the specified mode and size.

    Args:
        mode: The mode of the image.
        size: The size of the image.

    Returns:
        None
    """

def register_open(self, format, factory, accept=None):
    """
    Register a new image format.

    Args:
        format: The name of the format.
        factory: The factory function to create the image.
        accept: A function to check if the format is accepted.

    Returns:
        None
    """

def register_mime(self, format, mime):
    """
    Register a new MIME type for the image format.

    Args:
        format: The name of the format.
        mime: The MIME type to register.

    Returns:
        None
    """

def register_save(self, format, factory):
    """
    Register a new save format for the image.

    Args:
        format: The name of the format.
        factory: The factory function to create the image.

    Returns:
        None
    """

def register_save_all(self, format, factory):
    """
    Register a new save_all format for the image.

    Args:
        format: The name of the format.
        factory: The factory function to create the image.

    Returns:
        None
    """

def register_extensions(self, format, extensions):
    """
    Register new file extensions for the image format.

    Args:
        format: The name of the format.
        extensions: A list of file extensions to register.

    Returns:
        None
    """

def registered_extensions(self, format):
    """
    Get the registered file extensions for the image format.

    Args:
        format: The name of the format.

    Returns:
        A list of registered file extensions.
    """


def register_decoder(self, format, decoder):
    """
    Register a new decoder for the image format.

    Args:
        format: The name of the format.
        decoder: The decoder function to use.

    Returns:
        None
    """

def register_encoder(self, format, encoder):
    """
    Register a new encoder for the image format.

    Args:
        format: The name of the format.
        encoder: The encoder function to use.

    Returns:
        None
    """
