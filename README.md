# simetri.graphics ![logo](/images/logo4.svg)
*simetri.graphics* is a graphics library for Python that focuses on 2D symmetry operations and pattern generation. It uses the TikZ library (see https://tikz.net) and generates .tex files that can be compiled to create output files. It can also be used in Jupyter notebooks to create complex geometric patterns and designs. The library is designed to be easy to use and flexible, allowing users to create a wide variety of symmetrical patterns with minimal effort. The library also includes a number of computational geometry utility functions for working with 2D geometry.

It is designed to be used in conjunction with XeLaTeX rendering engine. Currently the project is in its late alpha stages and is not ready for production use. Beta release is expected to be in March 2025. Although this version is a proof of concept and is likely to change significantly in the future, it is already capable of producing some interesting results.

*simetri.graphics* can generate output files with .tex, .pdf, .ps, .eps, .svg, or .png extensions. It can also generate Jupyter notebook cells with the output embedded in them.

The documentation is available at [simetri](https://simetri). There is also a gallery of examples available at [simetri/gallery](https://github.com/mekanimo/simetri/blob/main/gallery/gallery.md).

## Version

This is the first alpha version of the library and is not yet ready for production use. The library is still in its early stages of development and is likely to change significantly in the near future. The beta release is expected to be in March 2025.

## Installation
### If you have Python installed
If you have a Python version >= 3.9 installed, execute the following command in the terminal:

```pip install simetri```

This will not install a LaTeX distribution, so you will need to install one separately.

### If you don't have Python installed

You can use `uv` to install both Python and the library.

- First install `uv` https://docs.astral.sh/uv/getting-started/installation/
- Then execute the following command in a project directory in the terminal:

```uv python install 3.13 && uv install simetri```

### Install a LaTeX distribution
There are several LaTeX distributions freely available for different operating systems. MikTeX handles package installations automatically, so it is recommended for users who are not familiar with LaTeX typesetting engines.

MiKTeX can be downloaded from https://miktex.org/download recommended.

## Requirements

- Python version 3.9 or later.
- A LaTeX distribution with XeLaTeX engine is required for rendering the output. Miktex is the recommended distribution since it handles installation of required packages automatically.

The library requires the following Python packages:

- `numpy`
- `networkx`
- `matplotlib`
- `Pillow`
- `IPython`
- `pymupdf`
- `strenum`
- `typing-extensions`

These extensions are installed automatically when you install the library using pip or uv.

## Example
```python
import simetri.graphics as sg

canvas = sg.Canvas()
star = sg.stars.Star(n=12)
swatch = sg.swatches_255[9]
lace = sg.Lace(star.level(4), offset=5, swatch=swatch)

canvas.draw(lace)

canvas.display()
```
![12 sided star](/images/star_example.svg)
## Documentation
> [!NOTE]
> Never import the library using `from simetri import *` or `from simetri.graphics import *`.

## Contact

If you have any questions or suggestions, please feel free to contact me at [fbasegmez@gmail.com](mailto:fbasegmez@gmail.com)

## Feedback

If you have any feedback or suggestions for the library, please feel free to open an issue on the GitHub repository or contact me by email. I am always looking for ways to improve the library and make it more useful for users.

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3) - see the LICENSE file for details.
