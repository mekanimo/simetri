# simetri.graphics ![logo](/images/logo.svg)
*simetri.graphics* is a graphics library for Python that focuses on 2D symmetry operations and pattern generation. It uses the TikZ library (see https://tikz.net) and generates .tex files that can be compiled to create output files. It can also be used in Jupyter notebooks to create complex geometric patterns and designs. The library is designed to be easy to use and flexible, allowing users to create a wide variety of symmetrical patterns with minimal effort. The library also includes a number of computational geometry utility functions for working with 2D geometry.

It is designed to be used in conjunction with XeLaTeX rendering engine. Currently the project is in its late alpha stages and is not ready for production use. Beta release is expected to be in May 2025. Although this version is a proof of concept and is likely to change significantly in the future, it is already capable of producing some interesting results.

*simetri.graphics* can generate output files with .tex, .pdf, .ps, .eps, .svg, or .png extensions. It can also generate Jupyter notebook cells with the output embedded in them.

The documentation is available at [simetri/docs](https://github.com/mekanimo/simetri/blob/master/docs/brief_overview.ipynb). There is also a gallery of examples available at [simetri/gallery](https://github.com/mekanimo/simetri/blob/master/gallery.ipynb).

[<img alt="gallery" src="images/gallery.png" />](https://github.com/mekanimo/simetri/blob/master/gallery.ipynb)

## New Version 0.0.6-alpha

This is the second alpha release of the library and is not yet ready for production use. The library is still in its early stages of development and is likely to change significantly in the near future. The beta release is expected to be in May 2025.
Note: First row from the bottom, second image from left is from Jannis Maroscheck's Shape Grammars.
[<img alt="gallery" src="images/img_grid2.png" />]
## What is new in 0.0.6-alpha
- Codebase has been refactored. Since we haven't finished the testing framework we cannot be sure if we introduced new bugs with this refactoring!
- Tested on macOS (Sequoia v.14) with MacTex, Linux (Ubuntu 24.04) with LiveTex and Windows 11 with MikTex.
- Quadratic and cubic Bezier curve objects.
- Hobby curve object.
- Elliptic arc object.
- Sine-wave object.
- LinPath: A path object incorporating both linear and curvilinear objects.
- Ellipse and circle objects.
- A geometric constraint solver. This is very useful for creating circle-packings without using geometry. There will be some examples of this soon.
- Turtle-geometry.
- Lindenmayer-systems.
- Frieze-patterns.
- Wallpaper-patterns (only partially implemented yet.)
- Most API inconsistencies (but not all) are fixed.

## Major issues

- Testing fonts in different systems is problematic. Latex default fonts work on all systems but other fonts may not work reliably on all systems. We may need to create a fall-back system in case a font doesn't exist in a user's system. This requires comprehensive testing. Volunteers needed.
- Scientific Python is used for some features in ellipses, elliptic arcs, and geometric constraint solver. It takes a significant amount of time to import the library. We may have to implement those features ourselves and eliminate the imports.

## What is in the pipeline.

- Documentation. Currently the new version is not documented at all. This will change soon.
- Testing. Before the Beta release we need to establish a comprehensive testing framework.
- Pattern object. This will be much more efficient (but less flexible) for creating large tilings and patterns. Pattern object reduces all transformations into a single matrix. A single matrix multiplication will perform all transformations at once. The resulting matrix will be split into submatrices to get the transformed Shape objects. It is already working but not tested yet. Tilings, frieze-patterns and wallpaper-patterns may inherit from this class.
- Canvas will be able to insert code into the .tex output now. This is very useful for TikZ users who would like to incorporate features that are not included in simetri.graphics.
- Animation facilities using external libraries.
- Sound interface using external libraries.

## Installation
If you have a Python version >= 3.9 installed, execute the following command in the terminal:

```pip install simetri```

This will not install a LaTeX distribution, so you will need to install one separately.

# Check if simetri.graphics installation is successful

- In a terminal window type ```xelatex -help```. If you get an error message, this may mean that you don't have a LaTeX installation or the path to the ```xelatex``` is not in your system path. Make sure that the system path includes the correct path to ```xelatex```.
- Open a Python shell and type

```>>> import simetri.graphics as sg```

```>>> sg.__version__ # this should return '0.0.6'```

```>>> sg.hello()```

You should see a browser window showing you a welcome picture. If you don't see it, try

```>>> sg.Canvas().draw(sg.logo()).save('/your/path/here.pdf')```

If you see the created PDF-file in the given path, this means that your system is preventing opening a web-browser window by third parties. You can use the simetri.graphics but your output files will not be shown automatically.

If you would like to use Jupyter notebooks or JupyterLab you can use the same commands in a Python code-cell.



### Install a LaTeX distribution
There are several LaTeX distributions freely available for different operating systems. The recommended distribution is MacTex for macOS, TexLive for Linux and MikTeX for windows.


TexLive can be downloaded from https://www.tug.org/texlive/
MacTeX can be downloaded from https://www.tug.org/mactex/
MiKTeX can be downloaded from https://miktex.org/download



### Install Jupyter notebooks
If you would like to run the scripts from a notebook environment, you need to install Jupyter Notebooks or JupyterLab. In some systems Jupyter may not use the system settings and behave different from the main Python environment.

See https://jupyter.org/install

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
- `scipy`

These extensions are installed automatically when you install the library using pip or uv.

## Example
```python
import simetri.graphics as sg

canvas = sg.Canvas()

for  i in range(8, 15, 2):
    star = sg.stars.Star(n=i, circumradius=150).level(4)
    star.translate(i * 170, 0)
    swatch = sg.random_swatch()
    lace = sg.Lace(star, swatch=swatch, offset=5)
    canvas.draw(lace.scale(.75))

canvas.display()
```
![12 sided star](/images/example.svg)

> [!NOTE]
> All examples use `canvas.display()` to show the results in a Jupyter notebook. If you are using it as a stand alone library, use `canvas.save("c:/temp/example.pdf")` to save the output as a pdf file. You can generate .pdf, .svg, .ps, .eps, .tex, and .png output.

## Contact

If you have any questions or suggestions, please feel free to contact me at [fbasegmez@gmail.com](mailto:fbasegmez@gmail.com)

## Feedback

If you have any feedback or suggestions for the library, please feel free to open an issue on the GitHub repository or contact me by email. I am always looking for ways to improve the library and make it more useful for users.

## License

This project is licensed under the GNU General Public License v2.0 (GPLv2) - see the LICENSE file for details.
