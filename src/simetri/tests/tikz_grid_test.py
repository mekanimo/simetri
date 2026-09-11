import simetri.graphics as sg

# border should be zero since the grid dictates the page bounds

# \documentclass[12pt,tikz,border=0pt]{standalone}
# \usepackage{tikz,pgf,xcolor}
# \usepackage{tikz}
# \begin{document}

# \begin{tikzpicture}[x=1pt, y=1pt, scale=1]
# \draw[step=15, gray, thin] (-10, -10) grid (70, 110);
# \tikzset{
# style1/.style={line width=2},
# }
# \draw[style1](0.0, 0.0)-- (20.0, 0.0) -- (20.0, 40.0) -- (40.0, 40.0) -- (40.0, 60.0) -- (20.0, 60.0) -- (20.0, 80.0) -- (50.0, 80.0)
# 	-- (50.0, 100.0) -- (0.0, 100.0) -- cycle;
# \end{tikzpicture}
# \end{document}

canvas = sg.Canvas()
canvas.grid((-10, -10), 75, 121, 15, line_dash_array=None)
canvas.draw(sg.letter_F(), fill=False, line_width=2, line_dash_array=[4, 2])

canvas.save(
    r"C:\uv_simetri_3.9\simetri\src\simetri\tests\tikz_grid_test.svg",
    overwrite=True,
)
