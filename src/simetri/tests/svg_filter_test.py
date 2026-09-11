# <svg xmlns="http://w3.org" width="100%" height="100%">
#   <defs>
#     <filter id="real-discoloration" x="0%" y="0%" width="100%" height="100%">
#       <!-- 1. Create a broad, sweeping noise map -->
#       <feTurbulence type="fractalNoise" baseFrequency="0.004" numOctaves="3" result="noise" />

#       <!-- 2. Keep the color dark gray/brown, but map the noise directly to transparency -->
#       <feColorMatrix type="matrix" values="
#         0    0    0    0    0.2
#         0    0    0    0    0.18
#         0    0    0    0    0.15
#         1    0    0    0    -0.3"
#         in="noise" result="shadows" />

#       <!-- 3. Multiply those dark patches over the base color -->
#       <feBlend mode="multiply" in="SourceGraphic" in2="shadows" />
#     </filter>
#   </defs>

#   <!-- Clean off-white paper base element -->
#   <rect width="100%" height="100%" fill="#fbfaf7" filter="url(#real-discoloration)" />
# </svg>

import simetri.graphics as sg

OUTPUT = "c:/tmp/real_discoloration.svg"

discoloration = sg.SVG_Filter(
    id="real-discoloration",
    x="0%",
    y="0%",
    width="100%",
    height="100%",
)

# `feColorMatrix` with `type="matrix"` is a **5×4 transform** from input RGBA to output RGBA:

# ```
# R' = a00·R + a01·G + a02·B + a03·A + a04
# G' = a10·R + a11·G + a12·B + a13·A + a14
# B' = a20·R + a21·G + a22·B + a23·A + a24
# A' = a30·R + a31·G + a32·B + a33·A + a34
# ```

# Thelist is row-major, so it lines up as:

# ```
# # R'                  G'                  B'                  A'
#   0, 0, 0, 0, 0.2,    0, 0, 0, 0, 0.18,   0, 0, 0, 0, 0.15,   1, 0, 0, 0, -0.3
# ```

# Which means:

# | Channel | Formula | Meaning |
# |--------|---------|---------|
# | **R'** | `0.2` | fixed reddish component |
# | **G'** | `0.18` | fixed green |
# | **B'** | `0.15` | fixed blue |
# | **A'** | `R − 0.3` | alpha comes from noise red, shifted down |

# So the matrix does two jobs:

# 1. **Force a stain color** — ignore the noise’s own RGB and always paint `(0.2, 0.18, 0.15)` (dark warm gray/brown).
# 2. **Turn noise into a mask** — use the noise’s **R** as opacity, then subtract `0.3` so only brighter noise regions become visible.

# ### How to tweak the numbers

# - **`0.2 / 0.18 / 0.15`** → stain color. Raise them for a lighter tint; change ratios for cooler/warmer (more B → cooler; more R → warmer).
# - **`-0.3` (alpha bias)** → coverage threshold. More negative (e.g. `-0.5`) → fewer, thinner-looking stains. Closer to `0` (e.g. `-0.1`) → more coverage.
# - **`1` in the alpha row** → how strongly noise R drives opacity. `0.5` softens the mask; `2` makes it harsher (then clamp mentally: SVG alpha is still 0–1 after the blend).

# `feTurbulence` noise is usually mid-grayish, so `A' = R - 0.3` is basically “show only the lighter parts of the noise as brown multiply patches.”
# fmt: off
_matrix_values = [0, 0, 0, 0, 0.2, 
                  0, 0, 0, 0, 0.18, 
                  0, 0, 0, 0, 0.15, 
                  1, 0, 0, 0, -0.3]
# fmt: on

discoloration.add(
    sg.feTurbulence(
        turbulence_type="fractalNoise",
        baseFrequency=0.003,
        numOctaves=3,
        result="noise",
    ),
    sg.feColorMatrix(
        in_="noise",
        matrix_type=sg.ColorMatrix.MATRIX,
        values=_matrix_values,
        result="shadows",
    ),
    sg.feBlend(
        in_="SourceGraphic",
        in2="shadows",
        mode="multiply",
    ),
)

paper = sg.Rectangle(
    center=(200, 150),
    width=400,
    height=300,
    fill=True,
    fill_color=sg.Color(*sg.hex2rgb("#fbfaf7")),
    stroke=False,
)

canvas = sg.Canvas()
canvas.draw(paper, filter=discoloration)
canvas.save(OUTPUT, overwrite=True)
print(f"Saved: {OUTPUT}")
