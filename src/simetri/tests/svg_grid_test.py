import simetri.graphics as sg

# <svg width="100%" height="100%" xmlns="http://w3.org">
#   <defs>
#     <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
#       <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#e5e7eb" stroke-width="3" />
#     </pattern>
#   </defs>
#   <rect width="100%" height="100%" fill="url(#grid)" />
# </svg>

canvas = sg.Canvas()

canvas.draw(sg.letter_F(), fill=False)

canvas.save(
    r"C:\uv_simetri_3.9\simetri\src\simetri\tests\svg_grid_test.svg",
    overwrite=True,
)
