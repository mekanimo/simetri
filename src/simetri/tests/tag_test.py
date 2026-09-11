"""Visual Tag style checks — open c:/tmp/tag_test.svg."""

import simetri.graphics as sg

canvas = sg.Canvas(border=40)
dx, dy = 160, 70


def section(title, x, y):
    canvas.draw(
        sg.Tag(
            title,
            (x, y),
            font_size=11,
            font_color=sg.gray,
            bold=True,
            draw_frame=False,
            align=sg.Align.LEFT,
        )
    )


def mark(pos):
    """Tiny cross so Tag.pos / anchor are visible."""
    x, y = pos[:2]
    canvas.draw(sg.Shape([(x - 3, y), (x + 3, y)]), line_color=sg.red, line_width=0.6)
    canvas.draw(sg.Shape([(x, y - 3), (x, y + 3)]), line_color=sg.red, line_width=0.6)


def draw_tag(tag, pos=None):
    if pos is not None:
        mark(pos)
    else:
        mark(tag.pos)
    canvas.draw(tag)


# --- defaults / basic frame ---
y = 0
section("default / framed / no frame", 0, y + 25)
draw_tag(sg.Tag("default", (0, y)))
draw_tag(
    sg.Tag(
        "framed",
        (dx, y),
        stroke=True,
        line_color=sg.black,
        fill=True,
        fill_color=sg.yellow,
    )
)
draw_tag(sg.Tag("no frame", (2 * dx, y), draw_frame=False, font_color=sg.blue))

# --- font ---
y -= dy
section("font size / color / bold / italic", 0, y + 25)
draw_tag(sg.Tag("size 18", (0, y), font_size=18))
draw_tag(sg.Tag("red text", (dx, y), font_color=sg.red, font_size=14))
draw_tag(sg.Tag("bold", (2 * dx, y), bold=True, font_size=14))
draw_tag(sg.Tag("italic", (3 * dx, y), italic=True, font_size=14))

# --- frame stroke / fill / width ---
y -= dy
section("line / fill / width / dash", 0, y + 25)
draw_tag(
    sg.Tag(
        "stroke",
        (0, y),
        stroke=True,
        line_color=sg.blue,
        fill=False,
    )
)
draw_tag(
    sg.Tag(
        "fill",
        (dx, y),
        stroke=True,
        fill=True,
        fill_color=sg.orange,
        line_color=sg.black,
    )
)
draw_tag(
    sg.Tag(
        "thick",
        (2 * dx, y),
        stroke=True,
        line_width=3,
        line_color=sg.purple,
    )
)
draw_tag(
    sg.Tag(
        "dashed",
        (3 * dx, y),
        stroke=True,
        line_color=sg.green,
        line_dash_array=[4, 2],
    )
)

# --- frame shapes ---
y -= dy
section("frame shapes", 0, y + 25)
shapes = [
    sg.FrameShape.RECTANGLE,
    sg.FrameShape.CIRCLE,
    sg.FrameShape.ELLIPSE,
    sg.FrameShape.DIAMOND,
    sg.FrameShape.SQUARE,
    sg.FrameShape.STAR,
]
for i, shape in enumerate(shapes):
    draw_tag(
        sg.Tag(
            shape.value,
            (i * 110, y),
            frame_shape=shape,
            stroke=True,
            fill=True,
            fill_color=sg.silver,
            line_color=sg.black,
            font_size=10,
            align=sg.Align.CENTER,
        )
    )

# --- rounded / double ---
y -= dy
section("fillets / double line", 0, y + 25)
draw_tag(
    sg.Tag(
        "fillets",
        (0, y),
        stroke=True,
        draw_fillets=True,
        fillet_radius=8,
        fill=True,
        fill_color=sg.cyan,
        line_color=sg.black,
    )
)
draw_tag(
    sg.Tag(
        "double",
        (dx, y),
        stroke=True,
        draw_double=True,
        double_distance=3,
        line_color=sg.black,
        double_color=sg.red,
        fill=True,
        fill_color=sg.white,
    )
)

# --- alpha ---
y -= dy
section("alpha (tag / font / frame)", 0, y + 25)
canvas.draw(sg.letter_F().scale(0.35).translate(-20, y - 15), fill_color=sg.blue)
t_alpha = sg.Tag(
    "tag α=0.4",
    (0, y),
    stroke=True,
    fill=True,
    fill_color=sg.yellow,
    alpha=0.4,
)
t_font = sg.Tag(
    "font α=0.35",
    (dx, y),
    stroke=True,
    fill=True,
    fill_color=sg.yellow,
)
t_font.font_alpha = 0.35
t_frame = sg.Tag(
    "frame α=0.35",
    (2 * dx, y),
    stroke=True,
    fill=True,
    fill_color=sg.yellow,
)
t_frame.frame_alpha = 0.35
draw_tag(t_alpha)
draw_tag(t_font)
draw_tag(t_frame)

# --- align ---
y -= dy
section("align (LEFT / CENTER / RIGHT) vs red cross = pos", 0, y + 25)
for i, align in enumerate([sg.Align.LEFT, sg.Align.CENTER, sg.Align.RIGHT]):
    pos = (i * dx, y)
    draw_tag(
        sg.Tag(
            align.name,
            pos,
            align=align,
            stroke=True,
            fill=True,
            fill_color=sg.pale_yellow,
            font_size=12,
        ),
        pos=pos,
    )

# --- copy preserves style ---
y -= dy
section("copy() keeps style; draw copy at new pos", 0, y + 25)
src = sg.Tag(
    "source",
    (0, y),
    stroke=True,
    fill=True,
    fill_color=sg.pink,
    line_color=sg.maroon,
    font_color=sg.maroon,
    bold=True,
)
copied = src.copy()
copied.text = "copy"
mark((dx, y))
canvas.draw(copied, pos=(dx, y))
draw_tag(src)

# --- mutate style after create ---
y -= dy
section("set style attrs after create", 0, y + 25)
mutated = sg.Tag("was default", (0, y))
mutated.font_color = sg.green
mutated.stroke = True
mutated.fill = True
mutated.fill_color = sg.beige
mutated.line_color = sg.green
mutated.bold = True
mutated.text = "mutated green"
draw_tag(mutated)

# --- TagFrame object ---
y -= dy
section("TagFrame(...) constructor frame", 0, y + 25)
frame = sg.TagFrame(
    frame_shape=sg.FrameShape.ELLIPSE,
    stroke=True,
    fill=True,
    back_color=sg.lavender,
    line_color=sg.purple,
    line_width=2,
    inner_sep=12,
)
draw_tag(sg.Tag("TagFrame", (0, y), frame=frame, font_size=14))

# --- anchors ---
y -= dy
section("anchors (cross = pos)", 0, y + 25)
anchors = [
    sg.Anchor.CENTER,
    sg.Anchor.NORTH,
    sg.Anchor.SOUTH,
    sg.Anchor.WEST,
    sg.Anchor.EAST,
]
for i, anchor in enumerate(anchors):
    pos = (i * 110, y)
    draw_tag(
        sg.Tag(
            anchor.name,
            pos,
            anchor=anchor,
            stroke=True,
            fill=True,
            fill_color=sg.azure,
            font_size=9,
            align=sg.Align.CENTER,
        ),
        pos=pos,
    )

canvas.save("c:/tmp/tag_test.svg", overwrite=True)
print("wrote c:/tmp/tag_test.svg")
