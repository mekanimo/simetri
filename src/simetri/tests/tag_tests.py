import simetri.graphics as sg

canvas = sg.Canvas()

families = ['Arial', 'Courier New', 'Georgia', 'Times New Roman', 'Verdana',
            sg.FontFamily.SANSSERIF, sg.FontFamily.SERIF, sg.FontFamily.MONOSPACE]

sizes = [8, 24, sg.FontSize.TINY, sg.FontSize.SMALL, sg.FontSize.NORMAL,
         sg.FontSize.LARGE, sg.FontSize.LARGE2, sg.FontSize.LARGE3,sg.FontSize.HUGE,
         sg.FontSize.HUGE2]

count = 0
for family in families:
    for size in sizes:
        for bold in [True, False]:
            for italic in [True, False]:
                text = f'family: {family}, size: {size}, bold: {bold}, italic: {italic}'
                y = count * -40
                tag = sg.Tag(text, (0, y), font_family=family, font_size=size, draw_frame=True)
                tag.bold = bold
                tag.italic = italic
                canvas.draw(tag, fill=True)
                count += 1

canvas.save('c:/tmp/tag_tests.svg', overwrite=True)
