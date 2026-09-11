import simetri.graphics as sg

canvas = sg.Canvas()
Star = sg.stars.Star
radius = 50
gap = 300
for i in range(10, 15, 2):
    star = Star(i, radius).level(4)
    star.translate(i * gap, 0)
    swatch = sg.random_swatch()
    lace = sg.Lace(star, offset=5, swatch=swatch)
    canvas.draw(lace)

canvas.save("c:/tmp/lace_verification_.pdf", overwrite=True)
canvas.save("c:/tmp/lace_verification_.svg", overwrite=True)
