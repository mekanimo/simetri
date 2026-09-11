import simetri.graphics as sg

p = sg.Pattern()

p.kernel = sg.letter_F()

p.translate(150, 0, reps=3)
p.rotate(sg.pi/4, reps=7)



for comp in p.transformation.components:
    print(comp.xform_matrix, comp.reps)