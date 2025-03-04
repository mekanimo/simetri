# Testing empty shapes and batches
import os
import simetri.graphics as sg

dir_path = "c:/tmp/simetri_test_dir/"
file_name = "simetri_simple_test"
file_path = os.path.join(dir_path, file_name)
extensions = [".svg", ".png", ".pdf", ".ps", ".eps", ".tex"]

canvas = sg.Canvas()
shape = sg.Shape()
canvas.draw(shape)

for ext in extensions:
    canvas.save(f"{file_path}{ext}", show=False, overwrite=True)

canvas = sg.Canvas()
batch = sg.Batch()
canvas.draw(batch)

for ext in extensions:
    canvas.save(f"{file_path}2{ext}", show=False, overwrite=True)

canvas = sg.Canvas()
shape = sg.Shape()
batch = sg.Batch([shape])
canvas.draw(batch)

for ext in extensions:
    canvas.save(f"{file_path}3{ext}", show=False, overwrite=True)

# Testing shapes with one point

canvas = sg.Canvas()
shape = sg.Shape([(0, 0)])
canvas.draw(shape)

for ext in extensions:
    canvas.save(f"{file_path}4{ext}", show=False, overwrite=True)

canvas = sg.Canvas()
shape = sg.Shape([(0, 0)])
batch = sg.Batch([shape])
canvas.draw(batch)

dot = sg.Dot([0, 0], radius=5)
canvas.draw(dot)

for ext in extensions:
    canvas.save(f"{file_path}5{ext}", show=False, overwrite=True)
