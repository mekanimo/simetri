import simetri.graphics as sg

print("Testing Shape vs LinPath drawing...")

canvas = sg.Canvas()

# Test LinPath
path = sg.LinPath()
path.h_line(50)
path.v_line(30)
path.close()

print(f"LinPath alpha: {path.alpha}")
print(f"LinPath fill: {getattr(path, 'fill', 'NO_FILL_ATTR')}")
print(f"LinPath stroke: {getattr(path, 'stroke', 'NO_STROKE_ATTR')}")

# Test Shape
shp = sg.Shape([(0, 0), (50, 0), (100, 50)], closed=True)

print(f"Shape alpha: {shp.alpha}")
print(f"Shape fill: {getattr(shp, 'fill', 'NO_FILL_ATTR')}")
print(f"Shape stroke: {getattr(shp, 'stroke', 'NO_STROKE_ATTR')}")

# Test drawing both
canvas.draw(path)
print("✓ LinPath drawn")

canvas.draw(shp)
print("✓ Shape drawn")

# Test with explicit alpha
canvas.draw(shp, alpha=0.5)
print("✓ Shape drawn with explicit alpha")

# Save and check what's in the output
canvas.save('test_shape_vs_path.pdf', overwrite=True)
print("✓ Saved PDF")

print("Done!")
