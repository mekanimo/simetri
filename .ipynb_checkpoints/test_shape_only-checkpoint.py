import simetri.graphics as sg

print("Testing Shape drawing with explicit check...")

canvas = sg.Canvas()

# Create the exact shape from alcazar script
shp = sg.Shape([(0, 0), (50, 0), (100, 50)], closed=True)

print(f"Shape attributes before drawing:")
print(f"  shp.fill: {shp.fill}")
print(f"  shp.stroke: {shp.stroke}")
print(f"  shp.alpha: {shp.alpha}")
print(f"  shp.closed: {shp.closed}")

# Test drawing
canvas.draw(shp)
print("✓ Shape drawing completed")

# Check what's in the sketches
print(f"Number of sketches: {len(canvas.active_page.sketches)}")
if canvas.active_page.sketches:
    sketch = canvas.active_page.sketches[0]
    print(f"Sketch attributes:")
    print(f"  sketch.fill: {sketch.fill}")
    print(f"  sketch.stroke: {sketch.stroke}")
    print(f"  sketch.alpha: {sketch.alpha}")
    print(f"  sketch.closed: {sketch.closed}")

# Save to file to verify it appears in output
canvas.save('c:\\Users\\manga\\OneDrive\\Documents\\uv_simetri_3.9\\simetri\\test_shape_only.pdf', overwrite=True)
print("✓ Shape saved to PDF")

print("Test completed!")
