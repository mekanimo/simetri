import simetri.graphics as sg

canvas = sg.Canvas()

points  = [(0, 0), (50, 0), (50, 50)]
shape = sg.Shape(points, closed=True)
shape.fill_color = sg.red

print(f"Shape fill_color: {shape.fill_color}")
print(f"Shape fill_color type: {type(shape.fill_color)}")
print(f"Shape fill (property): {shape.fill}")

# Let's check what resolve_property returns
resolved_fill_color = canvas.resolve_property(shape, 'fill_color')
print(f"Resolved fill_color: {resolved_fill_color}")
print(f"Resolved fill_color type: {type(resolved_fill_color)}")

resolved_fill = canvas.resolve_property(shape, 'fill')
print(f"Resolved fill: {resolved_fill}")
print(f"Resolved fill type: {type(resolved_fill)}")

# Let's also check getattr directly
direct_fill_color = getattr(shape, 'fill_color', 'NOT_FOUND')
print(f"Direct getattr fill_color: {direct_fill_color}")

# Check if shape has fill_color attribute
print(f"Shape has fill_color attr: {hasattr(shape, 'fill_color')}")
