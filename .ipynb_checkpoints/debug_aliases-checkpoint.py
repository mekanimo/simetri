import simetri.graphics as sg
from simetri.canvas.style_map import shape_style_map

print("Shape style map for fill_color:")
print(f"fill_color maps to: {shape_style_map.get('fill_color', 'NOT_FOUND')}")

# Let's also check the creation process
canvas = sg.Canvas()
points = [(0, 0), (50, 0), (50, 50)]
shape = sg.Shape(points, closed=True)

print(f"\nShape _aliases setup:")
print(f"_aliases in shape.__dict__: {'_aliases' in shape.__dict__}")
if '_aliases' in shape.__dict__:
    print(f"fill_color alias: {shape._aliases.get('fill_color', 'NOT_FOUND')}")

# Let's manually trace what happens
print(f"\nShape style structure:")
print(f"shape.style: {shape.style}")
print(f"shape.style.fill_style: {shape.style.fill_style}")
print(f"shape.style.fill_style.color: {shape.style.fill_style.color}")

# Try to set directly on style object
print(f"\nSetting fill_color on style.fill_style.color directly:")
shape.style.fill_style.color = sg.red
print(f"After direct set - shape.style.fill_style.color: {shape.style.fill_style.color}")
print(f"After direct set - shape.fill_color: {shape.fill_color}")
