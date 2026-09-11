import simetri.graphics as sg

points = [(0, 0), (50, 0), (50, 50)]
shape = sg.Shape(points, closed=True)

print("Method Resolution Order:")
for i, cls in enumerate(shape.__class__.__mro__):
    print(f"  {i}: {cls}")

print(f"\nShape has fill_color in __dict__: {'fill_color' in shape.__dict__}")
print(f"Shape _aliases has fill_color: {'fill_color' in shape._aliases}")

# Let's trace what happens when we call shape.fill_color
print(f"\nDirect alias resolution:")
if 'fill_color' in shape._aliases:
    obj, attrib = shape._aliases['fill_color']
    print(f"  obj: {obj}")
    print(f"  attrib: {attrib}")
    direct_result = getattr(obj, attrib)
    print(f"  getattr(obj, attrib): {direct_result}")

# Test if Base.__getattr__ is interfering
print(f"\nChecking what super().__getattr__ returns:")
try:
    # This should call Base.__getattr__ since Shape calls super() first
    base_result = super(sg.Shape, shape).__getattr__('fill_color')
    print(f"  Base.__getattr__('fill_color'): {base_result}")
except Exception as e:
    print(f"  Base.__getattr__('fill_color') failed: {e}")

# Test the actual access
print(f"\nActual shape.fill_color: {shape.fill_color}")

# Let's try setting it and see what happens
print(f"\nTesting assignment:")
shape.fill_color = sg.red
print(f"After assignment - shape.fill_color: {shape.fill_color}")
