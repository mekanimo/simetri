import simetri.graphics as sg
from simetri.canvas.style_map import shape_args

print("shape_args content:")
for arg in sorted(shape_args):
    print(f"  {arg}")

print("\nCreating objects and checking their attributes...")

# Test LinPath
path = sg.LinPath()
print(f"\nLinPath attributes:")
for attr in ['fill', 'stroke', 'alpha']:
    val = getattr(path, attr, 'MISSING')
    print(f"  path.{attr}: {val}")

# Test Shape
shp = sg.Shape([(0, 0), (10, 0), (10, 10)])
print(f"\nShape attributes:")
for attr in ['fill', 'stroke', 'alpha']:
    val = getattr(shp, attr, 'MISSING')
    print(f"  shp.{attr}: {val}")

# Test accessing via style objects
print(f"\nStyle object attributes:")
print(f"  path.style.fill: {getattr(path.style, 'fill', 'MISSING')}")
print(f"  shp.style.fill: {getattr(shp.style, 'fill', 'MISSING')}")

# Test if they're aliased
print(f"\nAliases check:")
print(f"  'fill' in path._aliases: {'fill' in path._aliases}")
print(f"  path._aliases.get('fill'): {path._aliases.get('fill', 'NOT_FOUND')}")
print(f"  'fill' in shp._aliases: {'fill' in shp._aliases}")
print(f"  shp._aliases.get('fill'): {shp._aliases.get('fill', 'NOT_FOUND')}")
