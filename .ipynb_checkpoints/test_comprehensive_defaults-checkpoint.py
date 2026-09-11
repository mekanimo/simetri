import simetri.graphics as sg
from simetri.settings.settings import defaults

print("=== Comprehensive Defaults System Test ===")

# Test 1: LinPath with None values should work
path = sg.LinPath()
print(f"✓ LinPath created")

# Test 2: Initial alpha should be None, but canvas should resolve to default
print(f"path.alpha: {path.alpha}")
print(f"defaults['alpha']: {defaults['alpha']}")

# Test 3: Style object should also have None initially
print(f"path.style.alpha: {path.style.alpha}")

# Test 4: Setting None should not break anything
path.alpha = None
print(f"After setting None: {path.alpha}")

# Test 5: Setting a value should work
path.alpha = 0.5
print(f"After setting 0.5: {path.alpha}")

# Test 6: Canvas property resolution should work
canvas = sg.Canvas()

# Create a path with None alpha
path2 = sg.LinPath()
path2.h_line(50)
path2.v_line(30)
path2.close()

print(f"path2.alpha before drawing: {path2.alpha}")

# Test drawing (this should use defaults for None values)
canvas.draw(path2)
print("✓ Canvas draw with None alpha successful")

# Test drawing with explicit alpha
canvas.draw(path2, alpha=0.7)
print("✓ Canvas draw with explicit alpha successful")

# Test 7: Mirror operation (was causing the original error)
path3 = sg.LinPath()
path3.h_line(50)
try:
    path3.mirror(sg.axis_x, reps=1)
    print("✓ Mirror operation successful")
except Exception as e:
    print(f"❌ Mirror operation failed: {e}")

# Test 8: Copy operation (was also causing validation errors)
try:
    path_copy = path3.copy()
    print("✓ Copy operation successful")
except Exception as e:
    print(f"❌ Copy operation failed: {e}")

# Test 9: Save operation
try:
    canvas.save('test_defaults_output.pdf', overwrite=True)
    print("✓ Save operation successful")
except Exception as e:
    print(f"❌ Save operation failed: {e}")

print("=== All tests completed! ===")
