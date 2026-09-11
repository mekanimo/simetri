import simetri.graphics as sg

print("Testing LinPath mirror operation...")

canvas = sg.Canvas()

# Create a simple path
path = sg.LinPath((0, 0))
path.h_line(50)
path.v_line(30)

print(f"Original path vertices: {path.vertices}")
print(f"Original path position: {path.pos}")

try:
    # Test mirror operation
    result = path.mirror(sg.axis_x, reps=1)
    print("✓ Mirror operation completed without errors")
    print(f"Mirror operation returned: {type(result)}")
    print(f"After mirror path vertices: {path.vertices}")
    print(f"After mirror path position: {path.pos}")

    # Test drawing the mirrored path
    canvas.draw(path)
    print("✓ Drawing mirrored path successful")

    # Save to check visual result
    canvas.save('test_mirror_result.pdf', overwrite=True)
    print("✓ Saved mirrored path to PDF")

except Exception as e:
    print(f"❌ Mirror operation failed: {e}")
    import traceback
    traceback.print_exc()

print("Mirror test completed!")
