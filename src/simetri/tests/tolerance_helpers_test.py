import random

import simetri.graphics as sg


def absolute_delta_values(
    simetri_values: set[float], expected_values: set[float]
) -> list[float]:
    """Return absolute deltas between sorted helper and expected values."""
    sorted_simetri_values = sorted(simetri_values)
    sorted_expected_values = sorted(expected_values)
    return [
        abs(simetri_value - expected_value)
        for simetri_value, expected_value in zip(
            sorted_simetri_values, sorted_expected_values
        )
    ]


def random_segment(random_generator: random.Random) -> sg.Shape:
    """Return a random non-degenerate segment shape."""
    while True:
        start_point = (
            random_generator.uniform(-100, 100),
            random_generator.uniform(-100, 100),
        )
        end_point = (
            random_generator.uniform(-100, 100),
            random_generator.uniform(-100, 100),
        )
        if start_point != end_point:
            return sg.Shape([start_point, end_point])


def expected_dist_tols(items, n: int) -> set[float]:
    """Return the expected closest positive distances using brute-force loops."""
    group = sg.Group(items)
    vertices = group.all_vertices
    distances = set()
    for first_index, first_point in enumerate(vertices):
        for second_point in vertices[first_index + 1 :]:
            point_distance = sg.distance(first_point[:2], second_point[:2])
            if point_distance > 0:
                distances.add(point_distance)

    return set(sorted(distances)[:n])


def expected_angle_tols(items, n: int) -> set[float]:
    """Return the expected closest positive angle gaps using brute-force loops."""
    group = sg.Group(items)
    angles = []
    for segment in group.all_segments:
        start_point, end_point = segment
        angles.append(sg.inclination_angle(start_point, end_point))

    differences = set()
    for first_index, angle1 in enumerate(angles):
        for angle2 in angles[first_index + 1 :]:
            angle_difference = abs(angle1 - angle2)
            angle_difference = min(angle_difference, sg.pi - angle_difference)
            if angle_difference > 0:
                differences.add(angle_difference)

    return set(sorted(differences)[:n])


def test_group_tolerance_helpers_match_bruteforce_loops():
    random_generator = random.Random(1729)

    for _ in range(8):
        shapes = [random_segment(random_generator) for _ in range(180)]
        group = sg.Group(shapes)

        for n in (1, 3, 7):
            sg_dist_res = group.check_dist_tol(n)
            test_dist_res = expected_dist_tols(shapes, n)
            print(f"group_dist_res_{n}:", sg_dist_res)
            print(f"group_test_dist_res_{n}:", test_dist_res)
            print(
                f"group absolute dist deltas_{n}:",
                absolute_delta_values(sg_dist_res, test_dist_res),
            )
            assert sg_dist_res == test_dist_res

            sg_angle_res = group.check_angle_tol(n)
            test_angle_res = expected_angle_tols(shapes, n)
            print(f"group_angle_res_{n}:", sg_angle_res)
            print(f"group_test_angle_res_{n}:", test_angle_res)
            print(
                f"group absolute angle deltas_{n}:",
                absolute_delta_values(sg_angle_res, test_angle_res),
            )
            assert sg_angle_res == test_angle_res


def test_public_tolerance_helpers_match_bruteforce_loops_for_mixed_inputs():
    random_generator = random.Random(31415)

    for _ in range(6):
        shapes = [random_segment(random_generator) for _ in range(120)]
        mixed_items = [
            sg.Group(shapes[:40]),
            shapes[40],
            sg.Group(shapes[41:80]),
            shapes[80],
            sg.Group(shapes[81:100]),
            sg.Group(shapes[100:]),
        ]

        for n in (1, 4, 8):
            sg_dist_res = sg.check_dist_tol(mixed_items, n)
            test_dist_res = expected_dist_tols(mixed_items, n)
            # print(f"sg_dist_res_{n}:", sg_dist_res)
            # print(f"test_dist_res_{n}:", test_dist_res)
            print(
                f"absolute dist deltas_{n}:",
                absolute_delta_values(sg_dist_res, test_dist_res),
            )
            assert sg_dist_res == test_dist_res

            sg_angle_res = sg.check_angle_tol(mixed_items, n)
            test_angle_res = expected_angle_tols(mixed_items, n)
            # print(f"sg_angle_res_{n}:", sg_angle_res)
            # print(f"test_angle_res_{n}:", test_angle_res)
            print(
                f"absolute angle deltas_{n}:",
                absolute_delta_values(sg_angle_res, test_angle_res),
            )
            assert sg_angle_res == test_angle_res


test_public_tolerance_helpers_match_bruteforce_loops_for_mixed_inputs()
