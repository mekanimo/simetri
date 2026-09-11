# Example: minimal displacement non-overlapping rectangles
# Requires: pip install pulp

from pulp import LpBinary, LpMinimize, LpProblem, LpVariable, lpSum, value


def solve_nonoverlapping_rectangles(rects, min_sep=0.0, bigM=1e4):
    n = len(rects)
    W = [r["w"] for r in rects]
    H = [r["h"] for r in rects]
    x0 = [r["x"] for r in rects]
    y0 = [r["y"] for r in rects]

    prob = LpProblem("RectLayout", LpMinimize)

    # Decision variables: final centers
    x = [LpVariable(f"x_{i}") for i in range(n)]
    y = [LpVariable(f"y_{i}") for i in range(n)]

    # L1 movement decomposition for x and y
    dxp = [LpVariable(f"dxp_{i}", lowBound=0) for i in range(n)]
    dxn = [LpVariable(f"dxn_{i}", lowBound=0) for i in range(n)]
    dyp = [LpVariable(f"dyp_{i}", lowBound=0) for i in range(n)]
    dyn = [LpVariable(f"dyn_{i}", lowBound=0) for i in range(n)]

    for i in range(n):
        prob += x[i] - x0[i] == dxp[i] - dxn[i]
        prob += y[i] - y0[i] == dyp[i] - dyn[i]

    # Objective: minimize sum of L1 displacements
    prob += lpSum(dxp[i] + dxn[i] + dyp[i] + dyn[i] for i in range(n))

    # Non-overlap constraints (disjunctive: left/right/below/above)
    for i in range(n):
        for j in range(i + 1, n):
            left = LpVariable(f"L_{i}_{j}", cat=LpBinary)
            right = LpVariable(f"R_{i}_{j}", cat=LpBinary)
            below = LpVariable(f"B_{i}_{j}", cat=LpBinary)
            above = LpVariable(f"A_{i}_{j}", cat=LpBinary)

            # At least one separation relation must hold
            prob += left + right + below + above >= 1

            Wi2 = W[i] / 2.0
            Wj2 = W[j] / 2.0
            Hi2 = H[i] / 2.0
            Hj2 = H[j] / 2.0

            # If left==1 => i is completely left of j (+ min_sep)
            prob += x[i] + Wi2 + min_sep <= x[j] - Wj2 + bigM * (1 - left)

            # If right==1 => i is completely right of j
            prob += x[j] + Wj2 + min_sep <= x[i] - Wi2 + bigM * (1 - right)

            # If below==1 => i is completely below j
            prob += y[i] + Hi2 + min_sep <= y[j] - Hj2 + bigM * (1 - below)

            # If above==1 => i is completely above j
            prob += y[j] + Hj2 + min_sep <= y[i] - Hi2 + bigM * (1 - above)

    prob.solve()

    if prob.status != 1:
        raise RuntimeError(f"Solver failed or infeasible. Status={prob.status}")

    return [(value(x[i]), value(y[i])) for i in range(n)]


if __name__ == "__main__":
    # Three overlapping rectangles (centers)
    rects = [
        {"x": 0.0, "y": 0.0, "w": 4.0, "h": 2.0},  # A
        {"x": 1.0, "y": 0.0, "w": 4.0, "h": 2.0},  # B (overlaps A)
        {"x": 0.5, "y": 0.5, "w": 2.0, "h": 2.0},  # C (overlaps A & B)
    ]

    print("Initial centers:")
    for i, r in enumerate(rects):
        print(f"  {i}: (x={r['x']}, y={r['y']}), w={r['w']}, h={r['h']}")

    final_centers = solve_nonoverlapping_rectangles(
        rects, min_sep=0.0, bigM=1000
    )

    print("\nFinal centers (non-overlapping, minimal L1 push):")
    for i, (xf, yf) in enumerate(final_centers):
        dx = xf - rects[i]["x"]
        dy = yf - rects[i]["y"]
        print(
            f"  {i}: (x={xf:.3f}, y={yf:.3f}), moved (dx={dx:.3f}, dy={dy:.3f})"
        )
