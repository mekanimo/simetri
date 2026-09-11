from pulp import LpBinary, LpMinimize, LpProblem, LpVariable, lpSum, value


def solve_nonoverlapping_rectangles(rects, min_sep=0.0, bigM=1e4):
    """
    rects: list of dicts with keys:
        - 'x', 'y' : initial center coordinates
        - 'w', 'h' : fixed width/height
    Returns: list of new centers (x, y) with no overlaps.
    """

    n = len(rects)
    W = [r["w"] for r in rects]
    H = [r["h"] for r in rects]
    x0 = [r["x"] for r in rects]
    y0 = [r["y"] for r in rects]

    # Create model
    prob = LpProblem("RectLayout", LpMinimize)

    # Decision variables: final positions
    x = [LpVariable(f"x_{i}", lowBound=None, upBound=None) for i in range(n)]
    y = [LpVariable(f"y_{i}", lowBound=None, upBound=None) for i in range(n)]

    # L1 movement variables: |x-x0| and |y-y0|
    dxp = [LpVariable(f"dxp_{i}", lowBound=0) for i in range(n)]
    dxn = [LpVariable(f"dxn_{i}", lowBound=0) for i in range(n)]
    dyp = [LpVariable(f"dyp_{i}", lowBound=0) for i in range(n)]
    dyn = [LpVariable(f"dyn_{i}", lowBound=0) for i in range(n)]

    # Link x/y to x0 with signed decomposition: x - x0 = dxp - dxn
    for i in range(n):
        prob += x[i] - x0[i] == dxp[i] - dxn[i]
        prob += y[i] - y0[i] == dyp[i] - dyn[i]

    # Objective: minimize sum of L1 displacements
    prob += lpSum(dxp[i] + dxn[i] + dyp[i] + dyn[i] for i in range(n))

    # Non-overlap constraints using binary variables
    # For each pair (i,j), enforce: i left of j OR i right of j OR i below j OR i above j
    for i in range(n):
        for j in range(i + 1, n):
            # Binary selectors
            left = LpVariable(f"L_{i}_{j}", cat=LpBinary)
            right = LpVariable(f"R_{i}_{j}", cat=LpBinary)
            below = LpVariable(f"B_{i}_{j}", cat=LpBinary)
            above = LpVariable(f"A_{i}_{j}", cat=LpBinary)

            # At least one separation direction must be active
            prob += left + right + below + above >= 1

            # Rectangle extents: (center +/- half width/height)
            Wi2 = W[i] / 2.0
            Wj2 = W[j] / 2.0
            Hi2 = H[i] / 2.0
            Hj2 = H[j] / 2.0

            # If left == 1: x[i] + Wi2 + min_sep <= x[j] - Wj2
            prob += x[i] + Wi2 + min_sep <= x[j] - Wj2 + bigM * (1 - left)

            # If right == 1: x[j] + Wj2 + min_sep <= x[i] - Wi2
            prob += x[j] + Wj2 + min_sep <= x[i] - Wi2 + bigM * (1 - right)

            # If below == 1: y[i] + Hi2 + min_sep <= y[j] - Hj2
            prob += y[i] + Hi2 + min_sep <= y[j] - Hj2 + bigM * (1 - below)

            # If above == 1: y[j] + Hj2 + min_sep <= y[i] - Hi2
            prob += y[j] + Hj2 + min_sep <= y[i] - Hi2 + bigM * (1 - above)

    # Solve
    prob.solve()

    if prob.status != 1:
        raise RuntimeError(f"Solver failed or infeasible. Status={prob.status}")

    res = []
    for i in range(n):
        res.append((value(x[i]), value(y[i])))
    return res
