from ortools.sat.python import cp_model

def cordial_status_cpsat(G, time_limit_s=None, workers=8, symmetry_break=True):
    """
    Returns (result, labeling, status_str)
      - result: True (cordial), False (not cordial), None (unknown / timeout)
      - labeling: dict node->0/1 if result=True else None
      - status_str: 'FEASIBLE'/'OPTIMAL'/'INFEASIBLE'/'UNKNOWN'
    """
    model = cp_model.CpModel()

    nodes = list(G.nodes())
    edges = list(G.edges())
    V = len(nodes)
    E = len(edges)

    x = {v: model.NewBoolVar(f"x[{v}]") for v in nodes}

    y = {}
    for i, (u, v) in enumerate(edges):
        ye = model.NewBoolVar(f"y[{i}]")
        y[(u, v)] = ye
        # ye = |x[u] - x[v]|  (XOR)
        model.Add(x[u] - x[v] <= ye)
        model.Add(x[v] - x[u] <= ye)
        model.Add(ye <= x[u] + x[v])
        model.Add(ye <= 2 - (x[u] + x[v]))

    # Vertex balance: sum(x) in [floor(V/2), ceil(V/2)]
    sx = sum(x[v] for v in nodes)
    model.Add(sx >= V // 2)
    model.Add(sx <= (V + 1) // 2)

    # Edge balance: sum(y) in [floor(E/2), ceil(E/2)]
    sy = sum(y[e] for e in y)
    model.Add(sy >= E // 2)
    model.Add(sy <= (E + 1) // 2)

    # Symmetry breaking: flip-all-labels symmetry
    if symmetry_break and nodes:
        model.Add(x[nodes[0]] == 0)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    if time_limit_s is not None:
        solver.parameters.max_time_in_seconds = float(time_limit_s)

    status = solver.Solve(model)
    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    status_str = status_map.get(status, str(status))

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        labeling = {v: int(solver.Value(x[v])) for v in nodes}
        return True, labeling, status_str
    if status == cp_model.INFEASIBLE:
        return False, None, status_str
    return None, None, status_str