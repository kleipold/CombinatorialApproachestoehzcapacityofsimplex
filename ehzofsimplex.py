#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import math
from itertools import combinations

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def check_even_dimension(d):
    """Check that d is even."""
    if d % 2 != 0:
        raise ValueError("The ambient dimension must be even, i.e. d = 2n.")


def symplectic_matrix(dim):
    """Return the standard symplectic matrix on R^dim."""
    check_even_dimension(dim)
    n = dim // 2
    I = np.eye(n)
    return np.block([
        [np.zeros((n, n)),  I],
        [-I,                np.zeros((n, n))]
    ])


def polytope_matrix(C):
    """Return C J C^T for the row matrix C."""
    C = np.asarray(C, dtype=float)
    if C.ndim != 2:
        raise ValueError("C must be a 2D array.")
    return C @ symplectic_matrix(C.shape[1]) @ C.T


def simplex_volume(vertices):
    """Return the Euclidean volume of a simplex with row vertices."""
    V = np.asarray(vertices, dtype=float)
    if V.ndim != 2:
        raise ValueError("vertices must be a 2D array.")

    m, d = V.shape
    if m != d + 1:
        raise ValueError("For a simplex in R^d, vertices must have shape (d+1, d).")

    A = np.vstack([V.T, np.ones((1, d + 1))])
    return abs(np.linalg.det(A)) / math.factorial(d)


def center_simplex(vertices):
    """Center the simplex at its barycenter."""
    V = np.asarray(vertices, dtype=float)
    if V.ndim != 2:
        raise ValueError("vertices must be a 2D array.")

    m, d = V.shape
    if m != d + 1:
        raise ValueError("For a simplex in R^d, vertices must have shape (d+1, d).")

    check_even_dimension(d)

    if np.linalg.matrix_rank(V[1:] - V[0]) != d:
        raise ValueError("The given vertices do not form a full-dimensional simplex.")

    center = np.mean(V, axis=0)
    return V - center, center


def polar_vertices_from_centered_vertices(vertices, tol=1e-9):
    """Compute the polar simplex vertices."""
    V = np.asarray(vertices, dtype=float)
    if V.ndim != 2:
        raise ValueError("vertices must be a 2D array.")

    m, d = V.shape
    if m != d + 1:
        raise ValueError("For a simplex in R^d, vertices must have shape (d+1, d).")

    polar = []

    for idx in combinations(range(m), d):
        sub = V[list(idx)]
        if np.linalg.matrix_rank(sub, tol=tol) < d:
            continue

        x = np.linalg.solve(sub, np.ones(d))
        if np.all(V @ x <= 1 + tol):
            if not any(np.linalg.norm(y - x) < tol for y in polar):
                polar.append(x)

    C = np.array(polar, dtype=float)
    if C.shape != (d + 1, d):
        raise ValueError(
            f"Polar vertex computation failed: expected shape {(d + 1, d)}, got {C.shape}."
        )

    return C


def standard_simplex_vertices(dim):
    """Return the standard simplex 0, e_1, ..., e_dim in R^dim."""
    V = np.zeros((dim + 1, dim), dtype=float)
    V[1:] = np.eye(dim)
    return V


def random_simplex_vertices(dim, seed=None):
    """Return random simplex vertices in R^dim."""
    check_even_dimension(dim)
    rng = np.random.default_rng(seed)

    while True:
        V = rng.standard_normal((dim + 1, dim))
        if np.linalg.matrix_rank(V[1:] - V[0]) == dim:
            return V


def solve_milp_single(C):
    """Solve the MILP and return one optimal L and the optimal value."""
    C = np.asarray(C, dtype=float)
    X = polytope_matrix(C)
    s = X.shape[0]

    model = gp.Model("milp_single")
    model.setParam("OutputFlag", 1)
    model.setParam("Threads", 1)
    model.setParam("MIPGap", 0)
    model.setParam("MIPGapAbs", 0)

    y = {}
    for i in range(s):
        for j in range(i + 1, s):
            y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    for i in range(s - 2):
        for j in range(i + 1, s - 1):
            for k in range(j + 1, s):
                model.addConstr(y[i, j] + y[j, k] - y[i, k] <= 1)
                model.addConstr(-y[i, j] - y[j, k] + y[i, k] <= 0)

    model.setObjective(
        gp.quicksum(
            X[i, j] * y[i, j] + X[j, i] * (1 - y[i, j])
            for i in range(s)
            for j in range(i + 1, s)
        ),
        GRB.MAXIMIZE,
    )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not prove optimality. Status code: {model.Status}")

    L = np.zeros((s, s), dtype=int)
    for i in range(s):
        for j in range(i + 1, s):
            if y[i, j].X > 0.5:
                L[i, j] = 1
            else:
                L[j, i] = 1

    return L, model.ObjVal


def ehz_capacity_from_milp_value(objval, ambient_dim):
    """Convert the MILP value into the EHZ capacity."""
    check_even_dimension(ambient_dim)
    if abs(objval) < 1e-14:
        raise ZeroDivisionError("MILP objective value is too close to zero.")

    s = ambient_dim + 1
    return (s ** 2) / (2.0 * objval)


def systolic_ratio(ehz_capacity, volume, ambient_dim):
    """Return the systolic ratio."""
    check_even_dimension(ambient_dim)
    n = ambient_dim // 2

    if volume <= 0:
        raise ValueError("Volume must be positive.")

    return abs(ehz_capacity / ((math.factorial(n) * volume) ** (1.0 / n)))


def analyze_simplex_from_vertices(vertices):
    """Run the full workflow from simplex vertices."""
    V = np.asarray(vertices, dtype=float)
    Vc, center = center_simplex(V)
    C = polar_vertices_from_centered_vertices(Vc)
    L, objval = solve_milp_single(C)

    vol = simplex_volume(Vc)
    cap = ehz_capacity_from_milp_value(objval, V.shape[1])
    sys = systolic_ratio(cap, vol, V.shape[1])

    return {
        "ambient_dim": V.shape[1],
        "center": center,
        "milp_value": objval,
        "volume": vol,
        "ehz_capacity": cap,
        "systolic_ratio": sys,
    }


def analyze_standard_simplex(n):
    """Analyze the standard simplex in R^(2n)."""
    if n <= 0:
        raise ValueError("n must be positive.")
    return analyze_simplex_from_vertices(standard_simplex_vertices(2 * n))


def analyze_random_simplex(n, seed=None):
    """Analyze a random simplex in R^(2n)."""
    if n <= 0:
        raise ValueError("n must be positive.")
    return analyze_simplex_from_vertices(random_simplex_vertices(2 * n, seed=seed))


def prompt_int(prompt, default=None):
    """Read an integer."""
    while True:
        raw = input(f"{prompt}" + (f" [{default}]: " if default is not None else ": ")).strip()
        if raw == "" and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


def prompt_array_2d(name):
    """Read a 2D array as a Python-style nested list."""
    print(f"Enter {name} as a Python-style nested list with vertices as rows.")
    print("Example: [[0,0],[1,0],[0,1]]")

    while True:
        raw = input(f"{name} = ").strip()
        try:
            arr = np.array(ast.literal_eval(raw), dtype=float)
        except Exception as exc:
            print(f"Could not parse {name}: {exc}")
            continue

        if arr.ndim != 2:
            print(f"{name} must be a 2D array.")
            continue

        return arr


def print_summary(result):
    """Print the main output."""
    print("\n" + "=" * 72)
    print("Ambient dimension:", result["ambient_dim"])
    print("Barycenter:", result["center"])
    print("MILP value:", result["milp_value"])
    print("EHZ capacity:", result["ehz_capacity"])
    print("Volume:", result["volume"])
    print("Systolic ratio:", result["systolic_ratio"])
    print("=" * 72)


def main():
    """Run the console interface once."""
    print("=" * 72)
    print("EHZ capacity / systolic ratio for simplices")
    print("1) standard simplex in R^(2n)")
    print("2) random simplex in R^(2n)")
    print("3) enter simplex vertices directly")
    print("=" * 72)

    choice = input("Your choice [1]: ").strip().lower() or "1"

    if choice == "1":
        n = prompt_int("Enter n")
        result = analyze_standard_simplex(n)

    elif choice == "2":
        n = prompt_int("Enter n")
        seed_raw = input("Seed [press Enter for random]: ").strip()
        seed = int(seed_raw) if seed_raw else None
        result = analyze_random_simplex(n, seed=seed)

    elif choice == "3":
        V = prompt_array_2d("vertices")
        result = analyze_simplex_from_vertices(V)

    else:
        raise ValueError("Unknown choice.")

    print_summary(result)


if __name__ == "__main__":
    main()