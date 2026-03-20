#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def symplectic_matrix(dim):
    if dim % 2 != 0:
        raise ValueError("The ambient dimension must be even.")

    n = dim // 2
    I = np.eye(n)
    return np.block([
        [np.zeros((n, n)),  I],
        [-I,                np.zeros((n, n))]
    ])


def polytope_matrix(C):
    C = np.asarray(C, dtype=float)

    if C.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    return C @ symplectic_matrix(C.shape[1]) @ C.T


def _pool_objval(model):
    try:
        return model.getAttr("PoolNObjVal")
    except Exception:
        return model.getAttr("PoolObjVal")


def solve_and_extract(C, pool_solutions=6000, active_tol_rel=1e-6, verbose=False):
    X = polytope_matrix(C)
    s = X.shape[0]

    model = gp.Model("milp_L_extract")
    model.setParam("OutputFlag", 1 if verbose else 0)
    model.setParam("Threads", 1)

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
        GRB.MAXIMIZE
    )

    model.setParam("PoolSearchMode", 2)
    model.setParam("PoolSolutions", pool_solutions)
    model.setParam("MIPGap", 0)
    model.setParam("MIPGapAbs", 0)
    model.setParam("PoolGap", active_tol_rel)

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi did not prove optimality. Status code: {model.Status}")

    objval = model.ObjVal
    L_list = []

    for sol_num in range(model.SolCount):
        model.setParam("SolutionNumber", sol_num)
        pool_obj = _pool_objval(model)

        if abs(objval) > 1e-14:
            rel_gap = abs(objval - pool_obj) / abs(objval)
        else:
            rel_gap = abs(objval - pool_obj)

        if rel_gap > active_tol_rel:
            continue

        L = np.zeros((s, s), dtype=int)
        for i in range(s):
            for j in range(i + 1, s):
                if y[i, j].Xn > 0.5:
                    L[i, j] = 1
                else:
                    L[j, i] = 1

        L_list.append(L)

    if verbose:
        print("Optimal objective value:", objval)
        print("Number of near-active solutions found:", len(L_list))

    return L_list, objval