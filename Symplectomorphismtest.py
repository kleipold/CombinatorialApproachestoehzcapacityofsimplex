#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import numpy as np
import networkx as nx
from scipy.linalg import qr
from networkx.algorithms import isomorphism as iso


def symplectic_matrix(dim):
    """Return the standard symplectic matrix in dimension 2n."""
    if dim % 2 != 0:
        raise ValueError("Dimension must be even.")
    n = dim // 2
    I = np.eye(n)
    Z = np.zeros((n, n))
    return np.block([[Z, I], [-I, Z]])


def center_columns(X):
    """Center the columns at their centroid."""
    return X - np.mean(X, axis=1, keepdims=True)


def validate_simplex_matrix(X, name="X"):
    """Check that X has shape (2n, 2n+1)."""
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")

    d, m = X.shape
    if m != d + 1:
        raise ValueError(f"{name} must have shape (2n, 2n+1). Got {X.shape}.")
    if d % 2 != 0:
        raise ValueError(f"{name} must have even row dimension. Got {d}.")

    return X


def vertices_rows_to_matrix(vertices, name="vertices"):
    """Convert row vertices to the column format used internally."""
    V = np.asarray(vertices, dtype=float)

    if V.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")

    m, d = V.shape
    if m != d + 1:
        raise ValueError(
            f"{name} must contain exactly 2n+1 vertices in R^(2n). Got {V.shape}."
        )
    if d % 2 != 0:
        raise ValueError(f"The ambient dimension must be even. Got {d}.")

    return V.T


def pairing_matrix(X, J=None):
    """Return X^T J X."""
    X = validate_simplex_matrix(X, "X")
    if J is None:
        J = symplectic_matrix(X.shape[0])
    return X.T @ J @ X


def quantized_label(x, quantum):
    """Quantize a float for graph labels."""
    return int(np.rint(x / quantum))


def graph_from_skew_matrix(W, quantum):
    """Build the labeled complete graph attached to W."""
    m = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(m))

    for i in range(m):
        for j in range(i + 1, m):
            G.add_edge(i, j, label=quantized_label(W[i, j], quantum))

    return G


def find_sigma_networkx(WX, WY, atol=1e-5):
    """Find sigma with WX = WY[sigma, sigma], if it exists."""
    scale = max(1.0, np.max(np.abs(WX)), np.max(np.abs(WY)))
    quantum = atol * scale

    GX = graph_from_skew_matrix(WX, quantum)
    GY = graph_from_skew_matrix(WY, quantum)

    matcher = iso.GraphMatcher(
        GX,
        GY,
        edge_match=iso.categorical_edge_match("label", None),
    )

    for mapping in matcher.isomorphisms_iter():
        sigma = np.array([mapping[i] for i in range(WX.shape[0])], dtype=int)
        if np.allclose(WX, WY[np.ix_(sigma, sigma)], atol=atol, rtol=0.0):
            return sigma

    return None


def choose_independent_columns(X, rank_tol=1e-12):
    """Choose 2n independent columns of X."""
    d, _ = X.shape
    _, R, piv = qr(X, pivoting=True, mode="economic")
    rank = np.sum(np.abs(np.diag(R)) > rank_tol)

    if rank < d:
        raise ValueError("The simplex is not full-dimensional.")

    return np.array(piv[:d], dtype=int)


def recover_symplectic_map(X, Y, sigma, rank_tol=1e-12):
    """Recover A from sigma."""
    X = validate_simplex_matrix(X, "X")
    Y = validate_simplex_matrix(Y, "Y")
    sigma = np.asarray(sigma, dtype=int)

    S = choose_independent_columns(X, rank_tol=rank_tol)
    T = sigma[S]

    A = np.linalg.solve(X[:, S].T, Y[:, T].T).T
    return A, S, T


def test_affine_symplectomorphism(X, Y, atol=1e-5, recenter=True, verify=True):
    """Test affine symplectomorphism for two simplices in column format."""
    X = validate_simplex_matrix(X, "X")
    Y = validate_simplex_matrix(Y, "Y")

    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape. Got {X.shape} and {Y.shape}.")

    if recenter:
        X = center_columns(X)
        Y = center_columns(Y)

    J = symplectic_matrix(X.shape[0])
    WX = pairing_matrix(X, J)
    WY = pairing_matrix(Y, J)

    sigma = find_sigma_networkx(WX, WY, atol=atol)

    if sigma is None:
        return {
            "is_symplectomorphic": False,
            "sigma": None,
            "A": None,
            "X_centered": X,
            "Y_centered": Y,
            "WX": WX,
            "WY": WY,
            "symplectic_error": None,
            "vertex_match_error": None,
            "basis_indices_X": None,
            "basis_indices_Y": None,
        }

    A, S, T = recover_symplectic_map(X, Y, sigma)

    sympl_err = None
    match_err = None

    if verify:
        sympl_err = np.linalg.norm(A.T @ J @ A - J, ord=np.inf)
        match_err = np.linalg.norm(A @ X - Y[:, sigma], ord=np.inf)

        ok = (
            np.allclose(A.T @ J @ A, J, atol=100 * atol, rtol=0.0)
            and np.allclose(A @ X, Y[:, sigma], atol=100 * atol, rtol=0.0)
        )
    else:
        ok = True

    return {
        "is_symplectomorphic": bool(ok),
        "sigma": sigma,
        "A": A,
        "X_centered": X,
        "Y_centered": Y,
        "WX": WX,
        "WY": WY,
        "symplectic_error": sympl_err,
        "vertex_match_error": match_err,
        "basis_indices_X": S,
        "basis_indices_Y": T,
    }


def test_from_vertices(vertices_X, vertices_Y, atol=1e-5, verify=True):
    """Test affine symplectomorphism for simplices given by row vertices."""
    X = vertices_rows_to_matrix(vertices_X, name="vertices_X")
    Y = vertices_rows_to_matrix(vertices_Y, name="vertices_Y")
    return test_affine_symplectomorphism(X, Y, atol=atol, recenter=True, verify=verify)


def read_vertices(prompt):
    """Read vertices as rows from the console."""
    print(prompt)
    print("Enter a Python-style list of vertices as rows, for example:")
    print("[[0, 0], [1, 0], [0, 1]]")

    while True:
        s = input("> ").strip()
        try:
            return np.array(ast.literal_eval(s), dtype=float)
        except Exception as exc:
            print(f"Could not parse the input: {exc}")
            print("Please try again.")


def print_result(result):
    """Print the result."""
    print("=" * 72)
    print("is_symplectomorphic:", result["is_symplectomorphic"])
    print("sigma:", result["sigma"])

    if result["A"] is not None:
        print("A:")
        print(result["A"])

    print("symplectic_error:", result["symplectic_error"])
    print("vertex_match_error:", result["vertex_match_error"])
    print("Centered X:")
    print(result["X_centered"])
    print("Centered Y:")
    print(result["Y_centered"])
    print("WX = X^T J X:")
    print(result["WX"])
    print("WY = Y^T J Y:")
    print(result["WY"])
    print("=" * 72)


def main():
    """Run the console interface."""
    print("=" * 72)
    print("Affine symplectomorphism test for simplices")
    print("Vertices are entered as rows and centered automatically.")
    print("=" * 72)
    print("1) built-in example")
    print("2) enter two simplices")
    print()

    choice = input("Your choice [1]: ").strip() or "1"
    atol_str = input("Absolute tolerance [1e-5]: ").strip()
    atol = float(atol_str) if atol_str else 1e-5

    if choice == "1":
        X_rows = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ])

        A_true = np.array([
            [2.0, 1.0],
            [1.0, 1.0],
        ])
        sigma_true = np.array([2, 0, 1], dtype=int)

        X = vertices_rows_to_matrix(X_rows)
        Xc = center_columns(X)
        Y_rows = (A_true @ Xc[:, sigma_true]).T

        result = test_from_vertices(X_rows, Y_rows, atol=atol, verify=True)
        print_result(result)

    elif choice == "2":
        VX = read_vertices("Enter the vertices of the first simplex:")
        VY = read_vertices("Enter the vertices of the second simplex:")
        result = test_from_vertices(VX, VY, atol=atol, verify=True)
        print_result(result)

    else:
        raise ValueError("Unknown menu choice.")


if __name__ == "__main__":
    main()