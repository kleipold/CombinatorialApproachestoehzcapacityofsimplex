#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import expm
import cvxpy as cp
import ilp




def max_abs_entry(A):
    """Return max_{i,j} |A_ij|."""
    return float(np.nanmax(np.abs(A)))


def unique_matrices(matrices):
    """Remove duplicate matrices."""
    unique = []
    seen = set()

    for M in matrices:
        A = np.asarray(M)
        key = (A.shape, A.dtype.str, A.tobytes())
        if key not in seen:
            seen.add(key)
            unique.append(A)

    return unique


def min_frobenius_in_convex_hull(matrices):
    """Return the minimal-Frobenius element in the convex hull."""
    matrices = unique_matrices(matrices)

    if len(matrices) == 0:
        raise ValueError("Empty list of matrices.")
    if len(matrices) == 1:
        return matrices[0].copy()

    m, n = matrices[0].shape
    for i, G in enumerate(matrices):
        if G.shape != (m, n):
            raise ValueError(
                "All matrices must have the same shape; "
                f"matrix {i} has shape {G.shape} instead of {(m, n)}."
            )

    V = np.column_stack([G.ravel() for G in matrices])

    alpha = cp.Variable(len(matrices), nonneg=True)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(V @ alpha)),
        [cp.sum(alpha) == 1],
    )

    problem.solve(
        solver=cp.OSQP,
        eps_abs=5e-6,
        eps_rel=5e-6,
        max_iter=230000,
        polish=True,
        verbose=False,
    )

    if alpha.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Convex-hull QP failed with status: {problem.status}")

    alpha_val = np.maximum(alpha.value, 0.0)
    s = alpha_val.sum()
    if s <= 0:
        raise RuntimeError("Degenerate convex combination returned by solver.")

    alpha_val /= s
    return (V @ alpha_val).reshape(m, n)


class SpecialLinearGroup:
    """Basic operations on SL(d)."""

    def __init__(self, d):
        self.d = d

    def random_point(self):
        """Sample a random point in SL(d)."""
        while True:
            A = np.random.randn(self.d, self.d)
            U, s, Vt = np.linalg.svd(A)

            if np.min(s) < 1e-5:
                continue

            s = s / (np.prod(s) ** (1.0 / self.d))
            X = U @ np.diag(s) @ Vt

            if np.linalg.det(X) < 0:
                U[:, -1] *= -1
                X = U @ np.diag(s) @ Vt

            if abs(np.linalg.det(X) - 1.0) < 1e-10:
                return X

    def project_tangent(self, X, A):
        """Project A onto T_X SL(d)."""
        X_inv = np.linalg.inv(X)
        correction = (
            np.trace(X_inv @ A) / np.trace(X_inv @ X_inv.T)
        ) * X_inv.T
        return A - correction

    def exp_map(self, X, V):
        """Apply the exponential map at X to V."""
        return X @ expm(np.linalg.inv(X) @ V)


def standard_simplex_constraint_matrix(dim):
    """Return the simplex matrix B."""
    B = np.zeros((dim, dim + 1))
    for i in range(dim):
        B[i, i] = -1.0
        B[i, -1] = 1.0
    return B


def symplectic_matrix(n):
    """Return the standard symplectic matrix on R^(2n)."""
    I = np.eye(n)
    return np.block([
        [np.zeros((n, n)), -I],
        [I, np.zeros((n, n))]
    ])


def cost_function(X):
    """Return (-ehz, active L-matrices)."""
    dim = X.shape[0]
    if dim % 2 != 0:
        raise ValueError("Ambient dimension must be even.")

    B = standard_simplex_constraint_matrix(dim)
    X_inv = np.linalg.inv(X)
    poly = X_inv.T @ B

    L_list, milp_value = ilp.solve_and_extract(
        poly.T,
        active_tol_rel=1e-6,
    )

    minus_ehz = -1.0 / (2 * milp_value)
    return minus_ehz, L_list


def subgradient_function(group, X):
    """Return the minimal-norm subgradient in the active convex hull."""
    dim = X.shape[0]
    n = dim // 2

    B = standard_simplex_constraint_matrix(dim)
    J = symplectic_matrix(n)
    X_inv = np.linalg.inv(X)

    minus_ehz, L_list = cost_function(X)

    G_list = []
    for L in L_list:
        inner = (
            B @ L.T @ B.T @ X_inv @ J
            + B @ L @ B.T @ X_inv @ J.T
        )
        G = -(X_inv.T @ inner @ X_inv.T)
        G = (minus_ehz ** 2) * G
        G_list.append(group.project_tangent(X, G))

    return min_frobenius_in_convex_hull(G_list)


def riemannian_gradient(group, X):
    """Return the descent direction."""
    G = subgradient_function(group, X)
    return -group.project_tangent(X, G)


def find_good_random_point(group, max_tries=5):
    """Return a suitable random start, or the best sampled point."""
    threshold = -1.2 / group.d

    best_X = None
    best_minus_ehz = np.inf

    for trial in range(1, max_tries + 1):
        X = group.random_point()
        minus_ehz, _ = cost_function(X)

        if minus_ehz < best_minus_ehz:
            best_X = X
            best_minus_ehz = minus_ehz

        if minus_ehz <= threshold:
            return X, minus_ehz, trial

    return best_X, best_minus_ehz, max_tries


def default_phases():
    """Return the default phase list."""
    return [
        ("loose",       3e-2,  800,  "fixed",    0.0),
        ("middle",      8e-3,  800,  "fixed",    0.0),
        ("tight",       4e-3,  800,  "fixed",    0.0),
        ("finer",       1e-3,  800,  "fixed",    0.0),
        ("finest",      7e-4,  800,  "fixed",    0.0),
        ("ultrafine",   3e-5,  800,  "fixed",    0.0),
        ("strict",      5e-3, 1000,  "tolerant", 1e-4),
        ("strict2",     8e-4, 1000,  "tolerant", 2e-5),
        ("strict3",     3e-4, 1000,  "tolerant", 7e-6),
        ("strict4",     8e-5, 1000,  "tolerant", 3e-6),
        ("strict5",     4e-5, 1000,  "tolerant", 1e-6),
        ("very_strict", 1e-5, 2000,  "armijo",   1e-6),
    ]


def extension_phases():
    """Return the extension phase list."""
    return [
        ("strict3",     3e-4, 1000, "tolerant", 7e-6),
        ("strict4",     8e-5,  200, "tolerant", 3e-6),
        ("strict5",     4e-5, 1000, "tolerant", 1e-6),
        ("very_strict", 1e-5, 1000, "armijo",   1e-6),
    ]


def uphill_tolerance(value, base_tol, relative=True):
    """Return the allowed uphill tolerance."""
    if relative:
        return base_tol * (abs(value) + 1e-12)
    return base_tol


def run_phase_list(
    group,
    X,
    current_value,
    phases,
    value_history,
    entry_tol=1e-7,
    min_grad_norm=1e-12,
    min_stepsize=1e-8,
    shrink_factor=0.5,
    max_backtracks=60,
    armijo_c1=1e-5,
    verbose=False,
    print_every=200,
):
    """Run one list of descent phases."""
    def log(msg):
        if verbose:
            print(msg, flush=True)

    g_last = riemannian_gradient(group, X)

    for phase_name, step_size, num_iters, mode, tol_up_base in phases:
        log(f"\n[{phase_name}] start  -ehz={current_value:.10e}")

        phase_best_X = X.copy()
        phase_best_value = current_value

        for k in range(num_iters):
            g = riemannian_gradient(group, X)
            g_last = g

            g_maxabs = max_abs_entry(g)
            g_fro = float(np.linalg.norm(g))

            if g_maxabs < entry_tol:
                log(
                    f"[{phase_name}] stopping: max|grad_ij|={g_maxabs:.3e} < {entry_tol:.3e}"
                )
                return X, current_value, riemannian_gradient(group, X)

            if mode == "armijo" and g_fro < min_grad_norm:
                log(
                    f"[{phase_name}] stopping: ||grad||_F={g_fro:.3e} < {min_grad_norm:.3e}"
                )
                return X, current_value, riemannian_gradient(group, X)

            direction = g / (g_fro + 1e-20)

            if mode == "fixed":
                X_new = group.exp_map(X, -step_size * direction)
                new_value, _ = cost_function(X_new)

            else:
                alpha = step_size
                accepted = False
                tol_up = uphill_tolerance(current_value, tol_up_base, relative=True)

                for _ in range(max_backtracks):
                    if alpha < min_stepsize:
                        break

                    X_trial = group.exp_map(X, -alpha * direction)
                    trial_value, _ = cost_function(X_trial)
                    decrease = current_value - trial_value

                    if mode == "tolerant":
                        accepted = (decrease >= -tol_up)
                    elif mode == "armijo":
                        armijo_rhs = armijo_c1 * alpha * g_fro
                        accepted = (decrease >= armijo_rhs) or (decrease >= -tol_up)
                    else:
                        raise ValueError(f"Unknown phase mode: {mode}")

                    if accepted:
                        X_new = X_trial
                        new_value = trial_value
                        break

                    alpha *= shrink_factor

                if not accepted:
                    log(f"[{phase_name}] no acceptable step found")
                    continue

            X = X_new
            current_value = new_value
            value_history.append(current_value)

            if current_value < phase_best_value:
                phase_best_value = current_value
                phase_best_X = X.copy()

            if verbose and ((k + 1) % print_every == 0 or k == num_iters - 1):
                if mode == "armijo":
                    log(
                        f"[{phase_name}] iter={k + 1:4d}/{num_iters:4d}  "
                        f"-ehz={current_value:.10e}  ||grad||_F={g_fro:.3e}  "
                        f"max|grad_ij|={g_maxabs:.3e}  alpha={alpha:.3e}"
                    )
                else:
                    log(
                        f"[{phase_name}] iter={k + 1:4d}/{num_iters:4d}  "
                        f"-ehz={current_value:.10e}  ||grad||_F={g_fro:.3e}  "
                        f"max|grad_ij|={g_maxabs:.3e}"
                    )

        X = phase_best_X.copy()
        current_value = phase_best_value
        g_last = riemannian_gradient(group, X)

        log(f"[{phase_name}] best -ehz in phase = {current_value:.10e}")

    return X, current_value, g_last


def gradient_descent_with_phases(
    group,
    X0,
    phases=None,
    extension=None,
    entry_tol=1e-7,
    min_grad_norm=1e-12,
    extension_threshold=1e-5,
    min_stepsize=1e-8,
    shrink_factor=0.5,
    max_backtracks=60,
    armijo_c1=1e-5,
    verbose=False,
    print_every=200,
):
    """Run the descent method with the default phase scheme."""
    if phases is None:
        phases = default_phases()
    if extension is None:
        extension = extension_phases()

    X = X0.copy()
    current_value, _ = cost_function(X)
    value_history = [current_value]

    X, current_value, g_last = run_phase_list(
        group=group,
        X=X,
        current_value=current_value,
        phases=phases,
        value_history=value_history,
        entry_tol=entry_tol,
        min_grad_norm=min_grad_norm,
        min_stepsize=min_stepsize,
        shrink_factor=shrink_factor,
        max_backtracks=max_backtracks,
        armijo_c1=armijo_c1,
        verbose=verbose,
        print_every=print_every,
    )

    g_last = riemannian_gradient(group, X)
    max_abs_g = max_abs_entry(g_last)

    if verbose:
        print(
            f"\n[post-check] final max|grad_ij| = {max_abs_g:.6e} "
            f"(threshold = {extension_threshold:.1e})",
            flush=True,
        )

    if max_abs_g >= extension_threshold:
        if verbose:
            print("[post-check] running one extension pass", flush=True)

        X, current_value, g_last = run_phase_list(
            group=group,
            X=X,
            current_value=current_value,
            phases=extension,
            value_history=value_history,
            entry_tol=entry_tol,
            min_grad_norm=min_grad_norm,
            min_stepsize=min_stepsize,
            shrink_factor=shrink_factor,
            max_backtracks=max_backtracks,
            armijo_c1=armijo_c1,
            verbose=verbose,
            print_every=print_every,
        )

        g_last = riemannian_gradient(group, X)

    elif verbose:
        print("[post-check] no extension pass needed", flush=True)

    return X, value_history, g_last


def run_single_descent(n, seed=None, max_random_tries=5, verbose=False):
    """Run one descent in dimension 2n."""
    if n <= 0:
        raise ValueError("n must be positive.")

    if seed is not None:
        np.random.seed(seed)

    dim = 2 * n
    group = SpecialLinearGroup(dim)

    X0, initial_minus_ehz, tries = find_good_random_point(
        group,
        max_tries=max_random_tries,
    )

    X_opt, minus_ehz_history, grad_last = gradient_descent_with_phases(
        group,
        X0,
        verbose=verbose,
        print_every=200,
    )

    final_minus_ehz, _ = cost_function(X_opt)
    grad_last = riemannian_gradient(group, X_opt)
    grad_norm_maxabs = max_abs_entry(grad_last)

    result = {
        "n": n,
        "dimension": dim,
        "seed": seed,
        "tries_for_start": tries,
        "initial_minus_ehz": initial_minus_ehz,
        "final_minus_ehz": final_minus_ehz,
        "Q0": X0,
        "Q_opt": X_opt,
        "minus_ehz_history": minus_ehz_history,
        "gradient_last": grad_last,
        "gradient_norm_maxabs": grad_norm_maxabs,
        "det_Q_opt": float(np.linalg.det(X_opt)),
    }

    print("\n===== Final summary =====")
    print(f"n                    = {result['n']}")
    print(f"dimension            = {result['dimension']}")
    print(f"tries for start      = {result['tries_for_start']}")
    print(f"initial -ehz         = {result['initial_minus_ehz']:.10e}")
    print(f"final -ehz           = {result['final_minus_ehz']:.10e}")
    print(f"det(Q_opt)           = {result['det_Q_opt']:.16e}")
    print(f"max|grad_last_ij|    = {result['gradient_norm_maxabs']:.10e}")

    return result


def main():
    """Console entry point."""
    print("Gradient descent on SL(2n,R)\n")

    n_in = input("Enter n (matrix size 2n x 2n): ").strip()
    if not n_in:
        raise ValueError("You must enter a positive integer n.")
    n = int(n_in)
    if n <= 0:
        raise ValueError("n must be positive.")

    seed_in = input("Seed [press Enter for random]: ").strip()
    seed = int(seed_in) if seed_in else None

    verbose_in = input("Verbose output? [y/N]: ").strip().lower()
    verbose = verbose_in in {"y", "yes"}

    run_single_descent(
        n=n,
        seed=seed,
        max_random_tries=5,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()