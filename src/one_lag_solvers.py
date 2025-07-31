import cvxpy as cp
from objective_function import *
from utils import *
import math


def solve_one_lag(X, Z, gamma_1, gamma_2, gamma_3, P):
    """
    Solve the convex Laplacian-learning problem for multiple time steps.

    For each time step, solves for a sequence of Laplacian matrices and a shared
    regressor vector, minimizing a multi-term objective including Dirichlet energy,
    log barrier, Frobenius norm, prediction error, and temporal smoothness.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (N, >= M), where N is the number of nodes and
        M is the number of time steps (must match Z and P), most recent first.
    Z : list of np.ndarray
        List of M Laplacian-related distance matrices, each of shape (N, N). Most recent first
    gamma_1 : float
        Weight for log-barrier term.
    gamma_2 : float
        Weight for Frobenius norm regularization.
    gamma_3 : float
        Weight for temporal smoothness term between consecutive Laplacians.
    P : list of np.ndarray
        List of M regressor matrices, each of shape (N, J), most recent first.

    Returns
    -------
    tuple
        (list of pruned Laplacian matrices [np.ndarray, ...], regressor weights np.ndarray) (most recent first)
    """
    N = X.shape[0]
    J = P[0].shape[1]
    M = len(P)

    L_vars = [cp.Variable((N, N), symmetric=True) for _ in range(M)]
    b = cp.Variable((J, M))

    Z_const = [cp.Constant(Z[i]) for i in range(M)]
    x_prevs = [cp.Constant(X[:, i]) for i in range(M)]
    P_c = [cp.Constant(P[i]) for i in range(M)]


    # Constraints
    constraints = []
    for L in L_vars:
        constraints.append(cp.diag(L) >= 1e-6)
        constraints.append(L @ np.ones(N) == 0)   # rows sum to zero
        # Off-diagonals nonpositive
        for i in range(N):
            for j in range(N):
                if i != j:
                    constraints.append(L[i, j] <= 0)
        constraints.append(L >> 0)

    # Dirichlet energy
    E_dir = cp.sum([-cp.trace(L_vars[i] @ Z_const[i]) for i in range(M)])

    # Log-barrier summed over all slices
    E_log = cp.sum([cp.sum(cp.log(cp.diag(L))) for L in L_vars])

    # Squared Frobenius norm summed over all slices
    E_frob = cp.sum([cp.sum_squares(L) for L in L_vars])

    adjacencies = [-L + cp.diag(cp.diag(L)) for L in L_vars]
    degrees = [cp.sum(A, axis=1) for A in adjacencies]

    errors = [
        cp.reshape((adjacencies[i+1]) @ x_prevs[i + 1], (N, 1), order = "C")
        for i in range(M - 1)
    ]
    E = cp.hstack(errors)  # Shape (N, M-1)

    pred_terms = [
        cp.reshape(P_c[i] @ b[:, i], (N, 1), order="C")
        for i in range(M)
    ]
    last_col = cp.reshape(x_prevs[-1], (N,1), order="C")
    s_hats   = cp.hstack(pred_terms) + cp.hstack([E, last_col])
    x_cols = [cp.reshape(x_prevs[i], (N, 1), order = "C") for i in range(M)]
    s_true = cp.hstack(x_cols)  # Shape (N, M)
    E_pred = cp.norm(s_true - s_hats, 'fro')

    # Frobenius norm for temporal smoothness
    E_time = cp.sum([cp.norm(L_vars[i] - L_vars[i + 1], 'fro') for i in range(M - 1)])

    # Objective
    obj = cp.Minimize(
        E_dir
        - gamma_1 * E_log
        + gamma_2 * E_frob
        + E_pred
        + gamma_3 * E_time
    )
    #print("Maximize(square(x)) is DCP:", obj.is_dcp())

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    return [prune(L.value, 0.00001) for L in L_vars], b.value

