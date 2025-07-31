import numpy as np
from utils import *

def dirichlet_energy(L, Z):
    """
    Compute the Dirichlet energy, which measures smoothness.

    The energy is computed as the sum of the elementwise product of the input
    Laplacian (or adjacency) matrix L and a distance matrix Z, with diagonals
    of L set to zero.

    Parameters
    ----------
    L : np.ndarray
        Laplacian or adjacency matrix (square).
    Z : np.ndarray
        Distance matrix, same shape as L.

    Returns
    -------
    float
        Absolute value of the Dirichlet energy.
    """
    W = L.copy()
    np.fill_diagonal(W, 0)
    mat = W * Z
    smooth = np.sum(mat)
    return abs(smooth)


def log_term(L):
    """
    Return the sum of the logarithms of the diagonal elements of L.

    Used to penalize disconnected nodes in the Laplacian.

    Parameters
    ----------
    L : np.ndarray
        Laplacian matrix (square).

    Returns
    -------
    float
        Sum of logarithms of diagonal elements.
    """
    diag = np.diag(L)
    return np.sum(np.log(diag))


def mat_dist(L, L_prev):
    """
    Compute the spectral norm (operator 2-norm) distance between two matrices.

    Parameters
    ----------
    L : np.ndarray
        First matrix.
    L_prev : np.ndarray
        Second matrix, same shape as L.

    Returns
    -------
    float
        Spectral norm of (L - L_prev).
    """
    return np.linalg.norm(L - L_prev, ord=2)


def s_hat_one(A, D, X, c_1, c_2, b, P):
    """
    Estimates the next signal using polynomial filters on the adjacency and degree matrices.

    Parameters:
    A: Adjacency matrix (numpy array)
    D: Degree matrix (numpy array)
    X: N x M matrix of previous signals (N = nodes, M = past observations)
    c_1, c_2: Polynomial coefficient vectors
    b: regressor weight vector
    P: regressor matrix

    Returns:
    Estimated next signal as a 1-D numpy array.
    """
    N, M = X.shape
    x_k = P @ b
    j = 0
    for i in range(M):
        p = np.zeros((N, N))
        for k in range(i+2):
            p += c_1[j] * np.linalg.matrix_power(A, k) + c_2[j] * np.linalg.matrix_power(D, k)
            j += 1
        x_k += p @ X[:, i]
    return x_k


def s_hat_two(A, X, P, b):
    """
    First-order signal prediction: x_k = (A+D) x_prev + P b

    Parameters:
    A: Adjacency matrix (numpy array)
    X: N x M matrix of signals
    P: regressor matrix
    b: regressor weight vector

    Returns:
    Estimated next signal as a 1-D numpy array.
    """
    D = np.diag(np.sum(A, axis=1))
    x_k = (A + D) @ X[:, 1] + P @ b
    return x_k


def objective_function(
    L, L_prev, X, gamma_1, gamma_2, gamma_3, b, P
):
    """
    Compute the objective function for Laplacian learning with graph signals.

    The objective is:
        D(L, Z) - gamma_1 * sum_i log L_ii
        + gamma_2 * ||L||_F^2
        + ||s_true - s_hat||_2
        + gamma_3 * d(L - L_prev)

    Parameters
    ----------
    L : np.ndarray
        Current estimated Laplacian matrix (N x N).
    L_prev : np.ndarray
        Previous Laplacian matrix (N x N).
    X : np.ndarray
        Data matrix; X[:, 0] is s_true (the current observed signal).
    gamma_1 : float
        Weight for log barrier term (prevents disconnected nodes).
    gamma_2 : float
        Weight for Frobenius norm penalty (regularization).
    gamma_3 : float
        Weight for Laplacian distance term (smoothness over iterations).
    b : np.ndarray
        Regressor coefficient vector.
    P : np.ndarray
        Regressor matrix.

    Returns
    -------
    float
        Value of the objective function.
    """
    Z = distance_matrix_unweighted_corr(X)
    s_true = X[:, 0]
    dirichlet = dirichlet_energy(L, Z)
    log_barrier = log_term(L)
    frob = np.linalg.norm(L, ord='fro') ** 2
    N, _ = L.shape
    I = np.identity(N)
    J = np.ones((N, N))
    # Recompute adjacency for s_hat_two by zeroing diagonal and taking abs
    s_hat = s_hat_two(np.abs((I - J) * L), X, P, b)
    dist = mat_dist(L, L_prev)
    return (
        dirichlet
        - gamma_1 * log_barrier
        + gamma_2 * frob
        + np.linalg.norm(s_true - s_hat)
        + gamma_3 * dist
    )