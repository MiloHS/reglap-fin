import numpy as np

def prune(matrix, threshold):
    """
    Set elements of the matrix with absolute value below a threshold to zero.

    Parameters
    ----------
    matrix : array_like
        Input matrix to prune.
    threshold : float
        Values with absolute value less than this threshold are set to zero.

    Returns
    -------
    np.ndarray
        Matrix after pruning small values.
    """
    temp_array = np.array(matrix, copy=True)
    for i in range(temp_array.shape[0]):
        for j in range(temp_array.shape[1]):
            if abs(temp_array[i, j]) < threshold:
                temp_array[i, j] = 0
    return temp_array


def degree_mat(matrix):
    """
    Compute the degree matrix from a Laplacian or adjacency matrix.

    The degree matrix is diagonal, where each diagonal element is the sum of
    the absolute values of the non-diagonal elements in the corresponding row.

    Parameters
    ----------
    matrix : np.ndarray
        Input Laplacian or adjacency matrix (square).

    Returns
    -------
    np.ndarray
        Degree matrix (diagonal matrix).
    """
    n, m = matrix.shape
    degree = np.zeros((n, n))
    for i in range(n):
        row_sum = 0
        for j in range(m):
            if i != j:
                row_sum += matrix[i, j]
        degree[i, i] = abs(row_sum)
    return degree


def distance_matrix_unweighted_corr(X):
    """
    Compute a distance matrix from unweighted sample correlations.

    Parameters
    ----------
    X : np.ndarray
        2D array where each row represents a signal.

    Returns
    -------
    np.ndarray
        Distance matrix Z computed from unweighted correlations.
        Z[i, j] = 1 / (|corr[i, j]| + 0.01) - 1 / 1.01, with Z[i, i] = 0.
    """
    corr = np.corrcoef(X)
    Z = 1 / (np.abs(corr) + 0.01) - 1 / 1.01
    np.fill_diagonal(Z, 0)
    return Z


def distance_matrix_weighted_corr(X, w):
    """
    Compute a distance matrix from weighted sample correlations.

    Parameters
    ----------
    X : np.ndarray
        2D array where each row represents a signal.
    w : np.ndarray or array_like
        1D array of weights for weighted covariance calculation.

    Returns
    -------
    np.ndarray
        Distance matrix Z computed from weighted correlations.
        Z[i, j] = 1 / (|R[i, j]| + 0.01), with Z[i, i] = 0.
    """
    C = np.cov(X, aweights=w)
    stds = np.sqrt(np.diag(C))
    inv_stds = 1.0 / stds
    R = C * np.outer(inv_stds, inv_stds)
    Z = 1.0 / (np.abs(R) + 0.01)
    np.fill_diagonal(Z, 0)
    return Z


