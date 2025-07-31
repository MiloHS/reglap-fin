import numpy as np
import pyomo.environ as pyo
import math
from pyomo.environ import quicksum, NonNegativeIntegers



def k_lag_graph_learning_model(
    X_data,
    P_data,
    Z_data,
    gamma1,
    gamma2,
    gamma3,
    k,
):
    """
    Learn time-varying graph Laplacians with lagged regressors.

    Minimizes:
        sum_{t=0}^{M-1}[
            -tr(L[t] Z[t])
            - gamma1 * sum_i log(L[t]_ii)
            + gamma2 * ||L[t]||_F^2
        ]
        + sum_{t=1}^{M-1}||x[t] - x_hat[t]||_2^2
        + gamma3 * sum_{t=1}^{M-1}||L[t] - L[t-1]||_F^2

    where:
        x_hat[t] = R[t] beta[t]
                 + sum_{lag=1}^{min(k,t)} sum_{p=0}^lag(
                       c1[lag,p] * A[t-lag]^p
                     + c2[lag,p] * D[t-lag]^p
                   ) x[t-lag]

    Constraints on L[t]:
      - row sums = 0
      - symmetric
      - off-diagonals <= 0
      - diagonals >= eps (1e-8)

    Parameters
    ----------
    X_data : array_like, shape (N, M)
        Observations over time (columns oldest to newest).
    P_data : array_like, shape (M, N, J)
        Time-varying regressors.
    Z_data : array_like, shape (M, N, N)
        Data for Laplacian trace term.
    gamma1 : float
    gamma2 : float
    gamma3 : float
    k      : int
        Maximum lag for x_hat computation.

    Returns
    -------
    model : pyo.ConcreteModel
    """
    # Convert inputs
    X = np.asarray(X_data, dtype=float)
    P = np.asarray(P_data, dtype=float)
    Z = np.asarray(Z_data, dtype=float)

    N, M = X.shape
    assert P.shape[:2] == (M, N), "P_data must have shape (M, N, J)"
    J = P.shape[2]
    assert Z.shape == (M, N, N), "Z_data must have shape (M, N, N)"
    assert M >= 2, "Need at least two time points"

    model = pyo.ConcreteModel()

    # Index sets
    model.N = pyo.RangeSet(0, N - 1)
    model.T = pyo.RangeSet(0, M - 1)
    model.T_pred = pyo.RangeSet(1, M - 1)
    model.T_pow = pyo.RangeSet(0, M - 2)
    model.J = pyo.RangeSet(0, J - 1)

    # Map (lag, power) to flat index
    lag_pow_map = {}
    idx = 0
    for lag in range(1, M):
        for p in range(lag + 1):
            lag_pow_map[(lag, p)] = idx
            idx += 1
    model.lag_pow_map = lag_pow_map
    model.C = pyo.RangeSet(0, idx - 1)

    # Max power per source time
    model.max_pow = pyo.Param(
        model.T_pow,
        initialize=lambda m, t: min(k, (M - 1) - t),
        within=pyo.NonNegativeIntegers,
        mutable=False,
    )
    model.TP_pow = pyo.Set(
        dimen=2,
        initialize=lambda m: (
            (t, p) for t in m.T_pow for p in range(int(m.max_pow[t]) + 1)
        ),
    )

    # Variables
    model.L = pyo.Var(model.T, model.N, model.N, domain=pyo.Reals)
    for t in model.T:
        for i in model.N:
            model.L[t, i, i].setlb(1e-8)

    model.c1 = pyo.Var(model.C, domain=pyo.Reals)
    model.c2 = pyo.Var(model.C, domain=pyo.Reals)
    model.beta = pyo.Var(model.J, model.T_pred, domain=pyo.Reals)

    # Parameters
    model.X = pyo.Param(
        model.N, model.T,
        initialize=lambda m, i, t: float(X[i, t]),
        mutable=False,
    )
    model.P = pyo.Param(
        model.T, model.N, model.J,
        initialize=lambda m, t, i, j: float(P[t, i, j]),
        mutable=False,
    )
    model.Z = pyo.Param(
        model.T, model.N, model.N,
        initialize=lambda m, t, i, j: float(Z[t, i, j]),
        mutable=False,
    )

    # Adjacency and degree expressions
    model.A = pyo.Expression(
        model.T, model.N, model.N,
        rule=lambda m, t, i, j: 0.0 if i == j else -m.L[t, i, j]
    )
    model.D = pyo.Expression(
        model.T, model.N, model.N,
        rule=lambda m, t, i, j: sum(m.A[t, i, r] for r in m.N) if i == j else 0.0
    )

    # Powers of A and D
    model.A_pow = pyo.Expression(
        model.TP_pow, model.N, model.N,
        rule=lambda m, s, p, i, j:
            1.0 if p == 0 and i == j else
            0.0 if p == 0 else
            sum(m.A[s, i, r] * m.A_pow[s, p - 1, r, j] for r in m.N)
    )
    model.D_pow = pyo.Expression(
        model.TP_pow, model.N, model.N,
        rule=lambda m, s, p, i, j:
            1.0 if p == 0 and i == j else
            0.0 if p == 0 else
            sum(m.D[s, i, r] * m.D_pow[s, p - 1, r, j] for r in m.N)
    )

    # True and predicted signals
    model.s_true = pyo.Expression(
        model.N, model.T_pred,
        rule=lambda m, i, t: m.X[i, t]
    )

    def s_hat_rule(m, i, t):
        expr = sum(m.P[t, i, j] * m.beta[j, t] for j in m.J)
        for lag in range(1, min(k, t) + 1):
            s = t - lag
            for p in range(lag + 1):
                idx = m.lag_pow_map[(lag, p)]
                expr += m.c1[idx] * sum(m.A_pow[s, p, i, r] * m.X[r, s] for r in m.N)
                expr += m.c2[idx] * sum(m.D_pow[s, p, i, r] * m.X[r, s] for r in m.N)
        return expr

    model.s_hat = pyo.Expression(
        model.N, model.T_pred, rule=s_hat_rule
    )

    # Objective function
    def obj_rule(m):
        trace = -sum(m.L[t, i, j] * m.Z[t, j, i] for t in m.T for i in m.N for j in m.N)
        log_term = -gamma1 * sum(pyo.log(m.L[t, i, i]) for t in m.T for i in m.N)
        frob = gamma2 * sum(m.L[t, i, j]**2 for t in m.T for i in m.N for j in m.N)
        pred = sum((m.s_true[i, t] - m.s_hat[i, t])**2 for t in m.T_pred for i in m.N)
        smooth = gamma3 * sum((m.L[t, i, j] - m.L[t - 1, i, j])**2
                               for t in m.T if t > 0 for i in m.N for j in m.N)
        return trace + log_term + frob + pred + smooth

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints
    model.zero_row = pyo.Constraint(
        model.T, model.N,
        rule=lambda m, t, i: sum(m.L[t, i, j] for j in m.N) == 0.0
    )
    model.symmetry = pyo.Constraint(
        model.T, model.N, model.N,
        rule=lambda m, t, i, j: m.L[t, i, j] == m.L[t, j, i] if i < j else pyo.Constraint.Skip
    )
    model.offdiag = pyo.Constraint(
        model.T, model.N, model.N,
        rule=lambda m, t, i, j: m.L[t, i, j] <= 0.0 if i != j else pyo.Constraint.Skip
    )

    return model    


def solve_k_lag(
    X,
    P,
    Z,
    k=2,
    gammas=(1e-3, 1e-1, 1e-3),
    ipopt_options=None,
):
    """
    Solve the k-lag graph learning model instance using IPOPT.

    Parameters
    ----------
    X : array_like, shape (N, M)
        Observations over time (columns oldest to newest).
    P : array_like, shape (M, N, J)
        Time-varying regressors.
    Z : array_like, shape (M, N, N)
        Data for Laplacian trace term.
    k : int, optional
        Maximum lag for x_hat computation. Default is 2.
    gammas : tuple of float, optional
        (gamma1, gamma2, gamma3) regularization weights.
    ipopt_options : dict, optional
        IPOPT solver options.

    Returns
    -------
    model : pyo.ConcreteModel
    results : SolverResults
    obj_val : float
    L : ndarray, shape (M, N, N)
    c1 : ndarray, shape (C,)
    c2 : ndarray, shape (C,)
    beta : ndarray, shape (J, M)
    s_hat : ndarray, shape (N, M)
    """
    gamma1, gamma2, gamma3 = gammas
    model = k_lag_graph_learning_model(
        X, P, Z, gamma1, gamma2, gamma3, k
    )

    # Initialize variables
    N, M = X.shape
    for t in model.T:
        for i in model.N:
            for j in model.N:
                if i == j:
                    model.L[t, i, i].value = 1.0
                elif i < j:
                    val = -0.05
                    model.L[t, i, j].value = val
                    model.L[t, j, i].value = val
        # Adjust diagonal to satisfy zero-row sum
        for i in model.N:
            off_sum = sum(
                model.L[t, i, j].value
                for j in model.N if j != i
            )
            model.L[t, i, i].value = max(-off_sum, 1e-6)

    for c in model.C:
        model.c1[c].value = 0.0
        model.c2[c].value = 0.0

    for t in model.T_pred:
        for j in model.J:
            model.beta[j, t].value = 0.0

    # Solve
    solver = pyo.SolverFactory('ipopt')
    if ipopt_options:
        solver.options.update(ipopt_options)
    results = solver.solve(model, tee=True)

    # Extract results
    obj_val = pyo.value(model.obj)

    L = np.array([
        [[pyo.value(model.L[t, i, j]) for j in model.N]
         for i in model.N]
        for t in model.T
    ])

    C = len(model.C)
    c1 = np.array([pyo.value(model.c1[c]) for c in model.C])
    c2 = np.array([pyo.value(model.c2[c]) for c in model.C])

    J = len(model.J)
    beta = np.zeros((J, M))
    for t in model.T_pred:
        for j in model.J:
            beta[j, t] = pyo.value(model.beta[j, t])

    s_hat = np.zeros((N, M))
    for t in model.T_pred:
        for i in model.N:
            s_hat[i, t] = pyo.value(model.s_hat[i, t])

    return model, results, obj_val, L, c1, c2, beta, s_hat
