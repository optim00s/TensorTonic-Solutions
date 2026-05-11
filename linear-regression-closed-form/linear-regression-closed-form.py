import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    pass
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    # Solve (XᵀX)w = Xᵀy directly; avoids explicit matrix inversion and improves numerical stability
    return np.linalg.solve(X.T @ X, X.T @ y)
    