import numpy as np

"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""


def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    ....
    """
    pass

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI
    
    ....
    """
    pass

def R_squared(X, y):
    """
    Calculate R-squared from multiple linear regression.

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)

    Returns
    -------
    float
        R-squared value (between 0 and 1) from OLS
    
    Raises
    ------
    TypeError
        If not isinstance(X, np.ndarray)
        If not isinstance(y, np.ndarray)

    ValueError
        If X.ndim != 2
        If y.ndim != 1
        If X.shape[0] != len(y)
        If all entries of y are equal

    LinAlgError
        If X.T @ X
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    if X.shape[0] != len(y):
        raise ValueError("X.shape[0] must equal len(y)")

    y_mean = np.mean(y)
    if np.allclose(y, y_mean): # Should probably set atol or rtol here
        raise ValueError("All entries of y are equal.")

    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ beta_hat
    
    return 1 - (y - y_hat) ** 2 / (y - y_mean) ** 2
