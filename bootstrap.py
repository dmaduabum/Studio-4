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
        Design matrix (must contain numeric data)
    y : array-like, shape (n,)
        Response vector (must contain numeric data)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    Raises
    ------
    TypeError
        If X or y are not array-like
        If X or y contain non-numeric data
    ValueError
        If X and y have incompatible shapes
        If n_bootstrap is not positive
        If X or y are empty
    """
    # Convert to numpy arrays
    X = np.asarray(X) 
    y = np.asarray(y)
    n = len(y)
    
    # Input validation
    if n == 0:
        raise ValueError("y cannot be empty")
    if X.size == 0:
        raise ValueError("X cannot be empty")
    
    # Check for numeric data types
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X must contain numeric data")
    if not np.issubdtype(y.dtype, np.number):
        raise TypeError("y must contain numeric data")
    
    if X.shape[0] != n:
        raise ValueError("X and y must have the same number of observations")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        # Compute statistic on bootstrap sample
        bootstrap_stats[i] = compute_stat(X_boot, y_boot)
    
    return bootstrap_stats


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
    
    Raises
    ------
    TypeError
        If not isinstance(bootstrap_stats, np.ndarray)
        If not isinstance(alpha, float)

    ValueError
        If bootstrap_stats.ndim != 1
        If alpha not in (0, 1)
    """
    if not isinstance(bootstrap_stats, np.ndarray):
        raise TypeError("bootstrap_stats must be a numpy array.")
    if not isinstance(alpha, float):
        raise TypeError("alpha must be a float.")
    
    if bootstrap_stats.ndim != 1:
        raise ValueError("bootstrap_stats must be 1D.")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")

    return np.quantile(bootstrap_stats, [alpha / 2, 1 - alpha / 2])

def r_squared(X, y):
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
        If X.T @ X is singular
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
    
    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - y_mean) ** 2)

    return 1 - rss / tss
