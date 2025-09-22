import pytest
import numpy as np
from numpy.linalg import LinAlgError
from bootstrap import bootstrap_sample, bootstrap_ci, r_squared


def test_bootstrap_integration():
    """
    Integration test that uses all three functions together.
    Tests the complete workflow from data generation to confidence intervals.
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate synthetic data with a linear relationship
    n = 100
    p = 3
    
    # Design matrix with intercept
    X = np.column_stack([np.ones(n), np.random.normal(0, 1, (n, p))])
    
    # True coefficients
    true_beta = np.array([1, 0.5, -0.3, 0.2])
    
    # Generate response with noise
    y = X @ true_beta + np.random.normal(0, 0.5, n)
    
    # Compute R-squared on original data
    r2_original = r_squared(X, y)
    assert 0 <= r2_original <= 1, "R-squared should be between 0 and 1"
    
    # Bootstrap the R-squared statistic
    n_bootstrap = 500
    bootstrap_stats = bootstrap_sample(X, y, r_squared, n_bootstrap)
    
    # Validate bootstrap output
    assert len(bootstrap_stats) == n_bootstrap
    assert np.all(bootstrap_stats >= 0) and np.all(bootstrap_stats <= 1)
    
    # Compute confidence interval
    ci = bootstrap_ci(bootstrap_stats, alpha=0.05)
    
    # Validate CI properties
    assert len(ci) == 2
    assert ci[0] <= ci[1]
    assert 0 <= ci[0] <= 1 and 0 <= ci[1] <= 1
    
    # Statistical validation: original R-squared should be within CI
    assert ci[0] <= r2_original <= ci[1]


def test_r_squared_happy_path():
    """Happy path tests for r_squared function"""
    # Perfect correlation
    X = np.column_stack([np.ones(5), [1, 2, 3, 4, 5]])
    y = np.array([2, 4, 6, 8, 10])
    result = r_squared(X, y)
    assert abs(result - 1.0) < 1e-10
    
    # Positive correlation
    X = np.column_stack([np.ones(5), [1, 2, 3, 4, 5]])
    y = np.array([1, 2, 3, 4, 5])
    result = r_squared(X, y)
    assert 0 <= result <= 1
    
    # No correlation (just intercept)
    X = np.column_stack([np.ones(5), np.random.normal(0, 1, 5)])
    y = np.array([1, 1, 1, 1, 1]) + np.random.normal(0, 0.1, 5)
    result = r_squared(X, y)
    assert 0 <= result <= 1


def test_r_squared_edge_cases():
    """Edge case tests for r_squared function"""
    # Single observation
    X = np.array([[1, 1]])
    y = np.array([2])
    with pytest.raises(ValueError, match="All entries of y are equal"):
        r_squared(X, y)
    
    # Constant y values
    X = np.column_stack([np.ones(3), [1, 2, 3]])
    y_constant = np.array([5, 5, 5])
    with pytest.raises(ValueError, match="All entries of y are equal"):
        r_squared(X, y_constant)


def test_r_squared_invalid_inputs():
    """Invalid input tests for r_squared function"""
    # Wrong dimensions
    with pytest.raises(ValueError, match="X must be 2D"):
        r_squared(np.array([1, 2, 3]), np.array([1, 2, 3]))
    
    with pytest.raises(ValueError, match="y must be 1D"):
        r_squared(np.ones((3, 2)), np.ones((3, 1)))
    
    # Mismatched lengths
    X = np.column_stack([np.ones(3), [1, 2, 3]])
    y_wrong_length = np.array([1, 2])
    with pytest.raises(ValueError, match="must equal len"):
        r_squared(X, y_wrong_length)
    
    # Singular matrix
    X_singular = np.column_stack([np.ones(3), [1, 2, 3], [2, 4, 6]])
    y = np.array([1, 2, 3])
    with pytest.raises(LinAlgError):
        r_squared(X_singular, y)


def test_r_squared_statistical_validation():
    """Statistical validation tests for r_squared"""
    # R-squared should be 1 for perfect linear relationship
    X = np.column_stack([np.ones(10), np.arange(10)])
    y = 3 + 2 * np.arange(10)  # y = 3 + 2x
    assert abs(r_squared(X, y) - 1.0) < 1e-10
    
    # R-squared should be low for uncorrelated data
    np.random.seed(42)
    X = np.column_stack([np.ones(100), np.random.normal(0, 1, 100)])
    y = np.random.normal(0, 1, 100)
    r2 = r_squared(X, y)
    assert 0 <= r2 < 0.5  # Should be low for random data


def test_bootstrap_sample_happy_path():
    """Happy path tests for bootstrap_sample"""
    np.random.seed(42)
    
    X = np.column_stack([np.ones(10), np.random.normal(0, 1, 10)])
    y = np.random.normal(0, 1, 10)
    
    def mean_stat(X, y):
        return np.mean(y)
    
    # Basic functionality
    stats = bootstrap_sample(X, y, mean_stat, n_bootstrap=100)
    assert len(stats) == 100
    assert isinstance(stats, np.ndarray)
    
    # Custom statistic function
    def custom_stat(X, y):
        return np.std(y)
    
    stats = bootstrap_sample(X, y, custom_stat, n_bootstrap=50)
    assert len(stats) == 50
    assert np.all(stats >= 0)


def test_bootstrap_sample_edge_cases():
    """Edge case tests for bootstrap_sample"""
    X = np.column_stack([np.ones(3), [1, 2, 3]])
    y = np.array([1, 2, 3])
    
    def simple_stat(X, y):
        return len(y)
    
    # Small sample size
    stats = bootstrap_sample(X, y, simple_stat, n_bootstrap=10)
    assert len(stats) == 10
    assert np.all(stats == 3)  # All bootstrap samples should have 3 observations


def test_bootstrap_sample_invalid_inputs():
    """Invalid input tests for bootstrap_sample"""
    X = np.column_stack([np.ones(3), [1, 2, 3]])
    y = np.array([1, 2, 3])
    
    def simple_stat(X, y):
        return len(y)
    
    # Invalid n_bootstrap
    with pytest.raises(ValueError, match="n_bootstrap must be positive"):
        bootstrap_sample(X, y, simple_stat, n_bootstrap=0)
    
    # Mismatched X and y
    with pytest.raises(ValueError, match="same number of observations"):
        bootstrap_sample(X, np.array([1, 2]), simple_stat)
    
    # Invalid statistic function
    def bad_stat(X, y):
        raise ValueError("Statistic computation failed")
    
    with pytest.raises(ValueError, match="Statistic computation failed"):
        bootstrap_sample(X, y, bad_stat, n_bootstrap=10)


def test_bootstrap_ci_happy_path():
    """Happy path tests for bootstrap_ci"""
    np.random.seed(42)
    
    # Normal distribution
    stats = np.random.normal(0, 1, 1000)
    ci = bootstrap_ci(stats, alpha=0.05)
    assert len(ci) == 2
    assert ci[0] < ci[1]
    
    # Different alpha values
    ci_90 = bootstrap_ci(stats, alpha=0.1)
    ci_99 = bootstrap_ci(stats, alpha=0.01)
    assert ci_90[0] > ci_99[0] and ci_90[1] < ci_99[1]  # 90% CI narrower than 99%


def test_bootstrap_ci_edge_cases():
    """Edge case tests for bootstrap_ci"""
    # Single value (degenerate case)
    stats = np.array([5.0])
    ci = bootstrap_ci(stats, alpha=0.05)
    assert ci[0] == ci[1] == 5.0
    
    # Two values
    stats = np.array([1.0, 2.0])
    ci = bootstrap_ci(stats, alpha=0.05)
    assert ci[0] == 1.0 and ci[1] == 2.0


def test_bootstrap_ci_invalid_inputs():
    """Invalid input tests for bootstrap_ci"""
    # Not numpy array
    with pytest.raises(TypeError):
        bootstrap_ci([1, 2, 3])
    
    # Wrong dimensions
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([[1, 2], [3, 4]]))
    
    # Invalid alpha
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([1, 2, 3]), alpha=1.5)
    
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([1, 2, 3]), alpha=0.0)
    
    # Empty array
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]))


def test_bootstrap_ci_statistical_validation():
    """Statistical validation tests for bootstrap_ci"""
    np.random.seed(42)
    
    # For large normal sample, CI should contain true mean
    stats = np.random.normal(5, 2, 10000)  # True mean = 5
    ci = bootstrap_ci(stats, alpha=0.05)
    assert ci[0] < 5 < ci[1]  # CI should contain true mean
    
    # CI should get wider with smaller alpha
    ci_95 = bootstrap_ci(stats, alpha=0.05)
    ci_99 = bootstrap_ci(stats, alpha=0.01)
    assert ci_99[0] < ci_95[0] and ci_99[1] > ci_95[1]