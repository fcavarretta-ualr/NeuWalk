import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.stats import brunnermunzel, fligner

# -------------------------
# Bootstrap p-value helpers
# -------------------------
def bootstrap_pvalue_mean_diff(x, y, nx=None, ny=None, B=10000, seed=42):
    """
    Bootstrap p-value for H0: mean_x = mean_y using a null-imposed (shift) bootstrap.
    Two-sided p-value on the mean difference.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        raise ValueError("Each group needs at least 2 observations.")

    # Observed statistic
    obs = np.mean(x) - np.mean(y)

    # Impose H0 by aligning both groups to the pooled mean
    pooled_mean = np.mean(np.concatenate([x, y]))
    x0 = x - np.mean(x) + pooled_mean
    y0 = y - np.mean(y) + pooled_mean

    # Bootstrap under H0
    if nx is None:
        nx = x.size
    if ny is None:
        ny = y.size
    stats = np.empty(B)
    for b in range(B):
        bx = rng.choice(x0, size=nx, replace=True)
        by = rng.choice(y0, size=ny, replace=True)
        stats[b] = np.mean(bx) - np.mean(by)

    # Two-sided p with +1 pseudo-count to avoid zero
    p = (np.sum(np.abs(stats) >= abs(obs)) + 1) / (B + 1)
    return float(p), float(obs), stats  # stats is the bootstrap null distribution


def bootstrap_pvalue_var_ratio(x, y, nx=None, ny=None, B=10000, seed=42):
    """
    Bootstrap p-value for H0: var_x = var_y using a null-imposed (scale) bootstrap.
    Two-sided p-value on the log variance ratio for symmetry.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        raise ValueError("Each group needs at least 2 observations.")

    # Observed (use log-ratio for symmetric tails)
    vx = np.var(x, ddof=1); vy = np.var(y, ddof=1)
    obs_ratio = vx / vy if vy > 0 else np.inf
    obs = np.log(obs_ratio)

    # Impose H0: standardize to unit variance (centering optional)
    x0 = (x - np.mean(x)) / np.sqrt(vx)
    y0 = (y - np.mean(y)) / np.sqrt(vy)

    # Bootstrap under H0
    if nx is None:
        nx = x.size
    if ny is None:
        ny = y.size
    stats = np.empty(B)
    for b in range(B):
        bx = rng.choice(x0, size=nx, replace=True)
        by = rng.choice(y0, size=ny, replace=True)
        vr = np.var(bx, ddof=1) / np.var(by, ddof=1)
        stats[b] = np.log(vr)

    p = (np.sum(np.abs(stats) >= abs(obs)) + 1) / (B + 1)
    return float(p), float(obs_ratio), stats  # stats is log-ratio under H0



def bootstrap_test_vs_theor(data, ref_mean=None, ref_var=None, 
                            n_iter=10000, random_state=None):
    """
    Bootstrap test for mean and variance against reference values.

    Parameters
    ----------
    data : array-like
        Experimental data (list or numpy array).
    ref_mean : float or None
        Theoretical/reference mean to test against (if None, mean test skipped).
    ref_var : float or None
        Theoretical/reference variance to test against (if None, variance test skipped).
    n_iter : int
        Number of bootstrap iterations (default: 10000).
    random_state : int or None
        For reproducibility.

    Returns
    -------
    dict with bootstrap estimates, confidence intervals, and p-values.
    """
    rng = np.random.default_rng(random_state)
    data = np.asarray(data)
    n = len(data)

    # Bootstrap resampling
    boot_means = np.empty(n_iter)
    boot_vars = np.empty(n_iter)
    for i in range(n_iter):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)
        boot_vars[i] = np.var(sample, ddof=1)  # unbiased sample variance

    results = {}

    # Mean test
    if ref_mean is not None:
        p_mean = np.mean(np.abs(boot_means - np.mean(data)) 
                         >= np.abs(ref_mean - np.mean(data)))
        mean_ci = np.percentile(boot_means, [2.5, 97.5])
        results["mean"] = {
            "estimate": np.mean(boot_means),
            "95%_CI": mean_ci,
            "p_value": p_mean
        }

    # Variance test
    if ref_var is not None:
        p_var = np.mean(np.abs(boot_vars - np.var(data, ddof=1)) 
                        >= np.abs(ref_var - np.var(data, ddof=1)))
        var_ci = np.percentile(boot_vars, [2.5, 97.5])
        results["variance"] = {
            "estimate": np.mean(boot_vars),
            "95%_CI": var_ci,
            "p_value": p_var
        }

    return results

# -------------------------
# Demo with synthetic data
# -------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # Example 1: Slight mean shift, similar variances
    x = rng.normal(loc=0.00, scale=1.0, size=100)
    y = rng.normal(loc=0.00, scale=55.0, size=8)

    print("Sample sizes:", len(x), len(y))
    print("Observed mean diff:", np.mean(x) - np.mean(y))
    print("Observed var ratio:", np.var(x, ddof=1) / np.var(y, ddof=1))

    # --- Bootstrap p-values ---
    p_mean, obs_mean_diff, _ = bootstrap_pvalue_mean_diff(x, y, B=5000, seed=123)
    p_var, obs_var_ratio, _ = bootstrap_pvalue_var_ratio(x, y, B=5000, seed=123)

    print("\n[Bootstrap p-values under imposed null]")
    print(f"Mean difference: p = {p_mean:.4f} (obs = {obs_mean_diff:.4f})")
    print(f"Variance ratio:  p = {p_var:.4f} (obs ratio = {obs_var_ratio:.4f})")


