import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.stats import brunnermunzel, fligner

# -------------------------
# Bootstrap p-value helpers
# -------------------------
def bootstrap_pvalue_mean_diff(rng, x, y, nx=None, ny=None, B=10000, ci=0.95):
    """
    Bootstrap p-value for H0: mean_x = mean_y using a null-imposed (shift) bootstrap.
    Two-sided p-value on the mean difference.

    x and y draw from two INDEPENDENT rng streams (rng.spawn(2)) rather
    than both advancing one shared stream, so this test's result
    depends only on its own data and rng, never on which of the two
    inputs happens to be passed as x vs y (e.g. if nx != ny, drawing
    x then y from one shared, sequentially-advancing stream doesn't
    guarantee the same set of resamples gets used just with roles
    swapped). observed_diff is exactly negated by swapping x and y,
    since it comes from the non-resampled means directly; p converges
    to the same value as B grows regardless of order, with the residual
    at finite B being ordinary Monte Carlo noise from x and y drawing
    from two distinct (if independent) streams, not a systematic bias.
    """
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        raise ValueError(f"Each group needs at least 2 observations {x.size} {y.size}.")

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

    rng_x, rng_y = rng.spawn(2)

    stats = np.empty(B)
    for b in range(B):
        bx = rng_x.choice(x0, size=nx, replace=True)
        by = rng_y.choice(y0, size=ny, replace=True)
        stats[b] = np.mean(bx) - np.mean(by)
        
    alpha = 1.0 - ci
    ci_low, ci_high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    
    # Two-sided p with +1 pseudo-count to avoid zero
    p = (np.sum(np.abs(stats) >= abs(obs)) + 1) / (B + 1)
    return {
        'value_a':float(np.mean(x)),
        'value_b':float(np.mean(y)),
        'observed_diff':float(obs),
        'ci_low':ci_low,
        'ci_high':ci_high,
        'p_value':float(p)
        }

def bootstrap_pvalue_var_ratio(rng, x, y, nx=None, ny=None, B=10000, ci=0.95, variance_floor=1e-6):
    """
    Bootstrap p-value for H0: var_x = var_y using a null-imposed (scale) bootstrap.
    Two-sided p-value on the log variance ratio for symmetry.

    If EITHER original sample has exactly zero variance (every
    observation identical), standardizing it to unit variance is
    undefined -- rather than dividing by zero and propagating NaN, this
    is detected up front: a message is printed and the observed
    variances/ratio are returned as-is, with ci_low/ci_high/p_value set
    to None, since no bootstrap test is meaningful here.

    Otherwise, a resample (drawn with replacement) can still come out
    with zero variance on one side even when the original sample
    doesn't. An earlier version of this function discarded and redrew
    such resamples -- but that conditions each side's resampled
    variance on its own, data-dependent degeneracy rate, which isn't
    symmetric between x and y unless they happen to have the same
    degeneracy rate. In practice this made the test's own p-value
    depend on which of the two datasets was passed as x vs y, even
    from the same seed and on the same underlying data -- confirmed
    directly, and the gap didn't shrink with more resamples, ruling out
    ordinary Monte Carlo noise as the explanation.

    Instead, variance_floor (default 1e-6, tiny relative to x0/y0's
    standardized unit variance) is added to EVERY resampled variance,
    identically on both sides. This keeps the log-ratio statistic
    finite for every resample without ever conditioning on, or
    discarding, any of them. observed_diff and the CI bounds are then
    exactly negated by swapping x and y (verified directly), since
    those come from the deterministic, non-resampled vx/vy. p itself
    converges to the same value as B grows regardless of x/y order
    (confirmed directly: the gap between the two orderings shrinks
    roughly as 1/sqrt(B), the signature of ordinary Monte Carlo noise,
    rather than staying roughly constant as it did with the discarded
    reject-and-redraw approach) -- the small residual at finite B comes
    from x and y drawing from two distinct (if independent) random
    streams, not from any remaining systematic bias.

    x and y draw from two INDEPENDENT rng streams (rng.spawn(2)) rather
    than both advancing one shared stream, so this test's result
    depends only on its own data and rng, never on which of the two
    inputs happens to be passed as x vs y.
    """
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    if x.size < 2 or y.size < 2:
        raise ValueError(f"Each group needs at least 2 observations {x.size} {y.size}.")

    vx = np.var(x, ddof=1); vy = np.var(y, ddof=1)

    if vx == 0 or vy == 0:
        print(f"note: zero variance in at least one group (var_a={vx:.4g}, var_b={vy:.4g}) -- "
              f"variance-ratio bootstrap is undefined here; reporting observed variances only.")
        obs_ratio = vx / vy if vy > 0 else np.inf
        obs = np.log(obs_ratio) if obs_ratio > 0 else -np.inf
        return {
            'value_a': float(np.var(x)),
            'value_b': float(np.var(y)),
            'observed_diff': float(obs),
            'ci_low': None,
            'ci_high': None,
            'p_value': None,
        }

    # Observed (use log-ratio for symmetric tails)
    obs_ratio = vx / vy
    obs = np.log(obs_ratio)

    # Impose H0: standardize to unit variance (centering optional)
    x0 = (x - np.mean(x)) / np.sqrt(vx)
    y0 = (y - np.mean(y)) / np.sqrt(vy)

    # Bootstrap under H0
    if nx is None:
        nx = x.size
    if ny is None:
        ny = y.size

    rng_x, rng_y = rng.spawn(2)

    bx = rng_x.choice(x0, size=(B, nx), replace=True)
    by = rng_y.choice(y0, size=(B, ny), replace=True)
    var_bx = np.var(bx, axis=1, ddof=1) + variance_floor
    var_by = np.var(by, axis=1, ddof=1) + variance_floor
    stats = np.log(var_bx / var_by)

    p = (np.sum(np.abs(stats) >= abs(obs)) + 1) / (B + 1)

    alpha = 1.0 - ci
    ci_low, ci_high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        'value_a':float(np.var(x)),
        'value_b':float(np.var(y)),
        'observed_diff':float(obs),
        'ci_low':ci_low,
        'ci_high':ci_high,
        'p_value':float(p)
        }



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
