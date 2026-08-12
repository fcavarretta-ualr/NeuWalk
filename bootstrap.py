"""
Two-sample bootstrap test for a difference in means.

Matches the procedure described in the Data Analysis section: for each
comparison, both populations are resampled with replacement `n_boot` times
to build an empirical distribution of the mean difference, from which a
two-tailed p-value is computed.
"""

import numpy as np


def bootstrap_mean_test(sample1, sample2, n_boot=10000, alpha=0.05, seed=None):
    """
    Two-tailed, two-sample bootstrap test comparing the means of two
    independent samples (e.g., synthetic vs. experimental measurements).

    Parameters
    ----------
    sample1, sample2 : array-like
        The two independent samples to compare. Order does not affect the
        p-value; `observed_diff` is reported as mean(sample1) - mean(sample2).
    n_boot : int, default=10000
        Number of bootstrap resamples.
    alpha : float, default=0.05
        Significance level used only to compute the reported confidence
        interval; the p-value itself is independent of alpha.
    seed : int or None, default=None
        Seed for the random number generator, for reproducibility.

    Returns
    -------
    dict
        observed_diff : float
            mean(sample1) - mean(sample2) computed on the original data.
        p_value : float
            Two-tailed bootstrap p-value (percentile method).
        ci : tuple of float
            (1 - alpha) bootstrap percentile confidence interval for the
            mean difference.
        boot_diffs : numpy.ndarray, shape (n_boot,)
            The full bootstrap distribution of mean differences, useful
            for diagnostics or plotting.

    Notes
    -----
    The p-value is computed via the percentile method: it is twice the
    smaller tail probability of the bootstrap distribution relative to
    zero, i.e. p = 2 * min(P(diff <= 0), P(diff >= 0)), capped at 1.
    This tests the null hypothesis that the two population means are
    equal, using only the shape of the resampled difference distribution
    (no normality assumption), which is appropriate given the limited
    sample sizes typical of experimental morphological reconstructions.

    When applying this test to many bins at once (e.g., across the radial
    bins of a Sholl plot), correct the resulting p-values for multiple
    comparisons (e.g., Bonferroni) within that family of tests, consistent
    with the correction applied in Cavarretta et al. (2025) and in the
    Sholl-plot comparisons of this work.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(sample1, dtype=float)
    y = np.asarray(sample2, dtype=float)
    n1, n2 = x.size, y.size

    if n1 == 0 or n2 == 0:
        raise ValueError("Both samples must contain at least one observation.")

    observed_diff = x.mean() - y.mean()

    # Vectorized resampling: draw n_boot resamples at once for each group.
    idx1 = rng.integers(0, n1, size=(n_boot, n1))
    idx2 = rng.integers(0, n2, size=(n_boot, n2))
    boot_diffs = x[idx1].mean(axis=1) - y[idx2].mean(axis=1)

    # Two-tailed p-value via the percentile method.
    p_le = np.mean(boot_diffs <= 0)
    p_ge = np.mean(boot_diffs >= 0)
    p_value = min(1.0, 2 * min(p_le, p_ge))

    lower, upper = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "ci": (lower, upper),
        "boot_diffs": boot_diffs,
    }


def bonferroni_correct(p_values, alpha=0.05):
    """
    Apply a Bonferroni correction to a family of p-values (e.g., one per
    Sholl-plot radial bin), matching the correction scope used for the
    Sholl-plot comparisons in this work (Section 4.2).

    Parameters
    ----------
    p_values : array-like
        Raw p-values from repeated application of `bootstrap_mean_test`
        within a single family of comparisons.
    alpha : float, default=0.05
        Family-wise significance level.

    Returns
    -------
    dict
        adjusted_alpha : float
            alpha / len(p_values), the per-comparison significance threshold.
        significant : numpy.ndarray of bool
            Whether each raw p-value is significant at `adjusted_alpha`.
    """
    p_values = np.asarray(p_values, dtype=float)
    adjusted_alpha = alpha / p_values.size
    return {
        "adjusted_alpha": adjusted_alpha,
        "significant": p_values < adjusted_alpha,
    }


if __name__ == "__main__":
    # Example usage: comparing synthetic vs. experimental branch-point counts
    # for a single cell type/arbor, as reported in Table 1.
    experimental = [2, 3, 2, 4, 3]        # e.g., SL basal branch-point counts
    synthetic = [3, 3, 4, 3, 5, 2]        # corresponding synthesized population

    result = bootstrap_mean_test(experimental, synthetic, n_boot=10000, seed=0)
    print(f"Observed difference (exp - sim): {result['observed_diff']:.3f}")
    print(f"95% CI: ({result['ci'][0]:.3f}, {result['ci'][1]:.3f})")
    print(f"Two-tailed p-value: {result['p_value']:.4f}")

    # Example: applying the test across multiple Sholl-plot bins, then
    # correcting for multiple comparisons within that family.
    exp_bins = [[2, 3, 2], [5, 6, 5], [3, 2, 4]]   # per-bin experimental samples
    sim_bins = [[3, 2, 3], [6, 5, 6], [2, 3, 3]]   # per-bin synthetic samples

    raw_p = [
        bootstrap_mean_test(e, s, n_boot=10000, seed=i)["p_value"]
        for i, (e, s) in enumerate(zip(exp_bins, sim_bins))
    ]
    correction = bonferroni_correct(raw_p, alpha=0.05)
    print("\nPer-bin p-values:", [f"{p:.4f}" for p in raw_p])
    print(f"Bonferroni-adjusted alpha: {correction['adjusted_alpha']:.4f}")
    print("Significant bins:", correction["significant"])
