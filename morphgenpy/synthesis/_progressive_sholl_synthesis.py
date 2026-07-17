"""Progressive Sholl-constrained synthesis using NeuriteTreeProfile.synthesize()."""

import numpy as np



def synthesize_progressive(
    tree,
    n_std=1.0,
    max_attempts_per_window=1,
    max_total_attempts=1000,
    verbose=False,
):
    """
    Synthesize a neurite tree using progressive Sholl-bin backtracking.

    The existing ``tree.synthesize(max_steps=...)`` method is used to generate
    each bin. If target bin ``i`` fails, the method retries increasingly large
    windows:

    ``i``;
    ``i - 1, i``;
    ``i - 2, i - 1, i``;
    and so on.

    Parameters
    ----------
    tree : NeuriteTreeProfile
        Tree containing a Sholl constraint and supporting ``synthesize()`` and
        ``undo_synthesize()``.
    n_std : float, default 1
        Allowed number of standard deviations from each Sholl mean.
    max_attempts_per_window : int, default 1
        Attempts before expanding the rollback window.
    max_total_attempts : int, default 1000
        Maximum total regeneration attempts.
    verbose : bool, default False
        Print synthesis, validation, and rollback messages.

    Returns
    -------
    list of NeuriteProfile
        Independently synthesized root profiles.
    """
    _validate_arguments(
        tree=tree,
        n_std=n_std,
        max_attempts_per_window=max_attempts_per_window,
        max_total_attempts=max_total_attempts,
    )

    constraint = tree.sholl_plot_constraint

    mean = np.asarray(constraint["mean"], dtype=float)
    std = np.asarray(constraint["std"], dtype=float)



    if mean.shape != std.shape:
        raise ValueError(
            "Sholl mean and standard-deviation arrays must have "
            "the same shape."
        )

    if mean.ndim != 1 or mean.size == 0:
        raise ValueError(
            "Sholl mean and standard deviation must be nonempty 1D arrays."
        )

    bin_size = float(tree.event_sampler.bin_size)
    step_size = float(tree.event_sampler.step_size)

    if verbose:
        print('Sholl plot')
        print('------------------------------------------------')
        for i, (m, s) in enumerate(zip(mean, std)):
            print(i * bin_size, '\t', round(m - n_std * s, 1), round(m + n_std * s, 1))
        print('------------------------------------------------')
    
    # Number of synthesis sweeps corresponding to one Sholl bin.
    steps_per_bin = int(np.ceil(bin_size / step_size))

    base_log_count = len(tree.synthesis_logs)

    # checkpoints[i] stores the number of logs after bin i is accepted.
    checkpoints = []

    target_bin = 0
    total_attempts = 0

    _print(
        verbose,
        f"Starting progressive synthesis for {len(mean)} Sholl bins.",
    )
    _print(
        verbose,
        f"Using {steps_per_bin} synthesis steps per bin.",
    )

    while target_bin < len(mean):
        start_bin = target_bin
        accepted = False

        _print(verbose, f"Targeting bin {target_bin}.")

        while not accepted:
            attempts_at_depth = 0

            _print(
                verbose,
                f"Regeneration window: bins {start_bin} through "
                f"{target_bin}.",
            )

            while attempts_at_depth < max_attempts_per_window:
                if total_attempts >= max_total_attempts:
                    _rollback_to_log_count(tree, base_log_count)

                    raise RuntimeError(
                        "Unable to satisfy the Sholl constraints within "
                        f"{max_total_attempts} attempts."
                    )

                total_attempts += 1
                attempts_at_depth += 1

                _print(
                    verbose,
                    f"Attempt {total_attempts}: regenerating bins "
                    f"{start_bin} through {target_bin} "
                    f"(window attempt {attempts_at_depth}/"
                    f"{max_attempts_per_window}).",
                )

                rollback_log_count = (
                    base_log_count
                    if start_bin == 0
                    else checkpoints[start_bin - 1]
                )

                _rollback_to_log_count(
                    tree,
                    rollback_log_count,
                    verbose=verbose,
                )

                del checkpoints[start_bin:]

                success = _regenerate_window(
                    tree=tree,
                    start_bin=start_bin,
                    target_bin=target_bin,
                    mean=mean,
                    std=std,
                    n_std=n_std,
                    bin_size=bin_size,
                    steps_per_bin=steps_per_bin,
                    checkpoints=checkpoints,
                    verbose=verbose,
                )

                if success:
                    _print(
                        verbose,
                        f"Bin {target_bin} accepted.",
                    )
                    accepted = True
                    break

                _print(
                    verbose,
                    f"Window {start_bin} through {target_bin} failed.",
                )

            if accepted:
                break

            if start_bin == 0:
                _rollback_to_log_count(
                    tree,
                    base_log_count,
                    verbose=verbose,
                )

                raise RuntimeError(
                    f"Unable to satisfy Sholl bins 0 through {target_bin}."
                )

            start_bin -= 1

            _print(
                verbose,
                f"Expanding rollback window to bins "
                f"{start_bin} through {target_bin}.",
            )

        target_bin += 1

    _print(
        verbose,
        "Progressive synthesis completed successfully.",
    )

    return tree.roots


def _regenerate_window(
    tree,
    start_bin,
    target_bin,
    mean,
    std,
    n_std,
    bin_size,
    steps_per_bin,
    checkpoints,
    verbose=False,
):
    """Regenerate and validate all bins in one rollback window."""
    for bin_index in range(start_bin, target_bin + 1):
        log_count_before = len(tree.synthesis_logs)

        if bin_index == 0:
            _print(
                verbose,
                "Initializing primary neurites.",
            )

            # max_steps=0 initializes the tree without performing
            # additional neurite synthesis.
            tree.synthesize(max_steps=0)

        else:
            _print(
                verbose,
                f"Synthesizing bin {bin_index} using "
                f"{steps_per_bin} steps.",
            )

            tree.synthesize(
                max_steps=steps_per_bin,
            )

        valid, generated, lower, upper = _sholl_bin_status(
            tree=tree,
            bin_index=bin_index,
            mean=mean,
            std=std,
            n_std=n_std,
            bin_size=bin_size,
        )

        _print(
            verbose,
            f"Bin {bin_index}: generated={generated:g}, "
            f"allowed=[{lower:g}, {upper:g}] -> "
            f"{'accepted' if valid else 'rejected'}.",
        )

        if not valid:
            # Undo logs generated for this failed bin.
            _rollback_to_log_count(
                tree,
                log_count_before,
                verbose=verbose,
            )
            return False

        checkpoints.append(len(tree.synthesis_logs))

    return True


def _sholl_bin_status(
    tree,
    bin_index,
    mean,
    std,
    n_std,
    bin_size,
):
    """Return the validation status for one Sholl bin."""
    generated_plot = tree.sholl_plot(
        max_distance=bin_index * bin_size,
    )

    generated = float(generated_plot[bin_index])

    if np.isclose(std[bin_index], 0.0):
        lower = upper = float(mean[bin_index])
        valid = np.isclose(
            generated,
            mean[bin_index],
        )
    else:
        lower = float(
            mean[bin_index] - n_std * std[bin_index]
        )
        upper = float(
            mean[bin_index] + n_std * std[bin_index]
        )

        valid = lower <= generated <= upper

    return bool(valid), generated, lower, upper


def _rollback_to_log_count(
    tree,
    log_count,
    verbose=False,
):
    """Undo synthesis logs until the requested checkpoint is restored."""
    undo_count = len(tree.synthesis_logs) - log_count

    if undo_count > 0:
        _print(
            verbose,
            f"Undoing {undo_count} synthesis log"
            f"{'s' if undo_count != 1 else ''}.",
        )

    while len(tree.synthesis_logs) > log_count:
        tree.undo_synthesize()


def _validate_arguments(
    tree,
    n_std,
    max_attempts_per_window,
    max_total_attempts,
):
    """Validate progressive-synthesis inputs."""
    if getattr(tree, "sholl_plot_constraint", None) is None:
        raise ValueError(
            "tree.sholl_plot_constraint is required."
        )

    if not hasattr(tree, "synthesize"):
        raise TypeError(
            "tree must provide a synthesize(max_steps=...) method."
        )

    if not hasattr(tree, "undo_synthesize"):
        raise TypeError(
            "tree must provide an undo_synthesize() method."
        )

    if n_std < 0:
        raise ValueError("n_std cannot be negative.")

    if max_attempts_per_window <= 0:
        raise ValueError(
            "max_attempts_per_window must be positive."
        )

    if max_total_attempts <= 0:
        raise ValueError(
            "max_total_attempts must be positive."
        )


def _print(verbose, message):
    """Print a progress message when verbose output is enabled."""
    if verbose:
        print(f"[progressive synthesis] {message}")
