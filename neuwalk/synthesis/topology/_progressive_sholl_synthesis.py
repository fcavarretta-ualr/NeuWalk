import numpy as np


def synthesize_progressive(
    tree,
    n_std=1.0,
    max_attempts_per_window=10,
    max_total_attempts=1000,
    verbose=False,
):
    """
    Synthesize a section tree progressively while enforcing Sholl constraints.

    The tree is generated one Sholl bin at a time. After each bin is synthesized,
    its number of Sholl intersections is compared with the allowed interval:

        mean - n_std * std <= intersections <= mean + n_std * std

    When a bin fails validation, the function rolls the tree back and retries
    that bin. If repeated attempts fail, the rollback window is progressively
    expanded to include earlier bins. For example, failure at bin ``i`` causes
    regeneration of:

        i
        i - 1 through i
        i - 2 through i
        ...

    Each accepted bin creates a checkpoint based on ``tree.synthesis_logs``.
    Rollback is performed through ``tree.undo_synthesize()``.

    Once every Sholl bin has been accepted, if ``tree.bifurcation_count_constraint``
    is set, the tree's total bifurcation count (``tree.bifurcation_count()``) is
    checked against it, using the same ``n_std`` as the Sholl bins:

        mean - n_std * std <= bifurcation count <= mean + n_std * std

    Unlike a failed Sholl bin, a failed bifurcation-count check rolls the tree
    all the way back to its initial state and regenerates every bin from
    scratch, since the count is a property of the whole tree rather than any
    one bin. This repeats until the count is accepted or the shared attempt
    budget (``max_total_attempts``) is exhausted. When
    ``tree.bifurcation_count_constraint`` is unset (``None``), this check is
    skipped entirely and the function returns as soon as every Sholl bin is
    accepted, exactly as before this check existed.

    Note that in the worst case (e.g. an unsatisfiable early bin), the total
    number of regeneration attempts can increase roughly quadratically in the
    number of bins, since each window expansion re-attempts every bin from
    the new start up to ``target_bin``. Whole-tree regeneration triggered by
    the bifurcation-count check compounds this further, since it re-runs the
    entire bin-by-bin process again. Tune ``max_attempts_per_window`` and
    ``max_total_attempts`` with this in mind.

    Parameters
    ----------
    tree : SectionTreeProfile
        Tree to synthesize. It must provide:

        - ``sholl_plot_constraint["mean"]`` and ``["std"]``
        - ``bifurcation_count_constraint`` (``None``, or a dict with
          ``"mean"`` and ``"std"``)
        - ``bifurcation_count()``
        - ``event_sampler.bin_size``
        - ``event_sampler.step_size``
        - ``synthesis_logs``
        - ``synthesize(max_steps=...)``
        - ``undo_synthesize()``
        - ``sholl_plot(max_distance=...)``
        - ``roots``, or ``soma`` if ``with_soma`` is True

    n_std : float, default=1.0
        Number of standard deviations allowed above and below each target
        Sholl mean, and (once every Sholl bin is accepted) around the target
        mean bifurcation count when ``tree.bifurcation_count_constraint`` is
        set. A target with zero standard deviation must match its mean
        numerically.

    max_attempts_per_window : int, default=10
        Maximum number of regeneration attempts for a rollback window before
        expanding the window to include one additional earlier bin.

    max_total_attempts : int, default=1000
        Maximum number of regeneration attempts across the entire synthesis,
        shared between per-bin regeneration and whole-tree regeneration
        triggered by the bifurcation-count check. The tree is restored to
        its initial state if this limit is reached.

    verbose : bool, default=False
        Print information about synthesis attempts, validation results,
        accepted bins, and rollback operations.

    Returns
    -------
    SectionSynthesizer or list
        ``tree.soma`` if ``tree.with_soma`` is True, otherwise the
        synthesized primary roots stored in ``tree.roots``.

    Raises
    ------
    ValueError
        If the Sholl constraint is missing, malformed, or an argument is
        outside its valid range.

    TypeError
        If the tree does not provide the required synthesis or rollback
        methods.

    RuntimeError
        If the Sholl constraints, or the bifurcation-count constraint,
        cannot be satisfied within the permitted regeneration attempts.

    Notes
    -----
    Bin zero initializes the primary sections using
    ``tree.synthesize(max_steps=0)``. Every subsequent bin advances synthesis
    by ``ceil(bin_size / step_size)`` steps.
    """

    def _log(message):
        if verbose:
            print(f"[progressive synthesis] {message}")

    mean, std = _validate(tree, n_std, max_attempts_per_window, max_total_attempts)

    # Convert one Sholl bin into synthesis steps.
    bin_size = float(tree.event_sampler.bin_size)

    # Record the initial state; every whole-tree regeneration rolls back to
    # this point and starts over.
    base_log_count = len(tree.synthesis_logs)
    total_attempts = 0

    if verbose:
        _log("Sholl plot")
        _log("-" * 48)
        for i, (m, s) in enumerate(zip(mean, std)):
            _log(f"{i * bin_size}\t{round(m - n_std * s, 1)}\t{round(m + n_std * s, 1)}")
        _log("-" * 48)

    _log(f"Starting synthesis for {len(mean)} bins.")

    while True:
        # Record the accepted state after each bin, reset for every
        # whole-tree attempt.
        checkpoints = []

        # Accept one Sholl bin at a time.
        for target_bin in range(len(mean)):
            start_bin = target_bin

            while True:
                _log(f"Regenerating bins {start_bin}-{target_bin}.")

                # Retry the current rollback window.
                for window_attempt in range(1, max_attempts_per_window + 1):
                    if total_attempts >= max_total_attempts:
                        _rollback(tree, base_log_count, _log)
                        raise RuntimeError(f"Unable to satisfy Sholl constraints within {max_total_attempts} attempts.")

                    total_attempts += 1
                    rollback_count = base_log_count if start_bin == 0 else checkpoints[start_bin - 1]

                    _log(f"Attempt {total_attempts}, window attempt {window_attempt}/{max_attempts_per_window}.")
                    _rollback(tree, rollback_count, _log)
                    del checkpoints[start_bin:]

                    if _regenerate_window(tree, start_bin, target_bin, mean, std, n_std, bin_size, checkpoints, _log):
                        _log(f"Bin {target_bin} accepted.")
                        break
                else:
                    # Expand the rollback window after repeated failure.
                    if start_bin == 0:
                        _rollback(tree, base_log_count, _log)
                        raise RuntimeError(f"Unable to satisfy Sholl bins 0-{target_bin}.")

                    start_bin -= 1
                    continue

                break

        # Every Sholl bin was accepted. If no bifurcation-count constraint
        # was given, synthesis is done.
        if tree.bifurcation_count_constraint is None:
            break

        valid, generated, lower, upper = _bifurcation_count_status(tree, n_std)
        status = "accepted" if valid else "rejected"
        _log(f"Bifurcation count: generated={generated:g}, allowed=[{lower:g}, {upper:g}] -> {status}.")

        if valid:
            break

        if total_attempts >= max_total_attempts:
            _rollback(tree, base_log_count, _log)
            raise RuntimeError(
                f"Unable to satisfy the bifurcation-count constraint within {max_total_attempts} attempts "
                f"(last generated count: {generated:g}, allowed: [{lower:g}, {upper:g}])."
            )

        _log("Bifurcation count out of range; regenerating the entire tree.")
        _rollback(tree, base_log_count, _log)

    _log("Progressive synthesis completed successfully.")
    return tree.soma if getattr(tree, "with_soma", False) else tree.roots


def _regenerate_window(tree, start_bin, target_bin, mean, std, n_std, bin_size, checkpoints, _log):
    """Regenerate and validate all bins in one rollback window.

    Note: on failure this does not roll back itself; the caller always rolls
    back to the window's starting checkpoint before the next attempt, so an
    extra rollback here would just be immediately undone.
    """
    for bin_index in range(start_bin, target_bin + 1):
        # Bin 0 initializes the primary sections.
        tree.synthesize(distance_limit=0 if bin_index == 0 else (bin_size * bin_index))

        valid, generated, lower, upper = _sholl_status(tree, bin_index, mean, std, n_std, bin_size)
        status = "accepted" if valid else "rejected"
        _log(f"Bin {bin_index}: generated={generated:g}, allowed=[{lower:g}, {upper:g}] -> {status}.")

        if not valid:
            return False

        checkpoints.append(len(tree.synthesis_logs))

    return True


def _sholl_status(tree, bin_index, mean, std, n_std, bin_size):
    """Return whether one generated Sholl bin satisfies its constraint."""
    generated = float(tree.sholl_plot(max_distance=bin_index * bin_size)[bin_index])
    target_mean, target_std = float(mean[bin_index]), float(std[bin_index])

    # Zero standard deviation requires an exact numerical match.
    if np.isclose(target_std, 0.0):
        return bool(np.isclose(generated, target_mean)), generated, target_mean, target_mean

    lower = target_mean - n_std * target_std
    upper = target_mean + n_std * target_std
    return lower <= generated <= upper, generated, lower, upper


def _bifurcation_count_status(tree, n_std):
    """Return whether the tree's total bifurcation count satisfies its constraint."""
    constraint = tree.bifurcation_count_constraint
    generated = float(tree.bifurcation_count())
    target_mean, target_std = float(constraint["mean"]), float(constraint["std"])

    # Zero standard deviation requires an exact numerical match.
    if np.isclose(target_std, 0.0):
        return bool(np.isclose(generated, target_mean)), generated, target_mean, target_mean

    lower = target_mean - n_std * target_std
    upper = target_mean + n_std * target_std
    return lower <= generated <= upper, generated, lower, upper


def _rollback(tree, log_count, _log):
    """Undo synthesis operations until the requested log count is restored."""
    undo_count = len(tree.synthesis_logs) - log_count

    if undo_count > 0:
        suffix = "s" if undo_count != 1 else ""
        _log(f"Undoing {undo_count} synthesis log{suffix}.")

    while len(tree.synthesis_logs) > log_count:
        tree.undo_synthesize()


def _validate(tree, n_std, max_attempts_per_window, max_total_attempts):
    """Validate the tree interface and synthesis arguments.

    Returns
    -------
    tuple of numpy.ndarray
        The validated ``(mean, std)`` Sholl target arrays.
    """
    # --- Argument validation ---
    if n_std < 0:
        raise ValueError("n_std cannot be negative.")
    if max_attempts_per_window <= 0:
        raise ValueError("max_attempts_per_window must be positive.")
    if max_total_attempts <= 0:
        raise ValueError("max_total_attempts must be positive.")

    # --- Tree interface validation ---
    if not callable(getattr(tree, "synthesize", None)):
        raise TypeError("tree must provide synthesize(max_steps=...).")
    if not callable(getattr(tree, "undo_synthesize", None)):
        raise TypeError("tree must provide undo_synthesize().")
    if not callable(getattr(tree, "sholl_plot", None)):
        raise TypeError("tree must provide sholl_plot(max_distance=...).")
    if not callable(getattr(tree, "bifurcation_count", None)):
        raise TypeError("tree must provide bifurcation_count().")
    if getattr(tree, "synthesis_logs", None) is None:
        raise TypeError("tree must provide synthesis_logs.")
    if not hasattr(tree, "bifurcation_count_constraint"):
        raise TypeError("tree must provide bifurcation_count_constraint.")

    root_attr = "soma" if getattr(tree, "with_soma", False) else "roots"
    if not hasattr(tree, root_attr):
        raise TypeError(f"tree must provide {root_attr}.")

    event_sampler = getattr(tree, "event_sampler", None)
    if event_sampler is None:
        raise TypeError("tree must provide event_sampler.")
    if getattr(event_sampler, "bin_size", None) is None:
        raise TypeError("tree.event_sampler must provide bin_size.")
    step_size = getattr(event_sampler, "step_size", None)
    if step_size is None:
        raise TypeError("tree.event_sampler must provide step_size.")
    if float(step_size) <= 0:
        raise ValueError("tree.event_sampler.step_size must be positive.")

    # --- Sholl constraint validation ---
    constraint = getattr(tree, "sholl_plot_constraint", None)
    if constraint is None:
        raise ValueError("tree.sholl_plot_constraint is required.")
    if "mean" not in constraint or "std" not in constraint:
        raise ValueError("tree.sholl_plot_constraint must have 'mean' and 'std' entries.")

    mean = np.asarray(constraint["mean"], dtype=float)
    std = np.asarray(constraint["std"], dtype=float)

    if mean.ndim != 1:
        raise ValueError(f"Sholl mean must be 1D, got shape {mean.shape}.")
    if std.ndim != 1:
        raise ValueError(f"Sholl std must be 1D, got shape {std.shape}.")
    if mean.size == 0:
        raise ValueError("Sholl mean must be nonempty.")
    if mean.shape != std.shape:
        raise ValueError(f"Sholl mean and std must have the same shape, got {mean.shape} and {std.shape}.")

    # --- Bifurcation-count constraint validation ---
    bifurcation_count_constraint = tree.bifurcation_count_constraint

    if bifurcation_count_constraint is not None:
        if "mean" not in bifurcation_count_constraint or "std" not in bifurcation_count_constraint:
            raise ValueError("tree.bifurcation_count_constraint must have 'mean' and 'std' entries.")

    return mean, std
