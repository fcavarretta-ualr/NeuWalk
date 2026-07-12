from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pyomo.environ as pyo


def initial_count_pmf(
    mean_primary_dendrites: float,
    sd_primary_dendrites: float,
    min_primary_dendrites: int,
    max_primary_dendrites: int,
    *,
    support_values: Optional[Sequence[float]] = None,
    epsilon: float = 1e-12,
    slack_penalty: float = 1e-1,
    use_variance_form: bool = True,
    use_abs_slack: bool = False,
    solver: str = "ipopt",
    solver_options: Optional[Dict[str, Union[str, int, float]]] = None,
) -> np.ndarray:
    """
    Maximum-entropy PMF for the (discrete) number of primary dendrites.

    This returns a numpy array p of length n = max_primary_dendrites + 1, where:
      - p[i] is the probability of observing i primary dendrites
      - p[i] = 0 for i < min_primary_dendrites or i > max_primary_dendrites

    The distribution is obtained by maximizing Shannon entropy:
        H(p) = -sum_i p[i] * log(p[i])

    Subject to:
      - Normalization: sum_{i in [min,max]} p[i] = 1
      - Soft mean constraint (with slack):
            sum i*p[i] - mean_primary_dendrites = slack_mean
      - Soft dispersion constraint (with slack):
        If use_variance_form=True (recommended):
            sum (i-mean)^2 * p[i] - (sd_primary_dendrites^2) = slack_disp
        If use_variance_form=False:
            sqrt( sum (i-mean)^2 * p[i] + tiny ) - sd_primary_dendrites = slack_disp

    The objective is penalized to keep slacks small:
        maximize  H(p) - slack_penalty * (slack terms)

    Parameters
    ----------
    mean_primary_dendrites : float
        Target mean number of primary dendrites
    sd_primary_dendrites : float
        Target standard deviation (>= 0)
    min_primary_dendrites : int
        Minimum allowed dendrite count (inclusive)
    max_primary_dendrites : int
        Maximum allowed dendrite count (inclusive). Also sets array length n=max+1

    Keyword-only parameters
    ----------------------
    support_values : Sequence[float] | None
        Optional support for indices 0..max. If None, uses support=i (integers).
        Keep this None if you truly mean "i is the dendrite count".
    epsilon : float
        Lower bound on active probabilities to avoid log(0)
    slack_penalty : float
        Larger values enforce closer moment matching
    use_variance_form : bool
        Recommended True: match variance to sd^2 (smoother than sqrt constraint)
    use_abs_slack : bool
        If True, use L1-like slack penalty via +/- variables; otherwise squared (smooth)
    solver : str
        Nonlinear solver name (typically "ipopt")
    solver_options : dict | None
        Passed to the solver (e.g., {"max_iter": 5000})

    Returns
    -------
    np.ndarray
        Probability vector p with length max_primary_dendrites + 1

    Raises
    ------
    ValueError
        For invalid inputs
    RuntimeError
        If the requested solver is not available
    """
    if max_primary_dendrites < 0:
        raise ValueError("max_primary_dendrites must be >= 0")
    if sd_primary_dendrites < 0:
        raise ValueError("sd_primary_dendrites must be nonnegative")
    if not (0 <= min_primary_dendrites <= max_primary_dendrites):
        raise ValueError("Require 0 <= min_primary_dendrites <= max_primary_dendrites")
    if slack_penalty <= 0:
        raise ValueError("slack_penalty must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    n = max_primary_dendrites + 1
    active = list(range(min_primary_dendrites, max_primary_dendrites + 1))

    # Support values for each index i (default: i itself)
    if support_values is None:
        support_values = list(range(n))
    if len(support_values) != n:
        raise ValueError("support_values must have length n = max_primary_dendrites + 1")

    support = {i: float(support_values[i]) for i in range(n)}
    mu = float(mean_primary_dendrites)
    sd = float(sd_primary_dendrites)
    target_var = sd * sd

    # -----------------------------
    # Pyomo model
    # -----------------------------
    m = pyo.ConcreteModel()
    m.A = pyo.Set(initialize=active, ordered=True)

    # Decision variables for active probabilities only
    m.p = pyo.Var(m.A, domain=pyo.NonNegativeReals, bounds=(epsilon, 1.0))

    # Normalization over active set
    m.norm = pyo.Constraint(expr=sum(m.p[i] for i in m.A) == 1.0)

    # Moment expressions
    mean_expr = sum(support[i] * m.p[i] for i in m.A)
    var_expr = sum((support[i] - mu) ** 2 * m.p[i] for i in m.A)
    m.mean_const = pyo.Constraint(expr=mean_expr == mu)

    # Soft constraints with slack
    if use_abs_slack:
        # L1 slack via +/- decomposition
        m.s_mean_pos = pyo.Var(domain=pyo.NonNegativeReals)
        m.s_mean_neg = pyo.Var(domain=pyo.NonNegativeReals)
        m.s_disp_pos = pyo.Var(domain=pyo.NonNegativeReals)
        m.s_disp_neg = pyo.Var(domain=pyo.NonNegativeReals)

        #m.mean_soft = pyo.Constraint(expr=mean_expr - mu == m.s_mean_pos - m.s_mean_neg)

        if use_variance_form:
            m.disp_soft = pyo.Constraint(expr=var_expr - target_var == m.s_disp_pos - m.s_disp_neg)
        else:
            tiny = 1e-18
            m.disp_soft = pyo.Constraint(
                expr=pyo.sqrt(var_expr + tiny) - sd == m.s_disp_pos - m.s_disp_neg
            )

        slack_term = (m.s_mean_pos + m.s_mean_neg) + (m.s_disp_pos + m.s_disp_neg)

    else:
        # Smooth squared slacks
        m.s_mean = pyo.Var(domain=pyo.Reals)
        m.s_disp = pyo.Var(domain=pyo.Reals)

        #m.mean_soft = pyo.Constraint(expr=mean_expr - mu == m.s_mean)

        if use_variance_form:
            m.disp_soft = pyo.Constraint(expr=var_expr - target_var == m.s_disp)
        else:
            tiny = 1e-18
            m.disp_soft = pyo.Constraint(expr=pyo.sqrt(var_expr + tiny) - sd == m.s_disp)

        slack_term = m.s_mean**2 + m.s_disp**2

    # Entropy objective (active probs only; inactive probs are exactly 0)
    entropy = -sum(m.p[i] * pyo.log(m.p[i]) for i in m.A)
    m.obj = pyo.Objective(expr=entropy - float(slack_penalty) * slack_term, sense=pyo.maximize)

    # Solve
    opt = pyo.SolverFactory(solver)
    if opt is None or not opt.available():
        raise RuntimeError(
            f"Solver '{solver}' is not available. Install/configure it (e.g., ipopt) "
            "or pass a different solver name."
        )
    if solver_options:
        for k, v in solver_options.items():
            opt.options[k] = v

    res = opt.solve(m, tee=True)

    # -----------------------------
    # Extract solution into numpy array
    # -----------------------------
    p = np.zeros(n, dtype=float)
    for i in active:
        p[i] = float(pyo.value(m.p[i]))

    # Optional: renormalize tiny numerical drift (keeps zeros outside band)
    s = p.sum()
    if s > 0:
        p[active] /= s

    return p


if __name__ == "__main__":
    p = maxent_primary_dendrite_pmf(
        mean_primary_dendrites=2.33,
        sd_primary_dendrites=1.53,
        min_primary_dendrites=1,
        max_primary_dendrites=4,
        slack_penalty=0.1,
        use_variance_form=True,
        use_abs_slack=False,
        solver="ipopt",
    )
    print("p shape:", p.shape)
    print("sum:", p.sum())
    print(p)
