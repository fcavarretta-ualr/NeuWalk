from pyomo.environ import *
import numpy as np
from pyomo.core.expr.numeric_expr import Expr_if

_Bif_Var_Penalty = 1.0

def _entropy_term(p):
    return Expr_if(
        IF=(p <= eps),
        THEN=0.0,
        ELSE=p * log(p) / log3,
    )



def _f_var_term(bi, gi, Zj, Zi, Vi, bin_size):    
    if np.isclose(gi, 0):
        val = 2 * bi * Zi * bin_size
    else:
        val = (2 * bi - gi) * Zj * (Zj - Zi) / (Zi * gi) + Vi * ((Zj / Zi)**2 - 1)
    return val


def _f_var(beta, gamma, Z, V, bin_size, jmax):
    Vi = V[0]
    for j in range(1, jmax + 1):
        Vi = _f_var_term(beta[j - 1], gamma[j - 1], Z[j], Z[j - 1], Vi, bin_size)
    return Vi


def _mk_objective(beta, gamma, Z, V, bin_size):
    terms = []            
    for i in range(1, gamma.size + 1):
        terms.append(_f_var(beta, gamma, Z, V, bin_size, i)**2)
    return terms


def _mk_bif_mean_constraint(beta, Z, gamma, bin_size):
    """
    Add the mean constraint:
      sum_{i=1..len(Z)-1} term_i  ==  n_bif[0]
    where
      term_i = b[i-1] * Z[i-1] * bin_size              if gamma[i-1] ~ 0
             = b[i-1] * (Z[i] - Z[i-1]) / gamma[i-1]  otherwise
    """
    mean_terms = []
    for i in range(1, len(Z)):
        if np.isclose(gamma[i - 1], 0.0):
            term = beta[i - 1] * Z[i - 1] * bin_size
        else:
            term = beta[i - 1] * (Z[i] - Z[i - 1]) / gamma[i - 1]

        mean_terms.append(term)

    return mean_terms



def _mk_bif_var_terms(beta, Z, gamma, V, bin_size):
    """
    Build and return the variance terms for i=1..len(Z)-1

    For each i:
      - If gamma[i-1] ~ 0, use the "zero-gamma" formula
      - Else, use the "nonzero-gamma" formula

    Returns
      var_terms: list of term expressions (same order as i=1..)
    """
    var_terms = []

    for i in range(1, len(Z)):
        gi = gamma[i - 1]
        bi = beta[i - 1]
        
        vf = _f_var(beta, gamma, Z, V, bin_size, i)

        if np.isclose(gi, 0.0):
            term1 = bi**2 * ((2.0 / 3.0) * bi * Z[i - 1] * bin_size**3 + vf * bin_size**2)
            term2 = bi * Z[i - 1] * bin_size
            term = term1 + term2
        else:
            inner1 = (2 * bi - gi) * (Z[i]**2 - 2 * gi * bin_size * Z[i] * Z[i - 1] - Z[i - 1]**2) / (Z[i - 1] * gi**3)

            inner2 = vf * (Z[i]**2 - 2 * Z[i - 1] + Z[i - 1]) / (Z[i - 1]**2 * gi**2)

            term1 = bi**2 * (inner1 + inner2)
            term2 = bi * (Z[i] - Z[i - 1]) / gi
            term = term1 + term2

        var_terms.append(term)

    return var_terms



def _mk_bif_covar_terms(beta, Z, gamma, V, bin_size):
    """
    Build covariance terms for all (i, j) with 1 <= i < j <= len(Z)-1

    Each term:
      b[i-1] * b[j-1] * term1(i) * term2(j) * (Z[j-1]/Z[i-1]) * variance_function(..., i-1)

    where
      term1(i) = bin_size                              if gamma[i-1] ~ 0
               = (Z[i] - Z[i-1]) / gamma[i-1]         otherwise
      term2(j) = bin_size                              if gamma[j-1] ~ 0
               = (Z[j] - Z[j-1]) / gamma[j-1]         otherwise
    """
    cov_terms = []

    for j in range(1, len(Z)):
        for i in range(1, j):
            gj = gamma[j - 1]
            bj = beta[j - 1]
            
            gi = gamma[i - 1]
            bi = beta[i - 1]

            term1 = bin_size if np.isclose(gi, 0.0) else (Z[i] - Z[i - 1]) / gi
            
            term2 = bin_size if np.isclose(gj, 0.0) else (Z[j] - Z[j - 1]) / gj
            
            term3 = Z[j - 1] / Z[i - 1]
            
            vf = _f_var(beta, gamma, Z, V, bin_size, i)

            cov_terms.append(bi * bj * term1 * term2 * term3 * vf)

    return cov_terms



def event_rates(
    bin_size,
    sholl_plot,
    step_size,
    bifurcation_count=None,
    no_bifurcation_bins=None,
    no_annihilation_bins=None
):

    """
    Estimate radial bifurcation and annihilation rates.

    The rates are inferred from summary statistics of Sholl intersection
    counts and, optionally, bifurcation counts. The resulting rates describe
    a branching-annihilating process evaluated over the radial Sholl bins.

    Parameters
    ----------
    bin_size : float
        Width of each Sholl bin. It must be positive and use the same spatial
        units as ``step_size``.

    sholl_plot : dict
        Sholl-plot summary statistics with the following structure:

        ``{
            "mean": array-like,
            "std": array-like,
        }``

        ``mean[i]`` is the mean number of Sholl intersections in bin ``i``,
        and ``std[i]`` is the corresponding standard deviation. Both arrays
        must be one-dimensional and have the same length.

    step_size : float
        Radial advancement associated with one synthesis step. It must be
        positive and expressed in the same units as ``bin_size``.
    bifurcation_count : dict or None
    
        Summary statistics for the total number of bifurcations:

        ``{
            "mean": float,
            "std": float,
        }``

        When provided, these statistics are used as constraints during rate
        estimation. Pass ``None`` to estimate rates without bifurcation-count
        constraints.
        
    no_bifurcation_bins : array-like of bool, optional
        Boolean mask indicating bins in which bifurcation is prohibited.
        ``True`` disables bifurcation in the corresponding bin. Its length
        should match the Sholl arrays.

    no_annihilation_bins : array-like of bool, optional
        Boolean mask indicating bins in which annihilation is prohibited.
        ``True`` disables annihilation in the corresponding bin. Its length
        should match the Sholl arrays.

    bifurcation_internal_density : array-like or float, optional
        Internal-bifurcation density for each radial bin. A scalar may be
        applied uniformly to all bins. When omitted, internal-bifurcation
        density is assumed to be zero.

    Returns
    -------
    dict
        Estimated event-rate arrays:

        ``{
            "bifurcation_rate": numpy.ndarray,
            "annihilation_rate": numpy.ndarray,
            "internal_bifurcation_rate": numpy.ndarray,            
        }``

        The arrays correspond to the Sholl bins. The final bin acts as a
        terminal boundary, with bifurcation disabled and annihilation forced.

    Raises
    ------
    ValueError
        If input values, dimensions, or array lengths are invalid.

    RuntimeError
        If the optimization solver is unavailable or fails to find a valid
        solution.

    Notes
    -----
    Estimation stops at the first Sholl bin whose mean intersection count is
    zero. Bins beyond that point are not included in the optimization.

    The bifurcation and annihilation rates are constrained so that the
    corresponding per-step event probabilities remain valid for the supplied
    ``step_size``.
    """

    global _Bif_Var_Penalty


    # Keep data only up to the first zero in either mean or std (whichever occurs earlier)
    mean = np.array(sholl_plot['mean'])
    std  = np.array(sholl_plot['std'])

    i_non_zeros = np.flatnonzero(mean == 0)[0] if (mean == 0).any() else mean.size

    Z = mean[:i_non_zeros]
    V = std[:i_non_zeros] ** 2
    
    # no branch or annihilation marked bins
    if no_bifurcation_bins is None:
        no_bifurcation_bins = np.array([False] * i_non_zeros)
    else:
        no_bifurcation_bins = np.array(no_bifurcation_bins[:i_non_zeros])

    if no_annihilation_bins is None:
        no_annihilation_bins = np.array([False] * i_non_zeros)
    else:
        no_annihilation_bins = np.array(no_annihilation_bins[:i_non_zeros])
        

    
    
        
    # Use bifurcation mean and variance if available; otherwise set to None
    n_bif = [bifurcation_count['mean'], bifurcation_count['std'] ** 2] if bifurcation_count else None

    # calculate the gamma
    gamma = np.log(Z[1:] / Z[:-1]) / bin_size
    # Initial value for b: copy gamma, clamp negatives to 0, then map index -> value
    init_b = dict(enumerate(gamma.clip(min=0).copy()))

    # boundaries for beta rates
    lower_bound = np.maximum(0, gamma)
    upper_bound = 0.5 * (1.0 / step_size + gamma)

    # create the model 
    model = ConcreteModel()
    
    # Define index set and variables
    model.b = Var(range(gamma.size), domain=NonNegativeReals, initialize=init_b, bounds=lambda model, i : (lower_bound[i], upper_bound[i]))


    # some beta rates might be fixed
    for i in range(gamma.size):
        if no_annihilation_bins[i] and (gamma[i] >= 0 or np.isclose(gamma[i], 0)):
            model.b[i].fix(gamma[i])
        elif no_bifurcation_bins[i] and (gamma[i] <= 0 or np.isclose(gamma[i], 0)): 
            model.b[i].fix(0)


    # define 1 slack variables for eventual constraints of variance of bifurcations
    model.s = Var(domain=Reals)
            
    # Constraint: 
    model.constraints = ConstraintList()     
        
    # if we have number of bifurcations, use it as contraints
    if n_bif:        
        # constraint the average number of bifurcations
        if n_bif[0]:
            mean_terms = _mk_bif_mean_constraint(model.b, Z, gamma, bin_size)
            model.constraints.add(sum(mean_terms) == n_bif[0])
        
        # constrain the variance for the number of bifurcations
        if n_bif[1]:
            var_terms = _mk_bif_var_terms(model.b, Z, gamma, V, bin_size)
            covar_terms = _mk_bif_covar_terms(model.b, Z, gamma, V, bin_size)               
            model.constraints.add(sum(var_terms + covar_terms) + model.s == n_bif[1])

        
    # Objective
    model.obj = Objective(
        expr=sum(_mk_objective(model.b, gamma, Z, V, bin_size)) + _Bif_Var_Penalty * model.s ** 2,
        sense=minimize
    )
  
    # Solve
    solver = SolverFactory('ipopt')
    solver.options["print_level"] = 0
    solver.options["sb"] = "yes"
    solver.solve(model, tee=False, report_timing=False)

    # check bifurcation average
    estimate_nbif = value(sum(x for x in _mk_bif_mean_constraint(model.b, Z, gamma, bin_size)))
    assert np.isclose(estimate_nbif, n_bif[0]), f"Constraint for the average number of bifurcation is broken {estimate_nbif} {n_bif[0]}."
    
    # Extract bifurcation rates as array
    b = np.array([value(model.b[i]) for i in model.b])
    b[b < 0] = 0.

    # calculate annihilation rates
    a = - gamma + b   
    a[a < 0] = 0.

    # implement barrier at the end of the sholl plots
    b = np.append(b, 0.)
    a = np.append(a, np.inf)

    return { 'bifurcation_rate':b, 'annihilation_rate':a }
