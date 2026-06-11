"""
robust_optimization.py
======================
Distributionally Robust Optimization (DRO) and CVaR-based chance constraints
for robust driver-order scheduling under travel-time uncertainty.

References
----------
* Mohajerin Esfahani & Kuhn (2018) — Data-driven distributionally robust
  optimization using the Wasserstein metric.
* Rockafellar & Uryasev (2000/2002) — Optimization of conditional value-at-risk.
"""

import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check_samples(samples: np.ndarray, name: str = "samples"):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 0 or samples.size == 0:
        raise ValueError(f"{name} must be a non-empty array.")
    return samples


# ---------------------------------------------------------------------------
# 1. Wasserstein DRO
# ---------------------------------------------------------------------------

class WassersteinDRO:
    """
    Distributionally Robust Optimization with Wasserstein ball B_ε(P̂).

    Dual reformulation (Mohajerin Esfahani & Kuhn 2018):

        min_x  max_{Q ∈ B_ε(P̂)}  E_Q[ℓ(x, ξ)]
            ≈  min_x { λε + (1/N) Σ_i sup_ξ [ ℓ(x,ξ) − λ·d(ξ, ξ_i) ] }

    Here the loss ℓ(x, ξ) models *deadline violation*:
        ξ  = travel time
        ℓ  = max(ξ − slack, 0)   (hinge loss on deadline slack)

    For a given confidence level α the class estimates the worst-case
    α-quantile of travel time under the Wasserstein ambiguity set.

    Parameters
    ----------
    epsilon : float  Wasserstein ball radius (robustness parameter).
    p_norm  : int    Order of Wasserstein distance (1 or 2).
    """

    def __init__(self, epsilon: float = 0.1, p_norm: int = 1):
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")
        if p_norm not in (1, 2):
            raise ValueError("p_norm must be 1 or 2.")
        self.epsilon = epsilon
        self.p_norm = p_norm
        self._samples: np.ndarray | None = None

    # ------------------------------------------------------------------

    def fit(self, travel_time_samples: np.ndarray):
        """
        Store empirical travel-time samples.

        Parameters
        ----------
        travel_time_samples : np.ndarray, shape (N,) or (N, D)
            Observed travel times.  For 1-D scheduling use shape (N,).
        """
        self._samples = _check_samples(travel_time_samples, "travel_time_samples")
        return self

    # ------------------------------------------------------------------

    def robust_deadline_slack(self, deadline: float, current_time: float,
                              confidence: float = 0.95) -> float:
        """
        Return the worst-case slack under Wasserstein ambiguity.

        Slack  =  deadline − current_time − τ_worst

        where τ_worst is the upper confidence quantile of travel time,
        inflated by the Wasserstein radius ε (dual penalty):

            τ_worst ≈ Q_{α}(P̂) + ε   (1-norm approximation)

        A positive slack means the deadline is still reachable under the
        worst distribution in B_ε(P̂).  A negative slack signals a
        high-confidence violation.

        Parameters
        ----------
        deadline     : float  Absolute deadline time.
        current_time : float  Current time t_now.
        confidence   : float  Confidence level α ∈ (0, 1).

        Returns
        -------
        slack : float  Worst-case deadline slack (may be negative).
        """
        if self._samples is None:
            raise RuntimeError("Call fit() with travel-time samples first.")

        samples_1d = self._samples.ravel()
        N = len(samples_1d)

        # Empirical α-quantile
        q_alpha = float(np.quantile(samples_1d, confidence))

        # Wasserstein inflation:
        #   For p=1 norm, the worst-case quantile shifts by ε·N/(N·(1−α))
        #   Simplified conservative bound: add ε directly to the quantile.
        if self.p_norm == 1:
            tau_worst = q_alpha + self.epsilon
        else:  # p=2 — tighter bound via Cauchy-Schwarz
            tau_worst = q_alpha + self.epsilon / np.sqrt(N * (1.0 - confidence) + 1e-9)

        available = deadline - current_time
        slack = available - tau_worst
        return float(slack)

    # ------------------------------------------------------------------

    def worst_case_expected_cost(self, cost_fn, x, lambda_dual: float | None = None) -> float:
        """
        Approximate the dual objective:
            λε + (1/N) Σ_i sup_ξ [ cost_fn(x, ξ) − λ·|ξ − ξ_i| ]

        Uses a simple grid search over ξ in the convex hull of samples.

        Parameters
        ----------
        cost_fn    : callable  cost_fn(x, xi) -> float
        x          : any       decision variable passed to cost_fn
        lambda_dual: float     Lagrange multiplier λ (auto-tuned if None)

        Returns
        -------
        worst_case_cost : float
        """
        if self._samples is None:
            raise RuntimeError("Call fit() first.")

        samples_1d = self._samples.ravel()
        N = len(samples_1d)

        # Auto-tune λ as the Lipschitz constant estimate of cost_fn
        if lambda_dual is None:
            span = float(np.ptp(samples_1d)) + 1e-9
            cost_range = max(abs(cost_fn(x, samples_1d.max())),
                             abs(cost_fn(x, samples_1d.min()))) + 1e-9
            lambda_dual = cost_range / span

        # Grid of candidate ξ values (100 points over extended range)
        xi_min = samples_1d.min() - 3 * self.epsilon
        xi_max = samples_1d.max() + 3 * self.epsilon
        xi_grid = np.linspace(xi_min, xi_max, 100)

        suprema = np.zeros(N)
        for i, xi_i in enumerate(samples_1d):
            vals = np.array([cost_fn(x, xi) - lambda_dual * abs(xi - xi_i)
                             for xi in xi_grid])
            suprema[i] = vals.max()

        return float(lambda_dual * self.epsilon + suprema.mean())


# ---------------------------------------------------------------------------
# 2. CVaR Scheduler
# ---------------------------------------------------------------------------

class CVaRScheduler:
    """
    CVaR Chance Constraint for driver-order assignments.

    CVaR (Conditional Value-at-Risk) at level α:

        CVaR_α[τ] = min_v { v + 1/(1−α) · E[ max(τ − v, 0) ] }

    Solved via Sample Average Approximation (SAA):

        CVaR_α[τ] ≈ min_v { v + 1/((1−α)·N) · Σ_i max(τ_i − v, 0) }

    The feasibility check ensures:
        CVaR_α[τ(driver_k, order_j)] ≤ deadline_j − t_now

    Parameters
    ----------
    alpha : float  Confidence level α ∈ (0, 1).  Default 0.95.
    """

    def __init__(self, alpha: float = 0.95):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = alpha

    # ------------------------------------------------------------------

    def compute_cvar(self, samples: np.ndarray) -> float:
        """
        Compute CVaR_α of travel-time samples via SAA.

        CVaR_α = min_v { v + 1/((1−α)·N) Σ_i max(s_i − v, 0) }

        The optimal v* is the empirical α-quantile (closed-form).

        Parameters
        ----------
        samples : np.ndarray, shape (N,)

        Returns
        -------
        cvar : float
        """
        samples = _check_samples(samples, "samples").ravel()
        v_star = float(np.quantile(samples, self.alpha))
        exceedances = np.maximum(samples - v_star, 0.0)
        cvar = v_star + exceedances.mean() / (1.0 - self.alpha)
        return float(cvar)

    # ------------------------------------------------------------------

    def compute_cvar_lp(self, samples: np.ndarray) -> float:
        """
        Compute CVaR_α via LP (scipy.optimize.linprog) for verification.

        Solves:
            min_{v, s_i}  v + 1/((1−α)N) Σ s_i
            s.t.          s_i >= samples_i − v    ∀ i
                          s_i >= 0                ∀ i

        Decision variables: [v, s_1, ..., s_N]
        """
        samples = _check_samples(samples, "samples").ravel()
        N = len(samples)

        # Objective: min v + 1/((1-alpha)*N) * sum(s_i)
        c_obj = np.zeros(1 + N)
        c_obj[0] = 1.0
        c_obj[1:] = 1.0 / ((1.0 - self.alpha) * N)

        # Inequality: s_i >= samples_i - v  =>  -v - s_i <= -samples_i
        # A_ub * x <= b_ub
        A_ub = np.zeros((N, 1 + N))
        A_ub[:, 0] = -1.0                       # -v
        A_ub[np.arange(N), 1 + np.arange(N)] = -1.0  # -s_i
        b_ub = -samples

        # Bounds: v unbounded, s_i >= 0
        bounds = [(None, None)] + [(0.0, None)] * N

        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            return float(result.fun)
        # Fallback to closed-form
        return self.compute_cvar(samples)

    # ------------------------------------------------------------------

    def is_feasible(self, travel_time_samples: np.ndarray,
                    deadline: float, t_now: float):
        """
        Check if CVaR_α[τ] ≤ deadline − t_now.

        Parameters
        ----------
        travel_time_samples : np.ndarray, shape (N,)
        deadline            : float
        t_now               : float

        Returns
        -------
        (feasible, cvar_value) : (bool, float)
        """
        cvar_val = self.compute_cvar(travel_time_samples)
        budget = deadline - t_now
        return (bool(cvar_val <= budget), float(cvar_val))

    # ------------------------------------------------------------------

    def safe_assignment_filter(self,
                               cost_matrix: np.ndarray,
                               travel_samples: list,
                               deadlines: list,
                               t_now: float) -> np.ndarray:
        """
        Zero out infeasible (driver, order) pairs in cost_matrix based on CVaR.

        Parameters
        ----------
        cost_matrix    : np.ndarray, shape (n_drivers, n_orders)
        travel_samples : list of np.ndarray, shape (n_drivers, n_orders, N_samples)
                         OR a single array of shape (n_drivers, n_orders, N_samples).
                         travel_samples[k][j] = array of travel-time samples
                         from driver k to order j.
        deadlines      : list of float, length n_orders
                         Absolute deadline b_j for each order.
        t_now          : float  Current time.

        Returns
        -------
        filtered_cost  : np.ndarray, shape (n_drivers, n_orders)
            A copy of cost_matrix with infeasible entries set to a large
            penalty (1e6), leaving feasible pairs unchanged.
        """
        cost_matrix = np.asarray(cost_matrix, dtype=float)
        n_drivers, n_orders = cost_matrix.shape
        filtered = cost_matrix.copy()

        travel_arr = np.asarray(travel_samples)  # try to convert to array

        for k in range(n_drivers):
            for j in range(n_orders):
                # Extract travel-time samples for pair (k, j)
                if travel_arr.ndim == 3:
                    samp_kj = travel_arr[k, j]
                elif isinstance(travel_samples, list):
                    row = travel_samples[k]
                    if isinstance(row, list):
                        samp_kj = np.asarray(row[j], dtype=float)
                    else:
                        samp_kj = np.asarray(row, dtype=float)
                else:
                    samp_kj = np.asarray(travel_samples, dtype=float).ravel()

                deadline_j = float(deadlines[j])
                feasible, _ = self.is_feasible(samp_kj, deadline_j, t_now)
                if not feasible:
                    filtered[k, j] = 1e6  # mark as infeasible

        return filtered


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # ---- WassersteinDRO ----
    tt_samples = rng.normal(loc=600, scale=120, size=200)  # seconds
    dro = WassersteinDRO(epsilon=60.0, p_norm=1)
    dro.fit(tt_samples)

    slack = dro.robust_deadline_slack(deadline=1800, current_time=900, confidence=0.95)
    print(f"WassersteinDRO robust_deadline_slack: {slack:.1f} s")

    # ---- CVaRScheduler ----
    cvar_sched = CVaRScheduler(alpha=0.95)
    cvar_val = cvar_sched.compute_cvar(tt_samples)
    print(f"CVaR_0.95 of travel times: {cvar_val:.1f} s")

    feasible, cvar_v = cvar_sched.is_feasible(tt_samples, deadline=1800, t_now=900)
    print(f"is_feasible: {feasible}, CVaR={cvar_v:.1f} s, budget={1800-900} s")

    # ---- safe_assignment_filter ----
    n_d, n_o, n_s = 3, 4, 50
    cost_mat = rng.uniform(1, 10, (n_d, n_o))
    travel_samp = rng.normal(loc=300, scale=60, size=(n_d, n_o, n_s))
    deadlines = [1500, 600, 1200, 900]  # order 1 and 3 will be tight

    filtered = cvar_sched.safe_assignment_filter(cost_mat, travel_samp, deadlines, t_now=400)
    print(f"filtered cost_matrix:\n{filtered.round(2)}")
    print("robust_optimization.py OK")
