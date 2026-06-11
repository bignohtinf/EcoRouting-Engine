"""
charging_sos2.py — SOS2 Piecewise-Linear Charging Curve Optimiser (Section 5.3).

Formulation
-----------
CC-CV charging curve approximated by S breakpoints:

    t_charge(y) = Σ_s  μ_s · λ_s            (time as convex combination of breakpoint times)
    y            = Σ_s  Δe_s · λ_s           (energy = convex combination of breakpoint energies)
    Σ_s λ_s = 1,  λ_s ≥ 0                   (weights are a convex combination)
    {λ_s} ∈ SOS2                             (at most 2 adjacent λ_s ≠ 0)

SOS2 is encoded via binary variables b_s ∈ {0,1} (big-M approach) and solved
with scipy.optimize.linprog (continuous relaxation is tight for SOS2 LP on
convex piecewise-linear functions).
"""

import math
import numpy as np
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Default CC-CV breakpoint generator
# ---------------------------------------------------------------------------

def _default_cccv_breakpoints(
    battery_cap_kwh: float = 84.0,
    max_power_kw: float = 150.0,
    n_segments: int = 10,
) -> list:
    """
    Simulate CC-CV curve and return (delta_e_kwh, cumulative_time_mins) breakpoints.

    Returns list of (cumulative_energy_kwh, cumulative_time_mins) pairs
    starting at (0, 0) — i.e., incremental energy from 0% SoC.
    """
    step_pct = 100.0 / n_segments
    breakpoints = [(0.0, 0.0)]
    cum_e = 0.0
    cum_t = 0.0
    soc = 0.0

    for _ in range(n_segments):
        kwh_needed = (step_pct / 100.0) * battery_cap_kwh
        if soc < 80.0:
            power = max_power_kw
        else:
            ratio = (100.0 - soc) / 20.0
            power = max(5.0, max_power_kw * (ratio ** 2))

        dt_mins = (kwh_needed / power) * 60.0
        cum_e += kwh_needed
        cum_t += dt_mins
        soc += step_pct
        breakpoints.append((round(cum_e, 4), round(cum_t, 4)))

    return breakpoints


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SOS2ChargingOptimizer:
    """
    Piecewise linear approximation of CC-CV charging curve using SOS2 constraints.

    The charging time as a function of energy added is convex and non-decreasing,
    which means the LP relaxation of the SOS2 big-M formulation is automatically
    tight (the LP optimal solution satisfies SOS2 naturally).

    Usage
    -----
    opt = SOS2ChargingOptimizer()
    result = opt.optimize_charging_schedule(start_soc_pct=20.0, target_soc_pct=80.0)
    """

    def __init__(self, breakpoints: list | None = None):
        """
        Parameters
        ----------
        breakpoints : list of (cumulative_energy_kwh, cumulative_time_mins) or None.
                      If None, uses 10-point default for 84 kWh / 150 kW charger.
        """
        self.breakpoints = breakpoints  # may be None; resolved lazily

    # ------------------------------------------------------------------
    # Breakpoint construction
    # ------------------------------------------------------------------

    def build_breakpoints(
        self,
        battery_cap_kwh: float = 84.0,
        max_power_kw: float = 150.0,
        n_segments: int = 10,
    ) -> list:
        """
        Build and store breakpoints for the CC-CV curve.

        Returns list of (cumulative_energy_kwh, cumulative_time_mins).
        """
        self.breakpoints = _default_cccv_breakpoints(battery_cap_kwh, max_power_kw, n_segments)
        self._battery_cap = battery_cap_kwh
        self._max_power = max_power_kw
        return self.breakpoints

    def _ensure_breakpoints(self, battery_cap_kwh: float, max_power_kw: float):
        if self.breakpoints is None:
            self.build_breakpoints(battery_cap_kwh, max_power_kw)
        self._battery_cap = getattr(self, '_battery_cap', battery_cap_kwh)
        self._max_power = getattr(self, '_max_power', max_power_kw)

    # ------------------------------------------------------------------
    # SOS2 big-M constraint builder
    # ------------------------------------------------------------------

    def _enforce_sos2_bigm(self, n: int, M: float = 1e6) -> list:
        """
        Generate big-M SOS2 constraints for n lambda variables.

        SOS2 requires at most 2 consecutive λ_s to be non-zero.
        Encoding: introduce binary b_s s.t.
          Σ b_s = 1
          λ_s ≤ b_{s-1} + b_s   (λ_s only active if one of its adjacent bins is chosen)

        Returns list of dicts compatible with scipy linprog 'A_ub' augmentation
        descriptions (informational only — caller builds matrices directly).
        """
        # Returns human-readable constraint descriptions; actual matrices built
        # in optimize_charging_schedule.
        constraints = []
        for s in range(n):
            constraints.append({
                'lambda_idx': s,
                'left_bin': max(0, s - 1),
                'right_bin': min(n - 1, s),
                'type': 'sos2_big_m',
            })
        return constraints

    # ------------------------------------------------------------------
    # Core LP optimisation
    # ------------------------------------------------------------------

    def optimize_charging_schedule(
        self,
        start_soc_pct: float,
        target_soc_pct: float,
        battery_cap_kwh: float = 84.0,
        max_power_kw: float = 150.0,
        time_limit_mins: float | None = None,
    ) -> dict:
        """
        Minimise charging time to reach target SoC from start SoC.

        Uses LP (scipy linprog) with SOS2 encoded as big-M constraints.
        The objective function (time) is convex piecewise-linear, so the LP
        relaxation of SOS2 is exact for minimisation.

        Parameters
        ----------
        start_soc_pct  : starting SoC in percent (0–100)
        target_soc_pct : desired SoC in percent (0–100)
        battery_cap_kwh: battery capacity in kWh
        max_power_kw   : charger max power
        time_limit_mins: optional maximum charging time (adds upper-bound constraint)

        Returns
        -------
        dict with keys:
            lambda_weights    : list of SOS2 weights
            optimal_time_mins : float
            energy_charged_kwh: float
            soc_trajectory    : list of (soc_pct, time_mins) milestones
        """
        if start_soc_pct >= target_soc_pct:
            return {
                'lambda_weights': [],
                'optimal_time_mins': 0.0,
                'energy_charged_kwh': 0.0,
                'soc_trajectory': [(start_soc_pct, 0.0)],
            }

        self._ensure_breakpoints(battery_cap_kwh, max_power_kw)

        # Energy range [e_start, e_target] in kWh (absolute)
        e_start = start_soc_pct / 100.0 * battery_cap_kwh
        e_target = target_soc_pct / 100.0 * battery_cap_kwh
        e_needed = e_target - e_start

        # Rescale breakpoints to the segment [e_start, e_target]
        # Full curve goes 0 → battery_cap_kwh
        bp_e = np.array([b[0] for b in self.breakpoints])   # cumulative energy kWh
        bp_t = np.array([b[1] for b in self.breakpoints])   # cumulative time mins

        # Interpolate breakpoints at e_start and e_target
        t_at_start = float(np.interp(e_start, bp_e, bp_t))
        t_at_target = float(np.interp(e_target, bp_e, bp_t))

        # Slice breakpoints to the window [e_start, e_target]
        # Add endpoints if they fall between existing breakpoints
        mask = (bp_e >= e_start) & (bp_e <= e_target)
        e_seg = np.concatenate([[e_start], bp_e[mask], [e_target]])
        t_seg = np.concatenate([[t_at_start], bp_t[mask], [t_at_target]])

        # Deduplicate and sort
        order = np.argsort(e_seg)
        e_seg = e_seg[order]
        t_seg = t_seg[order]
        _, uniq = np.unique(e_seg, return_index=True)
        e_seg = e_seg[uniq]
        t_seg = t_seg[uniq]

        # Shift so segment starts at 0
        t_seg = t_seg - t_at_start
        e_seg = e_seg - e_start

        S = len(e_seg)   # number of breakpoints in window

        # ---- LP formulation ------------------------------------------------
        # Variables: λ_0..λ_{S-1}  (S vars) + b_0..b_{S-2}  (S-1 binary-relaxed)
        # We use continuous relaxation of b (0 ≤ b_s ≤ 1) which is exact for
        # minimising a convex piecewise-linear objective.

        n_lam = S
        n_bin = max(1, S - 1)
        n_vars = n_lam + n_bin

        BIG_M = 2.0  # λ_s ≤ 1 always, so big-M=2 is sufficient

        # Objective: minimise Σ t_seg[s] · λ_s
        c = np.concatenate([t_seg, np.zeros(n_bin)])

        # Equality constraints
        #  (1) Σ λ_s = 1   (convex combination)
        #  (2) Σ e_seg[s] · λ_s = e_needed  (hit energy target)
        A_eq = np.zeros((2, n_vars))
        A_eq[0, :n_lam] = 1.0
        A_eq[1, :n_lam] = e_seg
        b_eq = np.array([1.0, e_needed])

        # Inequality constraints (big-M SOS2)
        #  (3)  Σ b_s = 1  →  encoded as  Σ b_s ≤ 1  and  Σ b_s ≥ 1
        #  (4)  λ_s ≤ b_{s-1} + b_s   for s=0..S-1
        #       (b_{-1} and b_{S-1} treated as 0 — boundary segments)
        A_ub_rows = []
        b_ub_rows = []

        # Σ b_s ≤ 1
        row = np.zeros(n_vars)
        row[n_lam:] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(1.0)

        # −Σ b_s ≤ −1  (i.e. Σ b_s ≥ 1)
        row = np.zeros(n_vars)
        row[n_lam:] = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(-1.0)

        # λ_s ≤ b_{s-1} + b_s
        for s in range(n_lam):
            row = np.zeros(n_vars)
            row[s] = 1.0           # λ_s
            if s > 0:
                row[n_lam + s - 1] = -1.0   # - b_{s-1}
            if s < n_bin:
                row[n_lam + s] = -1.0        # - b_s
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)

        # Time limit constraint
        if time_limit_mins is not None:
            row = np.zeros(n_vars)
            row[:n_lam] = t_seg
            A_ub_rows.append(row)
            b_ub_rows.append(float(time_limit_mins))

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)

        bounds = [(0.0, 1.0)] * n_lam + [(0.0, 1.0)] * n_bin

        res = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method='highs',
        )

        if res.success:
            lam = res.x[:n_lam]
            opt_time = float(np.dot(t_seg, lam))
            energy_charged = float(np.dot(e_seg, lam))
        else:
            # Fallback: linear interpolation
            lam = np.zeros(n_lam)
            lam[-1] = 1.0
            opt_time = t_seg[-1]
            energy_charged = e_needed

        # Build SoC trajectory milestones
        soc_traj = []
        cum_t = 0.0
        cum_e = 0.0
        for s in range(n_lam):
            w = float(lam[s])
            if w > 1e-9:
                cum_e += w * e_seg[s]
                cum_t += w * t_seg[s]
                soc_at = start_soc_pct + (cum_e / battery_cap_kwh) * 100.0
                soc_traj.append((round(soc_at, 2), round(cum_t, 3)))

        if not soc_traj:
            soc_traj = [(start_soc_pct, 0.0), (target_soc_pct, opt_time)]

        return {
            'lambda_weights': lam.tolist(),
            'optimal_time_mins': round(opt_time, 3),
            'energy_charged_kwh': round(energy_charged, 4),
            'soc_trajectory': soc_traj,
        }

    # ------------------------------------------------------------------
    # Comparison with analytical CC-CV simulation
    # ------------------------------------------------------------------

    def compare_with_simulation(
        self,
        start_soc_pct: float,
        target_soc_pct: float,
        battery_cap_kwh: float = 84.0,
        max_power_kw: float = 150.0,
    ) -> dict:
        """
        Compare SOS2 LP result vs step-wise CC-CV simulation.

        Returns dict with 'sos2' and 'simulation' sub-dicts plus 'abs_error_mins'.
        """
        self._ensure_breakpoints(battery_cap_kwh, max_power_kw)

        sos2_result = self.optimize_charging_schedule(
            start_soc_pct, target_soc_pct, battery_cap_kwh, max_power_kw
        )

        # Simulate with 0.1% SoC steps
        step = 0.1
        soc = start_soc_pct
        sim_time = 0.0
        sim_traj = [(soc, 0.0)]
        while soc < target_soc_pct:
            kwh_step = (step / 100.0) * battery_cap_kwh
            if soc < 80.0:
                power = max_power_kw
            else:
                ratio = (100.0 - soc) / 20.0
                power = max(5.0, max_power_kw * (ratio ** 2))
            dt = (kwh_step / power) * 60.0
            sim_time += dt
            soc = min(soc + step, target_soc_pct)
            sim_traj.append((round(soc, 2), round(sim_time, 3)))

        abs_err = abs(sos2_result['optimal_time_mins'] - sim_time)

        return {
            'sos2': sos2_result,
            'simulation': {
                'total_time_mins': round(sim_time, 3),
                'soc_trajectory': sim_traj[::10],  # subsample for readability
            },
            'abs_error_mins': round(abs_err, 4),
            'relative_error_pct': round(abs_err / max(sim_time, 1e-6) * 100.0, 3),
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    opt = SOS2ChargingOptimizer()
    opt.build_breakpoints(battery_cap_kwh=84.0, max_power_kw=150.0)
    print("Breakpoints:")
    for bp in opt.breakpoints:
        print(f"  e={bp[0]:.2f} kWh  t={bp[1]:.2f} min")

    result = opt.optimize_charging_schedule(20.0, 85.0)
    print(f"\nSOS2 LP  20%→85%: {result['optimal_time_mins']:.2f} min, "
          f"{result['energy_charged_kwh']:.2f} kWh charged")

    cmp = opt.compare_with_simulation(20.0, 85.0)
    print(f"Simulation:        {cmp['simulation']['total_time_mins']:.2f} min")
    print(f"Absolute error:    {cmp['abs_error_mins']:.4f} min  "
          f"({cmp['relative_error_pct']:.3f}%)")
