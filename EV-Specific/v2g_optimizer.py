"""
v2g_optimizer.py — Vehicle-to-Grid (V2G) Optimizer (Section 5.9).

Decision variables per vehicle k, time slot t:
  p_G2V_kt >= 0  : charging power  (Grid-to-Vehicle, kW)
  p_V2G_kt >= 0  : discharging power (Vehicle-to-Grid, kW)
  b_kt in {0,1}  : 1=charging mode, 0=discharging mode (linearises V4)

SoC dynamics:
  e_k(t+1) = e_k(t) + eta_plus * p_G2V * dt - (1/eta_minus) * p_V2G * dt

Constraints:
  V1: e_min <= e_k(t) <= e_max
  V2: p_G2V + p_V2G <= P_max
  V3: e_k(t_depart) >= e_depart
  V4 linearised: p_G2V <= P_max * b_kt;  p_V2G <= P_max * (1 - b_kt)

Objective: max sum_k sum_t [lambda_sell_t * p_V2G_kt - lambda_buy_t * p_G2V_kt] * dt
"""

import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_SCIPY_MILP = True
except ImportError:
    HAS_SCIPY_MILP = False


def haversine_dist(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


class V2GOptimizer:
    """
    V2G optimization (Section 5.9): maximise revenue from grid interaction during parking.

    Primary solver  : scipy.optimize.milp (if available)
    Fallback        : greedy (charge cheap slots, discharge expensive slots)
    """

    def __init__(self, eta_charge=0.95, eta_discharge=0.92, delta_t_hours=0.25):
        """
        Parameters
        ----------
        eta_charge    : charging efficiency (eta_plus)
        eta_discharge : discharging efficiency (eta_minus)
        delta_t_hours : time slot duration in hours (default 15 min = 0.25 h)
        """
        self.eta_plus = eta_charge
        self.eta_minus = eta_discharge
        self.dt = delta_t_hours

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def optimize(self, vehicles_schedule, electricity_prices):
        """
        Optimise V2G schedule for all parked vehicles.

        Parameters
        ----------
        vehicles_schedule : list of dicts with keys:
            id, t_arrive, t_depart, soc_arrive (%), soc_depart_min (%),
            e_max_kwh, p_max_kw, e_min_kwh
        electricity_prices : dict with keys:
            'buy'  : np.ndarray shape (T,)  buy price ($/kWh)
            'sell' : np.ndarray shape (T,)  sell price ($/kWh)

        Returns
        -------
        dict:
            v2g_schedule : {vehicle_id: {'p_G2V': array, 'p_V2G': array, 'soc': array}}
            total_revenue : float
            grid_impact   : np.ndarray(T,) net load on grid (kW, positive=drawn from grid)
        """
        lambda_buy = np.asarray(electricity_prices['buy'], dtype=float)
        lambda_sell = np.asarray(electricity_prices['sell'], dtype=float)
        T = len(lambda_buy)

        if HAS_SCIPY_MILP:
            try:
                return self._solve_scipy(vehicles_schedule, lambda_buy, lambda_sell, T)
            except Exception as exc:
                print(f"[V2GOptimizer] scipy.milp failed ({exc}), using greedy fallback.")
        return self._solve_greedy(vehicles_schedule, lambda_buy, lambda_sell, T)

    # ------------------------------------------------------------------
    # scipy.optimize.milp solver
    # ------------------------------------------------------------------

    def _solve_scipy(self, vehicles_schedule, lambda_buy, lambda_sell, T):
        """
        Build and solve MILP using scipy.optimize.milp.

        Variable layout per vehicle k (K vehicles):
          [p_G2V_k0 .. p_G2V_k(T-1)] length T  (continuous >= 0)
          [p_V2G_k0 .. p_V2G_k(T-1)] length T  (continuous >= 0)
          [b_k0     .. b_k(T-1)    ] length T  (binary / integer 0,1)
          [e_k0     .. e_k(T-1)    ] length T  (continuous, SoC kWh)
        Total per vehicle: 4*T
        Grand total: K * 4 * T variables.
        """
        K = len(vehicles_schedule)
        n_vars = K * 4 * T

        # Objective coefficients (negated for minimisation)
        # Revenue = sum_t (sell * p_V2G - buy * p_G2V) * dt
        c = np.zeros(n_vars)
        for k in range(K):
            base = k * 4 * T
            # p_G2V: cost = +buy (we pay)
            c[base:base+T] = lambda_buy * self.dt
            # p_V2G: revenue = -sell (we receive)
            c[base+T:base+2*T] = -lambda_sell * self.dt

        # Variable bounds
        lb = np.zeros(n_vars)
        ub = np.full(n_vars, np.inf)
        integrality = np.zeros(n_vars)  # 0=continuous, 1=integer

        for k, v in enumerate(vehicles_schedule):
            base = k * 4 * T
            ta, td = int(v['t_arrive']), int(v['t_depart'])
            pmax = float(v['p_max_kw'])
            emax = float(v['e_max_kwh'])
            emin = float(v['e_min_kwh'])

            # p_G2V bounds: 0 outside parking, [0, pmax] during parking
            for t in range(T):
                if ta <= t < td:
                    ub[base + t] = pmax          # p_G2V
                    ub[base + T + t] = pmax      # p_V2G
                else:
                    ub[base + t] = 0.0
                    ub[base + T + t] = 0.0

            # b_kt binary
            for t in range(T):
                integrality[base + 2*T + t] = 1
                lb[base + 2*T + t] = 0.0
                ub[base + 2*T + t] = 1.0

            # e_kt SoC bounds
            for t in range(T):
                lb[base + 3*T + t] = emin
                ub[base + 3*T + t] = emax

        bounds = Bounds(lb=lb, ub=ub)

        # Constraints (linear equalities / inequalities)
        A_rows, lb_c, ub_c = [], [], []

        for k, v in enumerate(vehicles_schedule):
            base = k * 4 * T
            ta, td = int(v['t_arrive']), int(v['t_depart'])
            pmax = float(v['p_max_kw'])
            emax = float(v['e_max_kwh'])
            emin = float(v['e_min_kwh'])
            e_arrive = float(v['soc_arrive']) / 100.0 * emax
            e_depart_min = float(v['soc_depart_min']) / 100.0 * emax

            # V1: SoC initial condition (e_k at t_arrive = e_arrive)
            if ta < T:
                row = np.zeros(n_vars)
                row[base + 3*T + ta] = 1.0
                A_rows.append(row)
                lb_c.append(e_arrive)
                ub_c.append(e_arrive)

            # SoC dynamics: e_k(t+1) = e_k(t) + eta+ * p_G2V(t) - (1/eta-) * p_V2G(t)
            for t in range(ta, min(td - 1, T - 1)):
                row = np.zeros(n_vars)
                row[base + 3*T + t + 1] = 1.0      # e(t+1)
                row[base + 3*T + t] = -1.0          # -e(t)
                row[base + t] = -self.eta_plus * self.dt    # -eta+ * p_G2V
                row[base + T + t] = (1.0/self.eta_minus) * self.dt  # +(1/eta-) * p_V2G
                A_rows.append(row)
                lb_c.append(0.0)
                ub_c.append(0.0)

            # V3: e_k(t_depart) >= e_depart_min
            if td - 1 < T:
                row = np.zeros(n_vars)
                row[base + 3*T + td - 1] = 1.0
                A_rows.append(row)
                lb_c.append(e_depart_min)
                ub_c.append(emax)

            # V2: p_G2V + p_V2G <= P_max  (for each slot)
            for t in range(ta, td):
                if t >= T:
                    break
                row = np.zeros(n_vars)
                row[base + t] = 1.0
                row[base + T + t] = 1.0
                A_rows.append(row)
                lb_c.append(0.0)
                ub_c.append(pmax)

            # V4 linearised: p_G2V <= P_max * b_kt
            for t in range(ta, td):
                if t >= T:
                    break
                row = np.zeros(n_vars)
                row[base + t] = 1.0
                row[base + 2*T + t] = -pmax
                A_rows.append(row)
                lb_c.append(-np.inf)
                ub_c.append(0.0)

            # V4 linearised: p_V2G <= P_max * (1 - b_kt)
            for t in range(ta, td):
                if t >= T:
                    break
                row = np.zeros(n_vars)
                row[base + T + t] = 1.0
                row[base + 2*T + t] = pmax
                A_rows.append(row)
                lb_c.append(-np.inf)
                ub_c.append(pmax)

        if not A_rows:
            A_rows = [np.zeros(n_vars)]
            lb_c = [0.0]
            ub_c = [0.0]

        A_mat = np.array(A_rows)
        constraints = LinearConstraint(A_mat, lb=np.array(lb_c), ub=np.array(ub_c))

        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

        if not res.success:
            print(f"[V2GOptimizer] MILP solver: {res.message} — falling back to greedy.")
            return self._solve_greedy(vehicles_schedule, lambda_buy, lambda_sell, T)

        # Extract results
        x = res.x
        v2g_schedule = {}
        total_revenue = 0.0
        grid_impact = np.zeros(T)

        for k, v in enumerate(vehicles_schedule):
            base = k * 4 * T
            p_g2v = x[base:base+T]
            p_v2g = x[base+T:base+2*T]
            soc_arr = x[base+3*T:base+4*T]
            rev = float(np.sum((lambda_sell * p_v2g - lambda_buy * p_g2v) * self.dt))
            total_revenue += rev
            grid_impact += p_g2v - p_v2g
            v2g_schedule[v['id']] = {
                'p_G2V': p_g2v,
                'p_V2G': p_v2g,
                'soc': soc_arr / float(v['e_max_kwh']) * 100.0,
            }

        return {
            'v2g_schedule': v2g_schedule,
            'total_revenue': round(total_revenue, 4),
            'grid_impact': grid_impact,
        }

    # ------------------------------------------------------------------
    # Greedy fallback: charge cheap, discharge expensive
    # ------------------------------------------------------------------

    def _solve_greedy(self, vehicles_schedule, lambda_buy, lambda_sell, T):
        """
        Greedy heuristic:
        1. Sort slots by buy price ascending -> charge in cheapest slots.
        2. Sort slots by sell price descending -> discharge in most expensive.
        3. Respect SoC dynamics, V1-V4 constraints.
        """
        v2g_schedule = {}
        total_revenue = 0.0
        grid_impact = np.zeros(T)

        for v in vehicles_schedule:
            ta, td = int(v['t_arrive']), int(v['t_depart'])
            pmax = float(v['p_max_kw'])
            emax = float(v['e_max_kwh'])
            emin = float(v['e_min_kwh'])
            e = float(v['soc_arrive']) / 100.0 * emax
            e_depart_min = float(v['soc_depart_min']) / 100.0 * emax

            parking_slots = [t for t in range(ta, min(td, T))]
            p_g2v = np.zeros(T)
            p_v2g = np.zeros(T)
            soc = np.full(T, emin)
            if ta < T:
                soc[ta] = e

            # Phase 1: charge to ensure departure SoC constraint
            for t in parking_slots:
                deficit = e_depart_min - e
                if deficit > 0:
                    charge = min(pmax, deficit / (self.eta_plus * self.dt))
                    charge = max(0.0, charge)
                    p_g2v[t] = charge
                    e += self.eta_plus * charge * self.dt
                    e = min(e, emax)
                soc[t] = e

            # Phase 2: exploit price spread
            sorted_cheap = sorted(parking_slots, key=lambda t: lambda_buy[t])
            sorted_expensive = sorted(parking_slots, key=lambda t: lambda_sell[t], reverse=True)

            for t in sorted_cheap:
                if e < emax - 1e-6 and p_g2v[t] == 0 and p_v2g[t] == 0:
                    space = emax - e
                    charge = min(pmax, space / (self.eta_plus * self.dt))
                    p_g2v[t] = max(0.0, charge)
                    e += self.eta_plus * charge * self.dt
                    e = min(e, emax)
                    soc[t] = e

            for t in sorted_expensive:
                if e - emin > 1e-6 and p_v2g[t] == 0 and p_g2v[t] == 0:
                    # Ensure we keep enough for departure
                    remaining_slots = len([s for s in parking_slots if s >= t])
                    e_need = e_depart_min
                    available = e - e_need
                    if available > 0:
                        discharge = min(pmax, available * self.eta_minus / self.dt)
                        p_v2g[t] = max(0.0, discharge)
                        e -= (1.0 / self.eta_minus) * discharge * self.dt
                        e = max(e, emin)
                        soc[t] = e

            rev = float(np.sum((lambda_sell * p_v2g - lambda_buy * p_g2v) * self.dt))
            total_revenue += rev
            grid_impact += p_g2v - p_v2g
            v2g_schedule[v['id']] = {
                'p_G2V': p_g2v,
                'p_V2G': p_v2g,
                'soc': soc / emax * 100.0,
            }

        return {
            'v2g_schedule': v2g_schedule,
            'total_revenue': round(total_revenue, 4),
            'grid_impact': grid_impact,
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    T = 24  # 24 quarter-hour slots = 6 hours
    rng = np.random.default_rng(0)
    prices = {
        'buy': 0.08 + 0.04 * np.sin(np.linspace(0, 2*np.pi, T)),
        'sell': 0.06 + 0.05 * np.sin(np.linspace(np.pi/4, 2*np.pi + np.pi/4, T)),
    }
    vehicles = [
        {'id': 'V1', 't_arrive': 2, 't_depart': 20, 'soc_arrive': 60.0,
         'soc_depart_min': 70.0, 'e_max_kwh': 84.0, 'p_max_kw': 11.0, 'e_min_kwh': 8.4},
        {'id': 'V2', 't_arrive': 4, 't_depart': 18, 'soc_arrive': 80.0,
         'soc_depart_min': 65.0, 'e_max_kwh': 60.0, 'p_max_kw': 7.4, 'e_min_kwh': 6.0},
    ]
    opt = V2GOptimizer()
    result = opt.optimize(vehicles, prices)
    print(f"Total V2G revenue: ${result['total_revenue']:.4f}")
    for vid, sched in result['v2g_schedule'].items():
        g2v = sched['p_G2V'].sum()
        v2g = sched['p_V2G'].sum()
        print(f"  {vid}: G2V={g2v:.2f} kW·slots, V2G={v2g:.2f} kW·slots")
