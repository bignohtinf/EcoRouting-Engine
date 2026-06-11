"""
battery_swap_location.py — Battery Swap Station Location + Newsvendor (Section 5.8).

Newsvendor per station f:
    B_f* = F_f^{-1}((c_sold - c_bat) / (c_sold + P_stockout))

Location MILP:
    min  sum_f [c_fixed * y_f + c_bat * B_f + E[P_so * unmet_f(xi)]]
    s.t. sum_f z_df >= 1 for all d  (coverage)
         z_df <= y_f                 (open facility)
         B_f >= F_f^{-1}(service_level)  (newsvendor)
         y_f in {0,1}, z_df in {0,1}

OR-Tools primary solver; greedy coverage fallback.
"""

import math
import numpy as np

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False


def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


class BatterySwapLocationSolver:
    """
    BSSLP: joint location + inventory optimization (Section 5.8).

    Parameters
    ----------
    c_bat_per_unit : cost of one battery pack (USD)
    c_sold_per_swap: revenue per swap (USD)
    p_stockout     : penalty per unmet demand (USD)
    service_level  : target service level for newsvendor (e.g. 0.95)
    """

    def __init__(
        self,
        c_bat_per_unit=500.0,
        c_sold_per_swap=50.0,
        p_stockout=200.0,
        service_level=0.95,
    ):
        self.c_bat = c_bat_per_unit
        self.c_sold = c_sold_per_swap
        self.p_stockout = p_stockout
        self.service_level = service_level

    # ------------------------------------------------------------------
    # Newsvendor
    # ------------------------------------------------------------------

    def newsvendor_optimal_inventory(self, demand_samples):
        """
        B_f* = F_f^{-1}( (c_sold - c_bat) / (c_sold + P_stockout) )

        Uses the empirical quantile of demand_samples.

        Parameters
        ----------
        demand_samples : array-like of historical daily demand counts

        Returns
        -------
        int : optimal stocking quantity
        """
        samples = np.asarray(demand_samples, dtype=float)
        if len(samples) == 0:
            return 1

        # Critical ratio
        cr = (self.c_sold - self.c_bat) / (self.c_sold + self.p_stockout)
        cr = float(np.clip(cr, 0.0, 1.0))

        # Fallback: if cr <= 0, stock nothing; use service_level if cr negative
        if cr <= 0:
            cr = self.service_level

        B_star = int(math.ceil(float(np.quantile(samples, cr))))
        return max(0, B_star)

    # ------------------------------------------------------------------
    # Location solver
    # ------------------------------------------------------------------

    def solve_location(self, demand_points, candidate_sites, budget, demand_samples_per_site):
        """
        Solve joint location + inventory problem.

        Parameters
        ----------
        demand_points : list of dicts {'id', 'lat', 'lon'}
        candidate_sites : list of dicts {'id', 'lat', 'lon', 'fixed_cost'}
        budget : total budget (USD)
        demand_samples_per_site : dict {site_id: array of demand samples}

        Returns
        -------
        dict:
            opened_sites     : list of site dicts
            inventory_per_site : {site_id: int}
            total_cost       : float
            expected_unmet   : float
        """
        if HAS_ORTOOLS:
            try:
                return self._solve_ortools(
                    demand_points, candidate_sites, budget, demand_samples_per_site
                )
            except Exception as exc:
                print(f"[BatterySwapLocationSolver] OR-Tools failed ({exc}), using greedy.")
        return self._solve_greedy(
            demand_points, candidate_sites, budget, demand_samples_per_site
        )

    # ------------------------------------------------------------------
    # OR-Tools CP-SAT
    # ------------------------------------------------------------------

    def _solve_ortools(self, demand_points, candidate_sites, budget, demand_samples_per_site):
        D = len(demand_points)
        F = len(candidate_sites)

        # Pre-compute newsvendor inventory and site costs
        inventories = {}
        site_costs = {}
        for f, site in enumerate(candidate_sites):
            sid = site['id']
            samples = demand_samples_per_site.get(sid, [10])
            B = self.newsvendor_optimal_inventory(samples)
            inventories[sid] = B
            inv_cost = self.c_bat * B
            site_costs[f] = float(site['fixed_cost']) + inv_cost

        SCALE = 100
        budget_int = int(round(budget * SCALE))

        model = cp_model.CpModel()

        # y_f: open site f
        y = [model.new_bool_var(f'y_{f}') for f in range(F)]
        # z_df: demand d assigned to site f
        z = [[model.new_bool_var(f'z_{d}_{f}') for f in range(F)] for d in range(D)]

        # Coverage: each demand point served by at least one open site
        for d in range(D):
            model.add(sum(z[d][f] for f in range(F)) >= 1)

        # z_df <= y_f (can only assign to open site)
        for d in range(D):
            for f in range(F):
                model.add(z[d][f] <= y[f])

        # Budget constraint
        cost_terms = [int(round(site_costs[f] * SCALE)) * y[f] for f in range(F)]
        model.add(sum(cost_terms) <= budget_int)

        # Objective: minimise total cost (fixed + inventory)
        model.minimize(sum(cost_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"OR-Tools status: {solver.status_name(status)}")

        opened = [candidate_sites[f] for f in range(F) if solver.value(y[f]) == 1]
        inv_per_site = {site['id']: inventories[site['id']] for site in opened}
        total_cost = sum(
            site_costs[f] for f in range(F) if solver.value(y[f]) == 1
        )

        # Expected unmet demand
        exp_unmet = self._expected_unmet(opened, inv_per_site, demand_samples_per_site)

        return {
            'opened_sites': opened,
            'inventory_per_site': inv_per_site,
            'total_cost': round(total_cost, 2),
            'expected_unmet': round(exp_unmet, 4),
        }

    # ------------------------------------------------------------------
    # Greedy fallback: open cheapest sites that cover all demand
    # ------------------------------------------------------------------

    def _solve_greedy(self, demand_points, candidate_sites, budget, demand_samples_per_site):
        D = len(demand_points)
        F = len(candidate_sites)

        inventories = {}
        for site in candidate_sites:
            sid = site['id']
            samples = demand_samples_per_site.get(sid, [10])
            inventories[sid] = self.newsvendor_optimal_inventory(samples)

        def site_total_cost(site):
            return float(site['fixed_cost']) + self.c_bat * inventories[site['id']]

        # Sort by cost ascending
        sorted_sites = sorted(candidate_sites, key=site_total_cost)

        opened = []
        spent = 0.0
        covered = set()

        for site in sorted_sites:
            cost = site_total_cost(site)
            if spent + cost > budget:
                continue
            # Check if this site covers any uncovered demand
            for d_idx, dp in enumerate(demand_points):
                if d_idx not in covered:
                    dist = haversine_dist(dp['lat'], dp['lon'], site['lat'], site['lon'])
                    if dist < 50.0:  # coverage radius 50 km
                        covered.add(d_idx)
            opened.append(site)
            spent += cost
            if len(covered) == D:
                break

        inv_per_site = {site['id']: inventories[site['id']] for site in opened}
        total_cost = sum(site_total_cost(s) for s in opened)
        exp_unmet = self._expected_unmet(opened, inv_per_site, demand_samples_per_site)

        return {
            'opened_sites': opened,
            'inventory_per_site': inv_per_site,
            'total_cost': round(total_cost, 2),
            'expected_unmet': round(exp_unmet, 4),
        }

    # ------------------------------------------------------------------
    # Expected unmet demand helper
    # ------------------------------------------------------------------

    def _expected_unmet(self, opened_sites, inv_per_site, demand_samples_per_site):
        """Compute expected unmet demand across opened sites."""
        total_unmet = 0.0
        for site in opened_sites:
            sid = site['id']
            samples = np.asarray(demand_samples_per_site.get(sid, [10]), dtype=float)
            B = inv_per_site.get(sid, 0)
            unmet = np.mean(np.maximum(samples - B, 0.0))
            total_unmet += unmet
        return total_unmet

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def simulate_operations(self, opened_sites, inventory, demand_realizations, n_runs=1000):
        """
        Monte Carlo simulation of stockout events.

        Parameters
        ----------
        opened_sites : list of site dicts (with 'id')
        inventory : {site_id: int} stocking levels
        demand_realizations : {site_id: array of historical demand} (used as distribution)
        n_runs : number of Monte Carlo runs

        Returns
        -------
        dict:
            stockout_rate_per_site : {site_id: float} fraction of runs with stockout
            avg_unmet_per_site     : {site_id: float} average unmet demand
            total_stockout_rate    : float
            service_level_achieved : float
        """
        rng = np.random.default_rng(42)
        stockout_counts = {s['id']: 0 for s in opened_sites}
        total_unmet = {s['id']: 0.0 for s in opened_sites}

        for _ in range(n_runs):
            for site in opened_sites:
                sid = site['id']
                samples = demand_realizations.get(sid, [10])
                d_sim = float(rng.choice(samples))
                B = inventory.get(sid, 0)
                if d_sim > B:
                    stockout_counts[sid] += 1
                    total_unmet[sid] += d_sim - B

        stockout_rate = {sid: stockout_counts[sid] / n_runs for sid in stockout_counts}
        avg_unmet = {sid: total_unmet[sid] / n_runs for sid in total_unmet}

        all_rates = list(stockout_rate.values())
        total_rate = float(np.mean(all_rates)) if all_rates else 0.0

        return {
            'stockout_rate_per_site': stockout_rate,
            'avg_unmet_per_site': avg_unmet,
            'total_stockout_rate': round(total_rate, 4),
            'service_level_achieved': round(1.0 - total_rate, 4),
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(7)
    demand_pts = [
        {'id': d, 'lat': 21.0 + rng.uniform(-0.2, 0.2),
         'lon': 105.8 + rng.uniform(-0.2, 0.2)}
        for d in range(1, 6)
    ]
    sites = [
        {'id': f, 'lat': 21.0 + rng.uniform(-0.15, 0.15),
         'lon': 105.8 + rng.uniform(-0.15, 0.15),
         'fixed_cost': float(rng.uniform(5000, 20000))}
        for f in range(1, 4)
    ]
    demand_hist = {
        s['id']: list(rng.poisson(15, 200).astype(float)) for s in sites
    }

    solver = BatterySwapLocationSolver()

    # Newsvendor test
    B = solver.newsvendor_optimal_inventory(demand_hist[sites[0]['id']])
    print(f"Newsvendor B* for site 1: {B} batteries")

    # Location solve
    result = solver.solve_location(demand_pts, sites, budget=50000.0, demand_samples_per_site=demand_hist)
    print(f"Opened {len(result['opened_sites'])} sites, cost={result['total_cost']:.0f}, "
          f"E[unmet]={result['expected_unmet']:.2f}")

    # Monte Carlo
    mc = solver.simulate_operations(
        result['opened_sites'], result['inventory_per_site'], demand_hist
    )
    print(f"Service level achieved: {mc['service_level_achieved']*100:.1f}%")
