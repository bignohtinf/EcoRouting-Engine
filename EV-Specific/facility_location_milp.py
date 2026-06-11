"""
facility_location_milp.py — Charging Network Planning MILP (FL1-FL5) (Section 5.7).

Capacitated Facility Location for EV charging network.

min  sum_f [c_fixed_f * y_f + c_cap_f * n_f]  +  sum_{d,f} c_travel_df * z_df

FL1: sum_f z_df >= 1 - eps_d   for all d  (coverage, eps=tolerance)
FL2: z_df <= y_f                            (only open facilities serve)
FL3: sum_d lambda_d * z_df <= mu_f * n_f   (capacity per station)
FL4: sum_f y_f <= K                         (max number of stations)
FL5: z_df = 0 if tau_df > R_max            (range feasibility, pre-filter)

y_f in {0,1} (open/closed), n_f integer >= 0 (charger count), z_df in {0,1}
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


class FacilityLocationMILP:
    """
    Capacitated Facility Location for EV charging network (Section 5.7).

    Parameters
    ----------
    R_max_km           : maximum reachable range for EV (range feasibility filter FL5)
    coverage_tolerance : epsilon in FL1 (allowed uncovered demand fraction, default 0.05)
    """

    def __init__(self, R_max_km=50.0, coverage_tolerance=0.05):
        self.R_max = R_max_km
        self.eps = coverage_tolerance

    # ------------------------------------------------------------------
    # Pre-processing: range filter (FL5)
    # ------------------------------------------------------------------

    def preprocess_range_filter(self, demand_points, candidate_sites):
        """
        Compute feasibility mask (n_demand x n_sites).
        feasible[d, f] = 1 if dist(d, f) <= R_max, else 0.

        This pre-filters arcs that violate FL5 before passing to the solver.

        Returns
        -------
        np.ndarray of shape (D, F), dtype int
        """
        D = len(demand_points)
        F = len(candidate_sites)
        mask = np.zeros((D, F), dtype=int)
        for d, dp in enumerate(demand_points):
            for f, site in enumerate(candidate_sites):
                dist = haversine_dist(dp['lat'], dp['lon'], site['lat'], site['lon'])
                if dist <= self.R_max:
                    mask[d, f] = 1
        return mask

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self, demand_points, candidate_sites, K, use_ortools=True):
        """
        Solve the capacitated facility location problem.

        Parameters
        ----------
        demand_points : list of dicts:
            {'id', 'lat', 'lon', 'load_kwh_per_day', 'range_anxiety_weight'}
        candidate_sites : list of dicts:
            {'id', 'lat', 'lon', 'fixed_cost', 'cap_cost_per_charger',
             'max_chargers', 'charger_capacity_kwh_per_day'}
        K : int — maximum number of stations to open (FL4)
        use_ortools : bool — prefer OR-Tools MILP solver

        Returns
        -------
        dict:
            opened          : list of opened site dicts
            chargers_per_site : {site_id: int}
            assignment      : {demand_id: site_id}
            total_cost      : float
        """
        feasibility = self.preprocess_range_filter(demand_points, candidate_sites)

        if use_ortools and HAS_ORTOOLS:
            try:
                return self._solve_ortools(demand_points, candidate_sites, K, feasibility)
            except Exception as exc:
                print(f"[FacilityLocationMILP] OR-Tools failed ({exc}), using greedy.")
        return self._solve_greedy(demand_points, candidate_sites, K, feasibility)

    # ------------------------------------------------------------------
    # OR-Tools CP-SAT MILP
    # ------------------------------------------------------------------

    def _solve_ortools(self, demand_points, candidate_sites, K, feasibility):
        D = len(demand_points)
        F = len(candidate_sites)

        # Pre-compute travel costs c_travel_df (distance-proportional)
        travel_cost = np.zeros((D, F))
        dist_matrix = np.zeros((D, F))
        for d, dp in enumerate(demand_points):
            for f, site in enumerate(candidate_sites):
                dist = haversine_dist(dp['lat'], dp['lon'], site['lat'], site['lon'])
                dist_matrix[d, f] = dist
                weight = float(dp.get('range_anxiety_weight', 1.0))
                travel_cost[d, f] = weight * dist

        SCALE = 10  # cost scale factor
        model = cp_model.CpModel()

        # y_f: open site f
        y = [model.new_bool_var(f'y_{f}') for f in range(F)]
        # n_f: number of chargers at site f (0..max_chargers)
        max_chargers_f = [int(site.get('max_chargers', 10)) for site in candidate_sites]
        n = [model.new_int_var(0, max_chargers_f[f], f'n_{f}') for f in range(F)]
        # z_df: demand d assigned to site f (FL5: only if feasible)
        z = [
            [model.new_bool_var(f'z_{d}_{f}') if feasibility[d, f] else model.new_constant(0)
             for f in range(F)]
            for d in range(D)
        ]

        # FL1: coverage (at least 1-eps of demand points covered)
        min_covered = max(1, int(math.ceil((1.0 - self.eps) * D)))
        covered_d = [
            model.new_bool_var(f'cov_{d}') for d in range(D)
        ]
        for d in range(D):
            # covered_d[d] = 1 iff sum_f z_df >= 1
            model.add(sum(z[d][f] for f in range(F)) >= covered_d[d])
            model.add(sum(z[d][f] for f in range(F)) <= D * covered_d[d])
        model.add(sum(covered_d) >= min_covered)

        # FL2: z_df <= y_f
        for d in range(D):
            for f in range(F):
                if feasibility[d, f]:
                    model.add(z[d][f] <= y[f])

        # FL3: capacity constraint sum_d lambda_d * z_df <= mu_f * n_f
        cap_per_charger = [
            float(site.get('charger_capacity_kwh_per_day', 200.0))
            for site in candidate_sites
        ]
        for f in range(F):
            demand_load = sum(
                int(round(float(demand_points[d].get('load_kwh_per_day', 50.0)))) * z[d][f]
                for d in range(D) if feasibility[d, f]
            )
            cap_int = int(round(cap_per_charger[f]))
            model.add(demand_load <= cap_int * n[f])

        # FL4: sum_f y_f <= K
        model.add(sum(y) <= K)

        # n_f = 0 if y_f = 0
        for f in range(F):
            model.add(n[f] <= max_chargers_f[f] * y[f])

        # Objective: min fixed costs + capacity costs + travel costs
        fixed_costs = sum(
            int(round(float(candidate_sites[f]['fixed_cost']) * SCALE)) * y[f]
            for f in range(F)
        )
        cap_costs = sum(
            int(round(float(candidate_sites[f]['cap_cost_per_charger']) * SCALE)) * n[f]
            for f in range(F)
        )
        travel_costs = sum(
            int(round(travel_cost[d, f] * SCALE)) * z[d][f]
            for d in range(D) for f in range(F) if feasibility[d, f]
        )
        model.minimize(fixed_costs + cap_costs + travel_costs)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60.0
        solver.parameters.num_workers = 4
        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"OR-Tools status: {solver.status_name(status)}")

        # Extract
        opened = [candidate_sites[f] for f in range(F) if solver.value(y[f]) == 1]
        chargers = {
            candidate_sites[f]['id']: solver.value(n[f])
            for f in range(F) if solver.value(y[f]) == 1
        }
        assignment = {}
        for d in range(D):
            for f in range(F):
                if feasibility[d, f] and solver.value(z[d][f]) == 1:
                    assignment[demand_points[d]['id']] = candidate_sites[f]['id']
                    break

        total_cost = sum(
            float(candidate_sites[f]['fixed_cost']) +
            float(candidate_sites[f]['cap_cost_per_charger']) * solver.value(n[f])
            for f in range(F) if solver.value(y[f]) == 1
        )
        total_cost += sum(
            travel_cost[d, f]
            for d in range(D) for f in range(F)
            if feasibility[d, f] and solver.value(z[d][f]) == 1
        )

        return {
            'opened': opened,
            'chargers_per_site': chargers,
            'assignment': assignment,
            'total_cost': round(total_cost, 2),
        }

    # ------------------------------------------------------------------
    # Greedy p-median heuristic
    # ------------------------------------------------------------------

    def _solve_greedy(self, demand_points, candidate_sites, K, feasibility):
        """
        Greedy p-median: iteratively open the site that maximally reduces
        total weighted travel cost.
        """
        D = len(demand_points)
        F = len(candidate_sites)

        dist_matrix = np.zeros((D, F))
        for d, dp in enumerate(demand_points):
            for f, site in enumerate(candidate_sites):
                dist_matrix[d, f] = haversine_dist(
                    dp['lat'], dp['lon'], site['lat'], site['lon']
                )

        weights = np.array([float(dp.get('range_anxiety_weight', 1.0)) for dp in demand_points])
        travel_cost = dist_matrix * weights[:, np.newaxis] * feasibility

        opened_f = []
        remaining_f = list(range(F))

        for _ in range(min(K, F)):
            if not remaining_f:
                break
            best_f = None
            best_reduction = -np.inf
            for f in remaining_f:
                candidate_open = opened_f + [f]
                # For each demand, cost = min over open sites (with feasibility)
                cost_mat = travel_cost[:, candidate_open]
                # Handle infeasible (zero in feasibility mask → set to large)
                feasible_open = feasibility[:, candidate_open]
                inf_val = 1e9
                adjusted = np.where(feasible_open > 0, cost_mat,
                                    np.full_like(cost_mat, inf_val))
                min_costs = adjusted.min(axis=1)
                total = float(np.sum(np.where(min_costs < inf_val, min_costs, 0.0)))
                if -total > best_reduction:
                    best_reduction = -total
                    best_f = f
            if best_f is not None:
                opened_f.append(best_f)
                remaining_f.remove(best_f)

        opened = [candidate_sites[f] for f in opened_f]

        # Assign each demand to nearest open site
        assignment = {}
        for d, dp in enumerate(demand_points):
            best_dist = np.inf
            best_site = None
            for f in opened_f:
                if feasibility[d, f] and dist_matrix[d, f] < best_dist:
                    best_dist = dist_matrix[d, f]
                    best_site = candidate_sites[f]['id']
            if best_site is not None:
                assignment[dp['id']] = best_site

        # Determine charger counts from load
        load_per_site = {s['id']: 0.0 for s in opened}
        for d, dp in enumerate(demand_points):
            sid = assignment.get(dp['id'])
            if sid is not None:
                load_per_site[sid] += float(dp.get('load_kwh_per_day', 50.0))

        chargers = {}
        for site in opened:
            sid = site['id']
            cap = float(site.get('charger_capacity_kwh_per_day', 200.0))
            n_needed = max(1, math.ceil(load_per_site.get(sid, 0.0) / max(cap, 1.0)))
            n_needed = min(n_needed, int(site.get('max_chargers', 10)))
            chargers[sid] = n_needed

        total_cost = sum(
            float(site['fixed_cost']) + float(site['cap_cost_per_charger']) * chargers[site['id']]
            for site in opened
        )
        total_cost += sum(
            dist_matrix[d, f]
            for d in range(D) for f in opened_f
            if feasibility[d, f] and demand_points[d]['id'] in assignment
            and assignment[demand_points[d]['id']] == candidate_sites[f]['id']
        )

        return {
            'opened': opened,
            'chargers_per_site': chargers,
            'assignment': assignment,
            'total_cost': round(total_cost, 2),
        }

    # ------------------------------------------------------------------
    # Solver comparison
    # ------------------------------------------------------------------

    def compare_solvers(self, demand_points, candidate_sites, K):
        """
        Run both OR-Tools MILP and greedy heuristic; return comparison dict.

        Returns
        -------
        dict:
            ortools  : result dict (or error string)
            greedy   : result dict
            cost_gap : relative gap (ortools vs greedy)
        """
        feasibility = self.preprocess_range_filter(demand_points, candidate_sites)

        # Greedy
        greedy_result = self._solve_greedy(demand_points, candidate_sites, K, feasibility)

        # OR-Tools
        if HAS_ORTOOLS:
            try:
                ortools_result = self._solve_ortools(
                    demand_points, candidate_sites, K, feasibility
                )
            except Exception as exc:
                ortools_result = {'error': str(exc), 'total_cost': float('inf')}
        else:
            ortools_result = {'error': 'OR-Tools not available', 'total_cost': float('inf')}

        # Cost gap
        c_or = float(ortools_result.get('total_cost', float('inf')))
        c_gr = float(greedy_result.get('total_cost', float('inf')))
        if c_gr > 0 and c_or < float('inf'):
            gap = (c_gr - c_or) / max(c_gr, 1e-6)
        else:
            gap = float('nan')

        return {
            'ortools': ortools_result,
            'greedy': greedy_result,
            'cost_gap_pct': round(gap * 100, 3),
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(99)
    center = (21.0285, 105.8522)

    demand_pts = [
        {
            'id': d,
            'lat': center[0] + rng.uniform(-0.3, 0.3),
            'lon': center[1] + rng.uniform(-0.3, 0.3),
            'load_kwh_per_day': float(rng.uniform(30, 120)),
            'range_anxiety_weight': float(rng.uniform(0.5, 2.0)),
        }
        for d in range(1, 9)
    ]
    sites = [
        {
            'id': f,
            'lat': center[0] + rng.uniform(-0.25, 0.25),
            'lon': center[1] + rng.uniform(-0.25, 0.25),
            'fixed_cost': float(rng.uniform(10000, 50000)),
            'cap_cost_per_charger': float(rng.uniform(2000, 8000)),
            'max_chargers': int(rng.integers(3, 10)),
            'charger_capacity_kwh_per_day': 200.0,
        }
        for f in range(1, 6)
    ]

    solver = FacilityLocationMILP(R_max_km=60.0)
    mask = solver.preprocess_range_filter(demand_pts, sites)
    print(f"Feasibility mask shape: {mask.shape}, feasible arcs: {mask.sum()}")

    cmp = solver.compare_solvers(demand_pts, sites, K=3)
    print(f"Greedy cost:   {cmp['greedy']['total_cost']:,.0f}")
    print(f"OR-Tools cost: {cmp['ortools'].get('total_cost', 'N/A')}")
    print(f"Cost gap:      {cmp['cost_gap_pct']:.2f}%")
    print(f"Opened sites: {[s['id'] for s in cmp['greedy']['opened']]}")
