"""
evrp_milp.py — E-VRP-CS MILP solver with SoC McCormick constraints (Section 5.2).

Primary solver: OR-Tools CP-SAT (integer-scaled variables).
Fallback: greedy nearest-neighbour (mirrors EVRPSolver logic).
"""

import math
import itertools
import numpy as np

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def haversine_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EVRPCSMILPSolver:
    """
    Electric VRP with Charging Stations — MILP formulation with SoC McCormick
    constraints (Section 5.2).

    Decision variables
    ------------------
    x_ijk  ∈ {0,1}   : vehicle k traverses arc i→j
    e_ki   ≥ 0       : SoC (kWh) at node i for vehicle k
    p_ijk  ∈ [0,1]   : auxiliary linearisation of  h_ij · x_ijk

    McCormick envelope for  p_ijk = h_ij · x_ijk
    -----------------------------------------------
    P1: p_ijk ≤ h_ij
    P2: p_ijk ≤ M · x_ijk
    P3: p_ijk ≥ h_ij − M · (1 − x_ijk)
    P4: p_ijk ≥ 0

    SoC continuity (E1a/b collapse to equality when x_ijk=1)
    ---------------------------------------------------------
    E1a: e_kj ≤ e_ki − p_ijk + M · (1 − x_ijk)
    E1b: e_kj ≥ e_ki − p_ijk − M · (1 − x_ijk)

    OR-Tools CP-SAT note: all floats scaled by SCALE=1000 → integers.
    SoC stored as integer in [0, 10000] representing [0%, 100%].
    """

    SCALE = 1000          # float → int scaling for CP-SAT
    SOC_SCALE = 100       # SoC% → integer  (100 units = 1%)

    def __init__(
        self,
        battery_cap_kwh: float = 84.0,
        min_soc_pct: float = 15.0,
        max_soc_pct: float = 95.0,
        speed_kmh: float = 30.0,
        M: float = 1000.0,
    ):
        self.battery_cap = battery_cap_kwh
        self.min_soc_pct = min_soc_pct
        self.max_soc_pct = max_soc_pct
        self.e_min = battery_cap_kwh * min_soc_pct / 100.0   # kWh
        self.e_max = battery_cap_kwh * max_soc_pct / 100.0   # kWh
        self.speed = speed_kmh
        self.M = M

    # ------------------------------------------------------------------
    # Energy pre-computation
    # ------------------------------------------------------------------

    def precompute_energy(self, nodes: list, energy_model=None) -> np.ndarray:
        """
        Compute h_ij matrix (energy consumed kWh on arc i→j).

        Parameters
        ----------
        nodes : list of dicts with keys 'lat', 'lon'
        energy_model : EVEnergyModel instance or None (uses simple α·d model)

        Returns
        -------
        h : ndarray shape (n, n)
        """
        n = len(nodes)
        h = np.zeros((n, n))
        alpha = 0.15  # kWh/km default

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = haversine_dist(
                    nodes[i]['lat'], nodes[i]['lon'],
                    nodes[j]['lat'], nodes[j]['lon'],
                )
                if energy_model is not None:
                    duration_h = d / self.speed
                    h[i, j] = energy_model.calculate_consumption(
                        dist_km=d,
                        elevation_gain_m=0.0,
                        speed_kmh=self.speed,
                        load_kg=80.0,
                        duration_h=duration_h,
                    )
                else:
                    h[i, j] = alpha * d
        return h

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        orders: list,
        charging_stations: list,
        vehicles: list,
        energy_model=None,
    ) -> dict:
        """
        Solve E-VRP-CS with MILP (OR-Tools) or greedy fallback.

        Parameters
        ----------
        orders : list of dicts {'id', 'lat', 'lon'}
        charging_stations : list of dicts {'id', 'lat', 'lon', 'max_power_kw'}
        vehicles : list of dicts {'id', 'depot_lat', 'depot_lon', 'initial_soc_pct'}
        energy_model : EVEnergyModel or None

        Returns
        -------
        dict : {vehicle_id: {'route', 'soc_trajectory', 'charging_events'}}
        """
        if HAS_ORTOOLS:
            try:
                return self._solve_ortools(orders, charging_stations, vehicles, energy_model)
            except Exception as exc:
                print(f"[EVRPCSMILPSolver] OR-Tools failed ({exc}), using greedy fallback.")
        return self._solve_greedy(orders, charging_stations, vehicles, energy_model)

    # ------------------------------------------------------------------
    # OR-Tools CP-SAT solver
    # ------------------------------------------------------------------

    def _solve_ortools(
        self,
        orders: list,
        charging_stations: list,
        vehicles: list,
        energy_model=None,
    ) -> dict:
        """
        Build and solve CP-SAT MILP.  Returns per-vehicle solution dict.
        """
        # --- Build node list: depot(s) + orders + charging stations -------
        # Node layout: [depot_0 ... depot_{K-1}] [order_0 ... order_{C-1}]
        #              [cs_0 ... cs_{S-1}]
        K = len(vehicles)
        C = len(orders)
        S = len(charging_stations)

        depot_nodes = [
            {'type': 'depot', 'id': v['id'],
             'lat': v['depot_lat'], 'lon': v['depot_lon']}
            for v in vehicles
        ]
        order_nodes = [
            {'type': 'order', 'id': o['id'], 'lat': o['lat'], 'lon': o['lon']}
            for o in orders
        ]
        cs_nodes = [
            {'type': 'cs', 'id': cs['id'], 'lat': cs['lat'], 'lon': cs['lon'],
             'max_power_kw': cs['max_power_kw']}
            for cs in charging_stations
        ]

        # For a single-depot case we duplicate depot for each vehicle
        # Full node list per vehicle  (each vehicle has its own depot idx=0)
        # To keep formulation tractable for CP-SAT we use a single shared
        # depot at index 0 (centroid of vehicle depots).
        depot_lat = np.mean([v['depot_lat'] for v in vehicles])
        depot_lon = np.mean([v['depot_lon'] for v in vehicles])
        nodes = (
            [{'type': 'depot', 'lat': depot_lat, 'lon': depot_lon}]
            + order_nodes
            + cs_nodes
        )
        N = len(nodes)  # total nodes including depot

        h = self.precompute_energy(nodes, energy_model)

        # Scale for CP-SAT integers
        SC = self.SCALE
        h_int = np.round(h * SC).astype(int)
        e_min_int = int(round(self.e_min * SC))
        e_max_int = int(round(self.e_max * SC))
        M_int = int(round(self.M * SC))

        model = cp_model.CpModel()

        # ---- Decision variables -------------------------------------------
        # x[k][i][j] : vehicle k uses arc i→j
        x = [
            [[model.new_bool_var(f'x_{k}_{i}_{j}') for j in range(N)]
             for i in range(N)]
            for k in range(K)
        ]

        # e[k][i] : SoC (scaled) at node i for vehicle k
        e = [
            [model.new_int_var(e_min_int, e_max_int, f'e_{k}_{i}') for i in range(N)]
            for k in range(K)
        ]

        # p[k][i][j] : auxiliary for h_ij * x_ijk  (scaled, upper-bounded by h_int)
        p = [
            [[model.new_int_var(0, max(1, h_int[i, j]), f'p_{k}_{i}_{j}') for j in range(N)]
             for i in range(N)]
            for k in range(K)
        ]

        # ---- Objective: minimise total energy consumed  --------------------
        total_energy = []
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    if i != j:
                        total_energy.append(p[k][i][j])
        model.minimize(sum(total_energy))

        # ---- Flow constraints ----------------------------------------------
        # Each order node visited exactly once across all vehicles
        order_start = 1
        order_end = 1 + C
        for j in range(order_start, order_end):
            model.add(
                sum(x[k][i][j] for k in range(K) for i in range(N) if i != j) == 1
            )

        # Flow conservation: in-degree = out-degree at every non-depot node
        for k in range(K):
            for v_node in range(1, N):
                model.add(
                    sum(x[k][i][v_node] for i in range(N) if i != v_node) ==
                    sum(x[k][v_node][j] for j in range(N) if j != v_node)
                )
            # Depot: each vehicle leaves and returns exactly once
            model.add(sum(x[k][0][j] for j in range(1, N)) <= 1)
            model.add(sum(x[k][i][0] for i in range(1, N)) <= 1)
            # No self-loops
            for i in range(N):
                model.add(x[k][i][i] == 0)

        # ---- McCormick SoC constraints  ------------------------------------
        for k in range(K):
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    hij = h_int[i, j]
                    # P1
                    model.add(p[k][i][j] <= hij)
                    # P2
                    model.add(p[k][i][j] <= M_int * x[k][i][j])
                    # P3
                    model.add(p[k][i][j] >= hij - M_int * (1 - x[k][i][j]))
                    # P4  (lower bound 0 encoded in variable definition)

                    # E1a: e[k][j] <= e[k][i] - p[k][i][j] + M*(1-x[k][i][j])
                    model.add(
                        e[k][j] <= e[k][i] - p[k][i][j] + M_int * (1 - x[k][i][j])
                    )
                    # E1b: e[k][j] >= e[k][i] - p[k][i][j] - M*(1-x[k][i][j])
                    model.add(
                        e[k][j] >= e[k][i] - p[k][i][j] - M_int * (1 - x[k][i][j])
                    )

        # Initial SoC at depot
        for k, v in enumerate(vehicles):
            init_e_int = int(round(v.get('initial_soc_pct', 90.0) / 100.0 * self.battery_cap * SC))
            init_e_int = max(e_min_int, min(e_max_int, init_e_int))
            model.add(e[k][0] == init_e_int)

        # ---- Solve ---------------------------------------------------------
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        solver.parameters.num_workers = 4
        status = solver.solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"CP-SAT returned status {solver.status_name(status)}")

        # ---- Extract solution ----------------------------------------------
        results = {}
        for k, v in enumerate(vehicles):
            route = []
            soc_traj = []
            charging_events = []

            # Reconstruct route by following x arcs from depot
            curr = 0
            visited = {0}
            soc_traj.append(solver.value(e[k][0]) / SC)
            for _ in range(N):
                next_node = None
                for j in range(N):
                    if j not in visited and solver.value(x[k][curr][j]) == 1:
                        next_node = j
                        break
                if next_node is None:
                    break
                node_info = nodes[next_node]
                if node_info['type'] == 'order':
                    route.append(f"Order_{node_info['id']}")
                elif node_info['type'] == 'cs':
                    route.append(f"CS_{node_info['id']}")
                    charging_events.append({
                        'station_id': node_info['id'],
                        'node_idx': next_node,
                        'soc_at_arrival_kwh': solver.value(e[k][next_node]) / SC,
                    })
                elif node_info['type'] == 'depot':
                    route.append('Depot')
                soc_traj.append(solver.value(e[k][next_node]) / SC)
                visited.add(next_node)
                curr = next_node

            results[v['id']] = {
                'route': route,
                'soc_trajectory': soc_traj,
                'charging_events': charging_events,
            }

        return results

    # ------------------------------------------------------------------
    # Greedy fallback (mirrors EVRPSolver nearest-neighbour logic)
    # ------------------------------------------------------------------

    def _solve_greedy(
        self,
        orders: list,
        charging_stations: list,
        vehicles: list,
        energy_model=None,
    ) -> dict:
        """
        Nearest-neighbour greedy with automatic charging station insertion.
        Mirrors the logic of EVRPSolver.solve_route_with_charging.
        """
        alpha = 0.15  # kWh/km

        def energy_kw(dist_km):
            if energy_model is not None:
                dur = dist_km / self.speed
                return energy_model.calculate_consumption(dist_km, 0.0, self.speed, 80.0, dur)
            return alpha * dist_km

        results = {}
        remaining_orders = list(orders)

        for v in vehicles:
            route = []
            soc_traj = []
            charging_events = []

            curr_lat = v['depot_lat']
            curr_lon = v['depot_lon']
            curr_soc = v.get('initial_soc_pct', 90.0) / 100.0 * self.battery_cap
            soc_traj.append(curr_soc / self.battery_cap * 100.0)

            unvisited = list(remaining_orders)

            while unvisited:
                # Nearest unvisited order
                best_order = min(
                    unvisited,
                    key=lambda o: haversine_dist(curr_lat, curr_lon, o['lat'], o['lon']),
                )
                d_next = haversine_dist(curr_lat, curr_lon, best_order['lat'], best_order['lon'])
                e_next = energy_kw(d_next)

                # Check if we need to charge first
                if curr_soc - e_next < self.e_min:
                    # Find nearest charging station reachable with current SoC
                    reachable_cs = [
                        cs for cs in charging_stations
                        if curr_soc - energy_kw(
                            haversine_dist(curr_lat, curr_lon, cs['lat'], cs['lon'])
                        ) >= self.e_min
                    ]
                    if not reachable_cs:
                        # Cannot continue — stop this vehicle
                        break
                    best_cs = min(
                        reachable_cs,
                        key=lambda cs: haversine_dist(curr_lat, curr_lon, cs['lat'], cs['lon']),
                    )
                    d_cs = haversine_dist(curr_lat, curr_lon, best_cs['lat'], best_cs['lon'])
                    e_cs = energy_kw(d_cs)
                    curr_soc -= e_cs
                    soc_traj.append(curr_soc / self.battery_cap * 100.0)
                    route.append(f"CS_{best_cs['id']}")

                    # Charge to 85%
                    target_soc = 0.85 * self.battery_cap
                    charging_events.append({
                        'station_id': best_cs['id'],
                        'soc_before': curr_soc / self.battery_cap * 100.0,
                        'soc_after': target_soc / self.battery_cap * 100.0,
                    })
                    curr_soc = target_soc
                    soc_traj.append(curr_soc / self.battery_cap * 100.0)
                    curr_lat, curr_lon = best_cs['lat'], best_cs['lon']

                    # Recalculate to order
                    d_next = haversine_dist(curr_lat, curr_lon, best_order['lat'], best_order['lon'])
                    e_next = energy_kw(d_next)

                # Move to order
                curr_soc -= e_next
                soc_traj.append(curr_soc / self.battery_cap * 100.0)
                route.append(f"Order_{best_order['id']}")
                curr_lat, curr_lon = best_order['lat'], best_order['lon']
                unvisited.remove(best_order)
                # Don't remove from remaining_orders here; multiple vehicles share orders
                # (for single-vehicle case remaining_orders will be fully consumed)

            results[v['id']] = {
                'route': route,
                'soc_trajectory': soc_traj,
                'charging_events': charging_events,
            }

            # Remove served orders from pool
            served_ids = {
                int(r.split('_')[1]) for r in route if r.startswith('Order_')
            }
            remaining_orders = [o for o in remaining_orders if o['id'] not in served_ids]

        return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    center = (21.0285, 105.8522)

    orders = [
        {'id': i, 'lat': center[0] + rng.uniform(-0.1, 0.1),
         'lon': center[1] + rng.uniform(-0.1, 0.1)}
        for i in range(1, 5)
    ]
    cs = [
        {'id': 1, 'lat': center[0] + 0.05, 'lon': center[1] + 0.05, 'max_power_kw': 150.0},
        {'id': 2, 'lat': center[0] - 0.05, 'lon': center[1] - 0.05, 'max_power_kw': 150.0},
    ]
    vehicles = [
        {'id': 'V1', 'depot_lat': center[0], 'depot_lon': center[1], 'initial_soc_pct': 90.0},
    ]

    solver = EVRPCSMILPSolver()
    result = solver.solve(orders, cs, vehicles)
    for vid, sol in result.items():
        print(f"Vehicle {vid}: route={sol['route']}, SoC={[f'{s:.1f}%' for s in sol['soc_trajectory']]}")
