"""
rtv_graph.py — Request-Trip-Vehicle (RTV) graph for high-capacity ride-pooling.

Implements the method from:
    Alonso-Mora et al., "On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment"
    Science 2017, Vol. 354, No. 6313.

Graph structure:
    R nodes : individual requests (trips)
    T nodes : feasible trips (subsets of R servable by one vehicle)
    V nodes : available vehicles

Edges:
    R–T : request r is a member of trip t
    T–V : vehicle v can serve trip t (based on location / time / SoC)

ILP assignment:
    Maximise number of served requests by assigning one trip per vehicle.
"""
import itertools
from typing import FrozenSet

import numpy as np

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

from darp_solver import DARPSolver, haversine_dist


class RTVGraph:
    """
    Request-Trip-Vehicle graph for ride-pooling (Alonso-Mora et al. Science 2017).

    Nodes:
    - R : individual requests (trips)
    - T : feasible trips (subsets of R that can be served by 1 vehicle without
          constraint violation)
    - V : available vehicles

    Edges:
    - R-T : request r ∈ trip t
    - T-V : vehicle v can serve trip t (based on current location + SoC + time)
    """

    def __init__(self, max_occupancy: int = 3,
                 max_wait_mins: float = 5.0,
                 max_detour_mins: float = 10.0,
                 speed_kmh: float = 30.0):
        self.max_occupancy = max_occupancy
        self.max_wait_mins = max_wait_mins
        self.max_detour_mins = max_detour_mins
        self.speed = speed_kmh
        self._darp = DARPSolver(speed_kmh=speed_kmh,
                                max_ride_time_mins=max_detour_mins + 5.0)

    # ------------------------------------------------------------------
    # Build feasible trips (T-nodes)
    # ------------------------------------------------------------------

    def build_feasible_trips(self, requests: list) -> list:
        """
        Generate all feasible trip combinations up to max_occupancy.

        A trip {r1, r2, …} is feasible if there exists at least one valid
        pickup/dropoff sequence satisfying:
        - All time windows
        - Max detour per passenger
        - Vehicle capacity ≤ max_occupancy

        Uses DARPSolver._try_insert_trip logic internally.

        Parameters
        ----------
        requests : list of trip dicts (id, pu_lat, pu_lon, do_lat, do_lon, class_tier)

        Returns
        -------
        list of frozenset — each frozenset contains integer indices into `requests`
        """
        n = len(requests)
        feasible = []

        # Single-request trips are always feasible (trivially)
        for i in range(n):
            feasible.append(frozenset([i]))

        # Multi-request trips up to max_occupancy
        for size in range(2, self.max_occupancy + 1):
            for combo in itertools.combinations(range(n), size):
                if self._is_combo_feasible(requests, combo):
                    feasible.append(frozenset(combo))

        return feasible

    def _is_combo_feasible(self, requests: list, combo: tuple) -> bool:
        """
        Check if a combination of request indices can form a feasible trip.

        Strategy: try all permutations of pickup order; for each, append
        dropoffs after all pickups and verify via DARPSolver feasibility check.
        Returns True as soon as one valid sequence is found.
        """
        # Dummy vehicle starting from the first request's pickup
        first = requests[combo[0]]
        dummy_vehicle = {
            'id': '__rtv_check__',
            'lat': first['pu_lat'],
            'lon': first['pu_lon'],
            'class_tier': 3,  # max tier to bypass tier checks
        }

        # Try each pickup permutation
        for pu_order in itertools.permutations(combo):
            # Build all interleavings of dropoffs respecting precedence
            for do_order in self._do_permutations(pu_order):
                route = (
                    [('PU', requests[i]) for i in pu_order] +
                    [('DO', requests[i]) for i in do_order]
                )
                feasible, _ = self._darp._check_route_feasibility(
                    dummy_vehicle, route)
                if feasible:
                    return True

        return False

    @staticmethod
    def _do_permutations(pu_order: tuple):
        """
        Generate all permutations of dropoffs that respect the requirement
        that each DO comes after its corresponding PU.
        Since PU is already ordered, all permutations of DO indices where
        DO_i appears at any position are valid candidates (precedence is
        enforced by _check_route_feasibility via pickup_times tracking).
        """
        return itertools.permutations(pu_order)

    # ------------------------------------------------------------------
    # Build vehicle–trip edges (V–T)
    # ------------------------------------------------------------------

    def build_vehicle_trip_edges(self, vehicles: list,
                                 trips: list) -> dict:
        """
        Determine which trips each vehicle can serve.

        A vehicle v can serve trip t if:
        1. v.class_tier ≥ max(class_tier for r in t)
        2. Travel time from v's current position to the first pickup ≤ max_wait_mins

        Parameters
        ----------
        vehicles : list of vehicle dicts (id, lat, lon, class_tier)
        trips    : list of frozensets from build_feasible_trips()

        Returns
        -------
        dict: {vehicle_idx: [trip_idx, ...]}
        """
        edges = {k: [] for k in range(len(vehicles))}

        for k, v in enumerate(vehicles):
            for t_idx, trip_set in enumerate(trips):
                if self._vehicle_can_serve(v, trip_set, []):
                    edges[k].append(t_idx)

        return edges

    def _vehicle_can_serve(self, vehicle: dict,
                           trip_set: FrozenSet,
                           requests: list) -> bool:
        """
        Check if vehicle can serve a trip set.
        trip_set contains integer indices when requests list is provided,
        or trip dicts directly when requests is empty.
        """
        if requests:
            trip_list = [requests[i] for i in trip_set]
        else:
            # trip_set already contains trip dicts (used in solve_assignment)
            trip_list = list(trip_set)

        if not trip_list:
            return False

        # Tier compatibility: vehicle tier must be >= max trip tier
        max_tier = max(t.get('class_tier', 1) for t in trip_list)
        if vehicle.get('class_tier', 1) < max_tier:
            return False

        # Wait time: can vehicle reach first pickup in time?
        first_trip = trip_list[0]
        dist_to_first = haversine_dist(
            vehicle['lat'], vehicle['lon'],
            first_trip['pu_lat'], first_trip['pu_lon'],
        )
        eta_mins = (dist_to_first / self.speed) * 60.0
        return eta_mins <= self.max_wait_mins

    # ------------------------------------------------------------------
    # ILP assignment
    # ------------------------------------------------------------------

    def solve_assignment(self, requests: list, vehicles: list) -> dict:
        """
        ILP assignment: maximise served requests.

        Formulation:
            min  Σ_r (1 - Σ_{t∋r} x_t)     [minimise unserved requests]
            s.t. Σ_t x_t ≤ 1  for each v   [each vehicle ≤ 1 trip]
                 x_t ∈ {0, 1}

        Uses OR-Tools CP-SAT if available; otherwise falls back to greedy.

        Parameters
        ----------
        requests : list of trip request dicts
        vehicles : list of vehicle dicts

        Returns
        -------
        dict:
            assignments      : {vehicle_id: [request_ids]}
            served_requests  : [request_ids that are served]
            unserved_requests: [request_ids that are NOT served]
        """
        feasible_trips = self.build_feasible_trips(requests)
        vt_edges = self.build_vehicle_trip_edges(vehicles, feasible_trips)

        if HAS_ORTOOLS:
            return self._solve_ortools(requests, vehicles,
                                       feasible_trips, vt_edges)
        return self._solve_greedy(requests, vehicles, feasible_trips, vt_edges)

    # ------------------------------------------------------------------
    # OR-Tools ILP
    # ------------------------------------------------------------------

    def _solve_ortools(self, requests: list, vehicles: list,
                       feasible_trips: list, vt_edges: dict) -> dict:
        """
        CP-SAT ILP:
        - Binary variable x_t for each feasible trip t
        - Each vehicle can be assigned ≤ 1 trip
        - Maximise total served requests
        """
        model = cp_model.CpModel()

        T = len(feasible_trips)
        K = len(vehicles)
        R = len(requests)

        # Binary variables: x[t] = 1 if trip t is selected
        x = [model.NewBoolVar(f'x_{t}') for t in range(T)]

        # Each vehicle assigned at most 1 trip
        for k in range(K):
            veh_trip_vars = [x[t] for t in vt_edges.get(k, [])]
            if veh_trip_vars:
                model.Add(sum(veh_trip_vars) <= 1)

        # Each request served at most once (across all selected trips)
        for r in range(R):
            req_trip_vars = [x[t] for t, trip_set in enumerate(feasible_trips)
                             if r in trip_set]
            if req_trip_vars:
                model.Add(sum(req_trip_vars) <= 1)

        # Objective: maximise total requests served
        obj_terms = []
        for t, trip_set in enumerate(feasible_trips):
            obj_terms.append(len(trip_set) * x[t])
        model.Maximize(sum(obj_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return self._solve_greedy(requests, vehicles,
                                      feasible_trips, vt_edges)

        # Decode solution
        selected_trips = [t for t in range(T) if solver.Value(x[t]) == 1]
        return self._decode_assignment(requests, vehicles,
                                       feasible_trips, vt_edges, selected_trips)

    # ------------------------------------------------------------------
    # Greedy fallback
    # ------------------------------------------------------------------

    def _solve_greedy(self, requests: list, vehicles: list,
                      feasible_trips: list, vt_edges: dict) -> dict:
        """
        Greedy: sort trips by size (largest first), assign to first
        available vehicle, mark requests as served.
        """
        # Sort trips by number of requests (largest first)
        trip_order = sorted(range(len(feasible_trips)),
                            key=lambda t: len(feasible_trips[t]),
                            reverse=True)

        used_vehicles = set()
        used_requests = set()
        selected_trips = []

        for t in trip_order:
            trip_set = feasible_trips[t]
            if trip_set & used_requests:
                continue  # some request already served

            # Find an available vehicle for this trip
            assigned_vehicle = None
            for k in range(len(vehicles)):
                if k in used_vehicles:
                    continue
                if t in vt_edges.get(k, []):
                    assigned_vehicle = k
                    break

            if assigned_vehicle is None:
                continue

            selected_trips.append(t)
            used_vehicles.add(assigned_vehicle)
            used_requests |= trip_set

        return self._decode_assignment(requests, vehicles,
                                       feasible_trips, vt_edges, selected_trips)

    # ------------------------------------------------------------------
    # Decode helper
    # ------------------------------------------------------------------

    def _decode_assignment(self, requests: list, vehicles: list,
                           feasible_trips: list, vt_edges: dict,
                           selected_trips: list) -> dict:
        """
        Convert list of selected trip indices into the output format.

        Returns
        -------
        dict:
            assignments      : {vehicle_id: [request dicts]}
            served_requests  : [request ids]
            unserved_requests: [request ids]
        """
        K = len(vehicles)
        assignments = {v['id']: [] for v in vehicles}
        served_request_indices = set()

        # For each selected trip, assign it to an eligible vehicle
        vehicle_used = [False] * K

        for t in selected_trips:
            trip_set = feasible_trips[t]
            trip_reqs = [requests[r] for r in sorted(trip_set)]

            # Find an eligible, unused vehicle
            for k in range(K):
                if vehicle_used[k]:
                    continue
                if t in vt_edges.get(k, []):
                    assignments[vehicles[k]['id']] = trip_reqs
                    vehicle_used[k] = True
                    served_request_indices |= trip_set
                    break

        served_ids = [requests[r]['id'] for r in sorted(served_request_indices)]
        unserved_ids = [requests[r]['id'] for r in range(len(requests))
                        if r not in served_request_indices]

        return {
            'assignments': assignments,
            'served_requests': served_ids,
            'unserved_requests': unserved_ids,
        }
