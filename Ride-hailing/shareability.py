import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2))
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class ShareabilityNetwork:
    """Shareability Network (Santi et al. 2014, PNAS) for Ride-pooling."""

    def __init__(self, speed_kmh=30.0, max_detour_mins=10.0):
        self.speed = speed_kmh
        self.max_detour_mins = max_detour_mins

    def check_shareable(self, t1, t2):
        """
        Check if trips t1 and t2 can be legally shared.
        Options:
          A: PU_1 -> PU_2 -> DO_1 -> DO_2
          B: PU_1 -> PU_2 -> DO_2 -> DO_1
          C: PU_2 -> PU_1 -> DO_1 -> DO_2
          D: PU_2 -> PU_1 -> DO_2 -> DO_1
        Returns (shareable, option, saved_dist, detour1, detour2).
        """
        dist_t1 = haversine_dist(t1['pu_lat'], t1['pu_lon'], t1['do_lat'], t1['do_lon'])
        time_t1 = (dist_t1 / self.speed) * 60.0
        dist_t2 = haversine_dist(t2['pu_lat'], t2['pu_lon'], t2['do_lat'], t2['do_lon'])
        time_t2 = (dist_t2 / self.speed) * 60.0

        p1 = (t1['pu_lat'], t1['pu_lon'])
        d1 = (t1['do_lat'], t1['do_lon'])
        p2 = (t2['pu_lat'], t2['pu_lon'])
        d2 = (t2['do_lat'], t2['do_lon'])

        dist_p1_p2 = haversine_dist(p1[0], p1[1], p2[0], p2[1])
        dist_p2_d1 = haversine_dist(p2[0], p2[1], d1[0], d1[1])
        dist_d1_d2 = haversine_dist(d1[0], d1[1], d2[0], d2[1])
        dist_p2_d2 = haversine_dist(p2[0], p2[1], d2[0], d2[1])
        dist_d2_d1 = haversine_dist(d2[0], d2[1], d1[0], d1[1])
        dist_p1_d2 = haversine_dist(p1[0], p1[1], d2[0], d2[1])

        # Option A: p1 -> p2 -> d1 -> d2
        t1a = ((dist_p1_p2 + dist_p2_d1) / self.speed) * 60.0
        t2a = ((dist_p2_d1 + dist_d1_d2) / self.speed) * 60.0
        det1a, det2a = t1a - time_t1, t2a - time_t2
        if det1a <= self.max_detour_mins and det2a <= self.max_detour_mins:
            saved = (dist_t1 + dist_t2) - (dist_p1_p2 + dist_p2_d1 + dist_d1_d2)
            return True, 'A', saved, det1a, det2a

        # Option B: p1 -> p2 -> d2 -> d1
        t1b = ((dist_p1_p2 + dist_p2_d2 + dist_d2_d1) / self.speed) * 60.0
        t2b = (dist_p2_d2 / self.speed) * 60.0
        det1b, det2b = t1b - time_t1, t2b - time_t2
        if det1b <= self.max_detour_mins and det2b <= self.max_detour_mins:
            saved = (dist_t1 + dist_t2) - (dist_p1_p2 + dist_p2_d2 + dist_d2_d1)
            return True, 'B', saved, det1b, det2b

        # Option C: p2 -> p1 -> d1 -> d2
        dist_p2_p1 = dist_p1_p2  # symmetric
        t1c = ((dist_p2_p1 + dist_t1) / self.speed) * 60.0
        t2c = ((dist_p2_p1 + dist_t1 + dist_d1_d2) / self.speed) * 60.0
        det1c, det2c = t1c - time_t1, t2c - time_t2
        if det1c <= self.max_detour_mins and det2c <= self.max_detour_mins:
            saved = (dist_t1 + dist_t2) - (dist_p2_p1 + dist_t1 + dist_d1_d2)
            return True, 'C', saved, det1c, det2c

        # Option D: p2 -> p1 -> d2 -> d1
        t1d = ((dist_p2_p1 + dist_p1_d2 + dist_d2_d1) / self.speed) * 60.0
        t2d = ((dist_p2_p1 + dist_p1_d2) / self.speed) * 60.0
        det1d, det2d = t1d - time_t1, t2d - time_t2
        if det1d <= self.max_detour_mins and det2d <= self.max_detour_mins:
            saved = (dist_t1 + dist_t2) - (dist_p2_p1 + dist_p1_d2 + dist_d2_d1)
            return True, 'D', saved, det1d, det2d

        return False, None, 0.0, 0.0, 0.0

    def build_graph(self, trips):
        """Build shareability graph. Returns list of edge dicts."""
        n = len(trips)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                shareable, opt, saved, det1, det2 = self.check_shareable(trips[i], trips[j])
                if shareable and saved > 0.0:
                    edges.append({
                        'node1': i,
                        'node2': j,
                        'option': opt,
                        'weight': saved,
                        'detour1': det1,
                        'detour2': det2
                    })
        return edges


class PassengerChoiceModel:
    """
    Passenger choice model (Section 6.3):
    P(accept | detour, discount) = 1 / (1 + exp(-(b0 - b1*detour + b2*discount)))
    """
    def __init__(self, beta0=1.5, beta1=0.25, beta2=4.0):
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2

    def calculate_probability(self, detour_mins, discount_pct):
        utility = self.beta0 - self.beta1 * detour_mins + self.beta2 * discount_pct
        return 1.0 / (1.0 + np.exp(-utility))


class MaxWeightMatching:
    """
    Solves maximum weight bipartite matching on the Shareability Graph.
    Input: edges from ShareabilityNetwork.build_graph()
    Output: set of non-overlapping pairs that maximize total weight (saved distance)
    Uses greedy matching (sorted by weight) with scipy.optimize.linear_sum_assignment fallback.
    """

    def solve(self, trips, edges) -> list:
        """
        Returns list of matched pairs:
        [{'trip1_idx': int, 'trip2_idx': int, 'weight': float, 'option': str}]

        Strategy:
        1. If scipy is available: Hungarian algorithm via linear_sum_assignment.
        2. Otherwise: greedy sort-by-weight with conflict avoidance.
        """
        if not edges:
            return []
        n = len(trips)
        if HAS_SCIPY and n >= 2:
            return self._solve_scipy(n, edges)
        return self._solve_greedy(edges)

    def _solve_scipy(self, n, edges) -> list:
        """Hungarian algorithm via scipy.optimize.linear_sum_assignment."""
        best_edge = {}
        for e in edges:
            key = (e['node1'], e['node2'])
            if key not in best_edge or e['weight'] > best_edge[key]['weight']:
                best_edge[key] = e

        cost = np.zeros((n, n))
        for (i, j), e in best_edge.items():
            cost[i][j] = -e['weight']
            cost[j][i] = -e['weight']

        row_ind, col_ind = linear_sum_assignment(cost)
        matched = []
        used = set()
        for r, c in zip(row_ind, col_ind):
            if r == c:
                continue
            pair = (min(r, c), max(r, c))
            if pair in used or pair not in best_edge:
                continue
            e = best_edge[pair]
            if e['weight'] <= 0:
                continue
            matched.append({
                'trip1_idx': e['node1'],
                'trip2_idx': e['node2'],
                'weight': e['weight'],
                'option': e['option'],
            })
            used.add(pair)
        return matched

    def _solve_greedy(self, edges) -> list:
        """Greedy: sort by weight descending, pick non-conflicting pairs."""
        sorted_edges = sorted(edges, key=lambda e: e['weight'], reverse=True)
        used_trips = set()
        matched = []
        for e in sorted_edges:
            i, j = e['node1'], e['node2']
            if i in used_trips or j in used_trips:
                continue
            matched.append({
                'trip1_idx': i,
                'trip2_idx': j,
                'weight': e['weight'],
                'option': e['option'],
            })
            used_trips.add(i)
            used_trips.add(j)
        return matched
