import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class MultiCriteriaCostBuilder:
    """
    Builds cost matrix c_{kj} = w1*tau(l_k, pu_j) + w2*urgency(o_j) + w3*compat(d_k, o_j) + w4*fairness(d_k)

    urgency(o_j)   = (b_j - t_now) / (b_j - a_j)   [0=very urgent, 1=not urgent yet]
    compat(d_k, o_j) = 1 if d_k.capacity >= o_j.demand else penalty (default 10.0)
    fairness(d_k)  = max(0, avg_income - d_k.income) / avg_income

    Parameters
    ----------
    w1 : float  weight for travel-time component
    w2 : float  weight for urgency component
    w3 : float  weight for compatibility component
    w4 : float  weight for fairness component
    speed_kmh : float  assumed average speed for ETA computation
    """

    def __init__(self, w1=0.5, w2=0.2, w3=0.15, w4=0.15, speed_kmh=30.0):
        if abs(w1 + w2 + w3 + w4 - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {w1+w2+w3+w4:.4f}")
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.speed_ms = speed_kmh * 1000.0 / 3600.0  # convert to m/s
        self.incompatibility_penalty = 10.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, drivers, orders, t_now=0.0, driver_incomes=None):
        """
        Build the (n_drivers, n_orders) cost matrix.

        Parameters
        ----------
        drivers : list of objects with attributes:
                    .location  (x, y) in metres  OR  .lat/.lon in degrees
                    .capacity  numeric
                    .income    cumulative income so far  (used for fairness)
        orders  : list of objects with attributes:
                    .pickup_location  (x, y) in metres
                    .demand           numeric (e.g. number of seats)
                    .release_time     a_j  (earliest time order is active)
                    .deadline         b_j
        t_now   : float  current time (same unit as release_time / deadline)
        driver_incomes : optional 1-D array override for driver incomes

        Returns
        -------
        cost_matrix : np.ndarray, shape (n_drivers, n_orders)
        """
        n_d = len(drivers)
        n_o = len(orders)

        if n_d == 0 or n_o == 0:
            return np.zeros((n_d, n_o))

        # --- resolve incomes -------------------------------------------
        if driver_incomes is not None:
            incomes = np.asarray(driver_incomes, dtype=float)
        else:
            incomes = np.array([getattr(d, 'income', 0.0) for d in drivers],
                               dtype=float)
        avg_income = float(np.mean(incomes)) if len(incomes) > 0 else 1.0
        if avg_income == 0.0:
            avg_income = 1.0  # avoid division by zero

        cost_matrix = np.zeros((n_d, n_o))

        for k, drv in enumerate(drivers):
            drv_loc = self._get_location(drv, loc_attr='location')
            fairness_k = max(0.0, avg_income - incomes[k]) / avg_income  # [0,1]

            for j, ord_ in enumerate(orders):
                # --- travel-time component ---
                pu_loc = self._get_location(ord_, loc_attr='pickup_location')
                dist_m = self._euclidean(drv_loc, pu_loc)
                tau_kj = dist_m / self.speed_ms  # seconds

                # --- urgency component ---
                a_j = float(getattr(ord_, 'release_time', t_now))
                b_j = float(getattr(ord_, 'deadline', t_now + 1.0))
                window = b_j - a_j
                if window <= 0.0:
                    urgency_j = 0.0  # treat as maximally urgent
                else:
                    urgency_j = np.clip((b_j - t_now) / window, 0.0, 1.0)

                # --- compatibility component ---
                drv_cap = float(getattr(drv, 'capacity', 1.0))
                ord_dem = float(getattr(ord_, 'demand', 1.0))
                compat_kj = (1.0 if drv_cap >= ord_dem
                             else self.incompatibility_penalty)

                cost_matrix[k, j] = (self.w1 * tau_kj
                                     + self.w2 * urgency_j
                                     + self.w3 * compat_kj
                                     + self.w4 * fairness_k)

        return cost_matrix

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_location(obj, loc_attr='location'):
        """Return (x, y) from an object. Tries loc_attr first, then lat/lon."""
        loc = getattr(obj, loc_attr, None)
        if loc is not None:
            return float(loc[0]), float(loc[1])
        # fallback: try lat/lon and convert roughly to metres
        lat = getattr(obj, 'lat', 0.0)
        lon = getattr(obj, 'lon', 0.0)
        x = float(lon) * 111_320.0 * np.cos(np.radians(float(lat)))
        y = float(lat) * 110_540.0
        return x, y

    @staticmethod
    def _euclidean(p1, p2):
        return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


class HungarianMatcher:
    """
    Bo giai thuat toan Hungarian (Kuhn-Munkres) de tim matching toi uu hoa chi phi bipartite graph.
    Co ho tro pad dummy neu kich thuoc ma tran khong can bang (|Drivers| != |Orders|).
    """
    def solve(self, cost_matrix):
        """
        cost_matrix: Ma tran chi phi dang numpy array (N x M)
        Tra ve: row_ind, col_ind (danh sach index duoc ghep cap)
        """
        rows, cols = cost_matrix.shape
        
        # 1. Pad dummy neu ma tran khong vuong
        if rows != cols:
            max_dim = max(rows, cols)
            padded_matrix = np.full((max_dim, max_dim), 1e6) # chi phi cuc lon cho nut ao
            padded_matrix[:rows, :cols] = cost_matrix
        else:
            padded_matrix = cost_matrix
            
        if HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(padded_matrix)
        else:
            # Fallback sang Greedy Matching tinh chat vi scipy bi thieu
            row_ind, col_ind = self._greedy_fallback(padded_matrix)
            
        # Lọc bo cac ghep cap cua nut ao (dummy nodes)
        valid_rows = []
        valid_cols = []
        for r, c in zip(row_ind, col_ind):
            if r < rows and c < cols and cost_matrix[r, c] < 1e5:
                valid_rows.append(r)
                valid_cols.append(c)
                
        return np.array(valid_rows), np.array(valid_cols)

    def _greedy_fallback(self, cost_matrix):
        n = cost_matrix.shape[0]
        row_ind = list(range(n))
        col_ind = [-1] * n
        
        assigned_cols = set()
        for r in range(n):
            best_c = -1
            best_cost = float('inf')
            for c in range(n):
                if c not in assigned_cols:
                    if cost_matrix[r, c] < best_cost:
                        best_cost = cost_matrix[r, c]
                        best_c = c
            if best_c != -1:
                col_ind[r] = best_c
                assigned_cols.add(best_c)
        return row_ind, col_ind


class BertsekasAuctionMatcher:
    """
    Bo giai matching bang thuat toan Dau gia Bertsekas (Auction Algorithm 1988).
    Thich hop cho tinh toan song song va dat muc tieu thoi gian tre < 10ms voi hang nghin nut.
    """
    def __init__(self, eps=0.1, max_iters=1000):
        self.eps = eps
        self.max_iters = max_iters

    def solve(self, cost_matrix):
        """
        Bertsekas hoat dong tren bai toan TOI DA HOA GIA TRI (Maximize value).
        Do do, ta chuyen doi: value_matrix = C_max - cost_matrix.
        cost_matrix: (N_drivers x M_orders)
        """
        N, M = cost_matrix.shape
        if N == 0 or M == 0:
            return np.array([]), np.array([])
            
        # Vuong hoa ma tran neu bat can bang
        dim = max(N, M)
        val_matrix = np.zeros((dim, dim))
        c_max = np.max(cost_matrix) + 1.0
        
        # Chuyen tu min cost sang max value
        for i in range(N):
            for j in range(M):
                val_matrix[i, j] = c_max - cost_matrix[i, j]
                
        # Thiet lap cac bien thuat toan
        # Bidders (Nguoi dau gia - Drivers): 0..dim-1
        # Objects (Vat pham - Orders): 0..dim-1
        prices = np.zeros(dim)       # bang gia cua tung object
        person_to_obj = np.full(dim, -1) # person_to_obj[i] = j
        obj_to_person = np.full(dim, -1) # obj_to_person[j] = i
        
        unassigned = list(range(dim))
        iters = 0
        
        while unassigned and iters < self.max_iters:
            iters += 1
            i = unassigned.pop(0)
            
            # Buoc 1: Bidding phase
            # Tim vat pham j* mang lai gia tri ròng lon nhat: value[i, j] - price[j]
            net_values = val_matrix[i, :] - prices
            
            j_star = np.argmax(net_values)
            v_star = net_values[j_star]
            
            # Tim gia tri rong lon thu hai de quyet dinh gia tra thau
            net_values[j_star] = -1e9
            w_star = np.max(net_values)
            
            # Bid increment: delta = v_star - w_star + epsilon
            bid_increment = v_star - w_star + self.eps
            
            # Buoc 2: Assignment phase
            # Gan vat pham j* cho nguoi i
            prices[j_star] += bid_increment
            
            # Neu object j* da co nguoi so huu truoc do, tra nguoi do ve unassigned pool
            prev_owner = obj_to_person[j_star]
            if prev_owner != -1:
                person_to_obj[prev_owner] = -1
                unassigned.append(prev_owner)
                
            person_to_obj[i] = j_star
            obj_to_person[j_star] = i
            
        # Loc ghep cap hop le (loai bo dummy nodes)
        valid_rows = []
        valid_cols = []
        for i in range(N):
            j = person_to_obj[i]
            if j != -1 and j < M and cost_matrix[i, j] < 1e5:
                valid_rows.append(i)
                valid_cols.append(j)
                
        return np.array(valid_rows), np.array(valid_cols)
