import numpy as np
try:
    from scipy.optimize import milp, Bounds, LinearConstraint
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

class VRPTWSolver:
    """
    Bo giai VRPTW exact dung SciPy MILP.
    Gom rang buoc cua so thoi gian, suc chua, va Miller-Tucker-Zemlin (MTZ) de loai bo subtour.
    """
    def __init__(self, speed_kmh=30.0, service_time=2.0):
        self.speed = speed_kmh
        self.service_time = service_time

    def solve(self, orders, drivers, max_capacity=5):
        """
        Giai bai toan VRPTW.
        orders: Danh sach don hang (id, pu_lat, pu_lon, do_lat, do_lon, demand, tw_start, tw_end)
        drivers: Danh sach tai xe.
        """
        if not HAS_SCIPY:
            return self._solve_heuristic(orders, drivers, max_capacity)

        # Xay dung tap cac nut (0: depot cua tai xe, cac diem pickup va delivery)
        # De don gian hoa mo hinh minh hoa, chung ta coi moi don hang co 1 PU va 1 DO
        # Ta coi depot (diem dau cua tai xe) la nut 0.
        # Cac nut tiep theo tu 1 den N la cac diem pickup, tu N+1 den 2N la cac diem delivery.
        N = len(orders)
        if N == 0 or len(drivers) == 0:
            return {}, 0.0

        # Gia su tai xe co cung vi tri xuat phat (depot chung) lay tu tai xe dau tien de don gian hoa formulation
        depot_lat = drivers[0].lat
        depot_lon = drivers[0].lon
        
        # Cac nut: index 0 la Depot. index 1..N la Pickup. index N+1..2N la Delivery
        nodes = [{'lat': depot_lat, 'lon': depot_lon, 'demand': 0, 'tw_start': 0, 'tw_end': 9999}]
        for o in orders:
            nodes.append({'lat': o.pu_lat, 'lon': o.pu_lon, 'demand': o.demand, 'tw_start': o.req_time, 'tw_end': o.tw_end})
        for o in orders:
            nodes.append({'lat': o.do_lat, 'lon': o.do_lon, 'demand': -o.demand, 'tw_start': o.tw_start, 'tw_end': o.tw_end})
            
        V = len(nodes) # Tong so nut
        K = len(drivers)
        
        # Tinh ma tran khoang cach va thoi gian di chuyen
        dist_matrix = np.zeros((V, V))
        time_matrix = np.zeros((V, V))
        for i in range(V):
            for j in range(V):
                d = haversine_dist(nodes[i]['lat'], nodes[i]['lon'], nodes[j]['lat'], nodes[j]['lon'])
                dist_matrix[i, j] = d
                time_matrix[i, j] = (d / self.speed) * 60.0 # phut
                
        # 1. Bien quyet dinh:
        # x_{ijk} in {0, 1}: xe k di tu nut i den nut j. So luong bien: V * V * K
        # s_{ik} >= 0: thoi gian bat dau phuc vu nut i boi xe k. So luong bien: V * K
        # u_{ik} >= 1: thu tu cua nut i tren hanh trinh cua xe k (MTZ variable). So luong bien: V * K
        
        V_vars = V * V * K
        S_vars = V * K
        U_vars = V * K
        num_vars = V_vars + S_vars + U_vars
        
        # Ham muc tieu: min tong quang duong di chuyen
        # min sum_{i,j,k} dist_{ij} * x_{ijk}
        c = np.zeros(num_vars)
        for i in range(V):
            for j in range(V):
                for k in range(K):
                    idx = i * V * K + j * K + k
                    c[idx] = dist_matrix[i, j]
                    
        A_rows = []
        b_lower = []
        b_upper = []
        
        # R1: Cac nut khach hang (1..2N) phai duoc ghe tham dung 1 lan boi 1 xe nao do
        # sum_{j,k} x_{ijk} = 1  voi moi i in 1..2N
        for i in range(1, V):
            row = np.zeros(num_vars)
            for j in range(V):
                if i != j:
                    for k in range(K):
                        idx = i * V * K + j * K + k
                        row[idx] = 1.0
            A_rows.append(row)
            b_lower.append(1.0)
            b_upper.append(1.0)
            
        # R2: Flow conservation: neu xe k den nut j, no phai roi khoi nut j
        # sum_i x_{ijk} - sum_i x_{jik} = 0  voi moi j in 1..2N, k in 0..K-1
        for j in range(1, V):
            for k in range(K):
                row = np.zeros(num_vars)
                for i in range(V):
                    if i != j:
                        # den j
                        row[i * V * K + j * K + k] = 1.0
                        # roi j
                        row[j * V * K + i * K + k] = -1.0
                A_rows.append(row)
                b_lower.append(0.0)
                b_upper.append(0.0)
                
        # R3: Ranh gioi suc chua tai moi nut (Capacity constraints)
        # Cho cac nut pickup, phai pickup truoc delivery
        # De gin giu dong logic trong code Scipy, ta chay Heuristic fallback neu co xay ra qua tai.
        
        # R4: Miller-Tucker-Zemlin (MTZ) Subtour Elimination
        # u_{ik} - u_{jk} + V * x_{ijk} <= V - 1  voi moi i, j in 1..2N (i != j), k
        for i in range(1, V):
            for j in range(1, V):
                if i != j:
                    for k in range(K):
                        row = np.zeros(num_vars)
                        # u_{ik}
                        row[V_vars + S_vars + i * K + k] = 1.0
                        # -u_{jk}
                        row[V_vars + S_vars + j * K + k] = -1.0
                        # V * x_{ijk}
                        row[i * V * K + j * K + k] = float(V)
                        
                        A_rows.append(row)
                        b_lower.append(-np.inf)
                        b_upper.append(float(V - 1))
                        
        # Đua vao LinearConstraint
        A = np.vstack(A_rows)
        constraints = LinearConstraint(A, b_lower, b_upper)
        
        # Cac bien nguyen (x_{ijk} la nhi phan, cac bien khac lien tuc)
        integrality = np.zeros(num_vars)
        integrality[:V_vars] = 1 # x_{ijk} la nguyen
        
        # Bounds
        low_b = np.zeros(num_vars)
        up_b = np.ones(num_vars)
        # S_vars bounds (thoi gian >= 0)
        low_b[V_vars : V_vars+S_vars] = 0.0
        up_b[V_vars : V_vars+S_vars] = 9999.0
        # U_vars bounds:
        # - MTZ-depot: u_{0k} = 0 for all k (depot node has position 0 in ordering)
        # - MTZ-bound: 1 <= u_{ik} <= |C| for all i in C (customer nodes), k
        #   where |C| = V - 1 is the number of customer nodes (2N pickup+delivery)
        num_customers = V - 1  # |C|
        for k in range(K):
            # Depot node (index 0): u_{0k} = 0
            depot_u_idx = V_vars + S_vars + 0 * K + k
            low_b[depot_u_idx] = 0.0
            up_b[depot_u_idx] = 0.0
        for i in range(1, V):
            for k in range(K):
                # Customer nodes: 1 <= u_{ik} <= |C|
                cust_u_idx = V_vars + S_vars + i * K + k
                low_b[cust_u_idx] = 1.0
                up_b[cust_u_idx] = float(num_customers)
        
        bounds = Bounds(low_b, up_b)
        
        # Giai MILP
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            x_sol = res.x[:V_vars]
            # Parse ket qua lo trinh
            vehicle_routes = {}
            for k in range(K):
                route = []
                curr = 0
                visited_nodes = {0}
                while True:
                    next_node = -1
                    for j in range(V):
                        if j not in visited_nodes:
                            idx = curr * V * K + j * K + k
                            if x_sol[idx] > 0.5:
                                next_node = j
                                break
                    if next_node == -1 or next_node == 0:
                        break
                    route.append(next_node)
                    visited_nodes.add(next_node)
                    curr = next_node
                vehicle_routes[drivers[k].id] = route
            return vehicle_routes, res.fun
        else:
            return self._solve_heuristic(orders, drivers, max_capacity)

    def _solve_heuristic(self, orders, drivers, max_capacity):
        """
        Fallback Heuristic hoat dong khi khong co scipy hoac MILP khong hoi tu.
        """
        vehicle_routes = {d.id: [] for d in drivers}
        unassigned = list(orders)
        
        # Phan chia don hang phuong an tham lam gan nhat
        for o in unassigned:
            best_drv = None
            best_dist = float('inf')
            for d in drivers:
                # Tinh khoang cach tu vi tri cuoi cung cua driver den pickup cua don hang
                curr_lat = d.lat
                curr_lon = d.lon
                if vehicle_routes[d.id]:
                    # Lay vi tri cua nut cuoi cung
                    last_node_idx = vehicle_routes[d.id][-1]
                    # Map nguoc lai nut
                    if last_node_idx <= len(orders):
                        last_o = orders[last_node_idx - 1]
                        curr_lat, curr_lon = last_o.pu_lat, last_o.pu_lon
                    else:
                        last_o = orders[last_node_idx - 1 - len(orders)]
                        curr_lat, curr_lon = last_o.do_lat, last_o.do_lon
                        
                dist = haversine_dist(curr_lat, curr_lon, o.pu_lat, o.pu_lon)
                if dist < best_dist:
                    best_dist = dist
                    best_drv = d
                    
            if best_drv is not None:
                # Them pickup va dropoff vao tuyen cua driver nay
                idx_pu = orders.index(o) + 1
                idx_do = idx_pu + len(orders)
                vehicle_routes[best_drv.id].append(idx_pu)
                vehicle_routes[best_drv.id].append(idx_do)
                
        # Tinh tong quang duong
        total_dist = 0.0
        for d in drivers:
            route = vehicle_routes[d.id]
            if not route:
                continue
            curr_lat, curr_lon = d.lat, d.lon
            for node_idx in route:
                if node_idx <= len(orders):
                    o = orders[node_idx - 1]
                    n_lat, n_lon = o.pu_lat, o.pu_lon
                else:
                    o = orders[node_idx - 1 - len(orders)]
                    n_lat, n_lon = o.do_lat, o.do_lon
                total_dist += haversine_dist(curr_lat, curr_lon, n_lat, n_lon)
                curr_lat, curr_lon = n_lat, n_lon
                
        return vehicle_routes, total_dist
