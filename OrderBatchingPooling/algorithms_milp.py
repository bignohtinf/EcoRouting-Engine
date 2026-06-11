import numpy as np

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

from algorithms_heuristic import HeuristicBatchMatcher, GreedyInsertionRouter
from data_generator import haversine_distance


class MILPBatchMatcher:
    """
    Bo toi uu hoa ghep cap don hang bang mo hinh Tuyen tinh nguyen hon hop (MILP)
    su dung OR-Tools CP-SAT. Mo hinh hoa day du McCormick Envelope Linearization
    cho bien tich w_ijk = y_ik * y_jk, cung rang buoc C1-C6 theo cong thuc.

    Bien quyet dinh:
      y_ik in {0,1}   : batch i duoc gan cho tai xe k
      z_k  in {0,1}   : tai xe k duoc kich hoat
      w_ijk in {0,1}  : ca batch i va batch j (i<j) deu duoc gan cho tai xe k
                        (bien tuyen tinh hoa McCormick cho tich y_ik * y_jk)

    Muc tieu:
      min  sum_{i,k} cost_ik * y_ik  +  mu * sum_k z_k

    Rang buoc:
      C1: sum_k y_ik <= 1                      forall i
      C2: sum_i y_ik <= z_k * B_max            forall k
      C4: sum_i y_ik <= B_max                  forall k
      C5: w_ijk <= y_ik, w_ijk <= y_jk         forall i<j, k  (McCormick upper)
      C6: w_ijk >= y_ik + y_jk - 1             forall i<j, k  (McCormick lower)
    """

    def __init__(self, lambda_delay=15.0, mu_activation=5.0, b_max=4):
        self.lambda_delay = lambda_delay
        self.mu_activation = mu_activation
        self.b_max = b_max
        self.router = GreedyInsertionRouter()

    def solve(self, clusters, drivers):
        """
        Giai bai toan gan batch cho tai xe.

        Tham so:
          clusters : list of batches, moi batch la list[Order]
          drivers  : list[Driver]

        Tra ve:
          assignments     : list of dict {driver, batch, route, distance, delay, cost}
          unassigned_orders: list[Order] cac don hang chua duoc gan
        """
        if not HAS_ORTOOLS:
            print(
                "\n[Notice] OR-Tools chua duoc cai dat. "
                "Chuyen sang thuat toan Heuristic thay the."
            )
            fallback = HeuristicBatchMatcher(self.lambda_delay, self.mu_activation)
            return fallback.solve(clusters, drivers)

        num_batches = len(clusters)
        num_drivers = len(drivers)

        if num_batches == 0 or num_drivers == 0:
            return [], []

        # ------------------------------------------------------------------
        # Buoc 1: Tinh ma tran chi phi va kiem tra tinh kha thi (i, k)
        # ------------------------------------------------------------------
        cost_matrix = np.zeros((num_batches, num_drivers))
        feasibility_matrix = np.zeros((num_batches, num_drivers), dtype=bool)
        routes = {}
        distances = {}
        delays = {}

        for bi in range(num_batches):
            for dk in range(num_drivers):
                batch = clusters[bi]
                driver = drivers[dk]

                route, dist, delay, feasible = self.router.solve_route(
                    batch_orders=batch,
                    start_lat=driver.lat,
                    start_lon=driver.lon,
                    start_time=0.0,
                    max_capacity=driver.capacity,
                )

                if feasible:
                    cost_matrix[bi, dk] = dist + self.lambda_delay * delay
                    feasibility_matrix[bi, dk] = True
                    routes[(bi, dk)] = route
                    distances[(bi, dk)] = dist
                    delays[(bi, dk)] = delay
                else:
                    # Chi phi rat lon cho phuong an khong kha thi
                    cost_matrix[bi, dk] = 1e6
                    feasibility_matrix[bi, dk] = False

        # ------------------------------------------------------------------
        # Buoc 2: Xay dung mo hinh CP-SAT
        # ------------------------------------------------------------------
        model = cp_model.CpModel()

        # OR-Tools CP-SAT yeu cau he so nguyen trong ham muc tieu.
        # Nhan chi phi voi 1000 va lam tron de giu 3 chu so thap phan.
        SCALE = 1000
        cost_int = np.round(cost_matrix * SCALE).astype(int)
        mu_int = int(round(self.mu_activation * SCALE))

        # --- Bien y_ik in {0,1} ---
        y = {}
        for bi in range(num_batches):
            for dk in range(num_drivers):
                y[bi, dk] = model.new_bool_var("y_%d_%d" % (bi, dk))

        # --- Bien z_k in {0,1} ---
        z = {}
        for dk in range(num_drivers):
            z[dk] = model.new_bool_var("z_%d" % dk)

        # --- Bien w_ijk in {0,1} cho bi < bj, forall dk (McCormick) ---
        w = {}
        for bi in range(num_batches):
            for bj in range(bi + 1, num_batches):
                for dk in range(num_drivers):
                    w[bi, bj, dk] = model.new_bool_var("w_%d_%d_%d" % (bi, bj, dk))

        # --- Bien s_i in {0,1}: batch i KHONG duoc gan (slack) ---
        s = {}
        for bi in range(num_batches):
            s[bi] = model.new_bool_var("s_%d" % bi)

        # ------------------------------------------------------------------
        # Buoc 3: Rang buoc
        # ------------------------------------------------------------------

        # C1: Moi batch bi duoc gan toi da 1 tai xe
        #     sum_k y_ik <= 1
        for bi in range(num_batches):
            model.add(sum(y[bi, dk] for dk in range(num_drivers)) <= 1)

        # C2: Rang buoc kich hoat tai xe (lien ket z_k voi y_ik)
        #     sum_i y_ik <= z_k * B_max
        for dk in range(num_drivers):
            model.add(
                sum(y[bi, dk] for bi in range(num_batches)) <= self.b_max * z[dk]
            )

        # C4: Toi da B_max batch moi tai xe
        #     sum_i y_ik <= B_max
        for dk in range(num_drivers):
            model.add(sum(y[bi, dk] for bi in range(num_batches)) <= self.b_max)

        # C5 & C6: McCormick Envelope Linearization
        #   w_ijk = y_ik AND y_jk  (tuyen tinh hoa tich nhi phan)
        #
        #   C5 (upper bounds):
        #     w_ijk <= y_ik
        #     w_ijk <= y_jk
        #
        #   C6 (lower bound):
        #     w_ijk >= y_ik + y_jk - 1
        for bi in range(num_batches):
            for bj in range(bi + 1, num_batches):
                for dk in range(num_drivers):
                    # C5: McCormick upper
                    model.add(w[bi, bj, dk] <= y[bi, dk])
                    model.add(w[bi, bj, dk] <= y[bj, dk])
                    # C6: McCormick lower
                    model.add(w[bi, bj, dk] >= y[bi, dk] + y[bj, dk] - 1)

        # Loai bo cac phuong an khong kha thi
        for bi in range(num_batches):
            for dk in range(num_drivers):
                if not feasibility_matrix[bi, dk]:
                    model.add(y[bi, dk] == 0)

        # Rang buoc slack: s_i + sum_k y_ik = 1
        # (s_i = 1 neu batch bi khong duoc gan cho ai)
        for bi in range(num_batches):
            model.add(s[bi] + sum(y[bi, dk] for dk in range(num_drivers)) == 1)

        # ------------------------------------------------------------------
        # Buoc 4: Ham muc tieu
        #   min  sum_{i,k} cost_ik * y_ik
        #      + mu * sum_k z_k
        #      + M * sum_i s_i        (phat lon neu batch khong duoc gan)
        #
        # M du lon de solver uu tien phu tat ca batch neu co the.
        # ------------------------------------------------------------------
        max_fc = int(np.max(cost_int[feasibility_matrix]) + 1) if feasibility_matrix.any() else 1
        M_penalty = max(max_fc * (num_batches + num_drivers + 1), 10_000_000)

        obj_terms = []
        for bi in range(num_batches):
            for dk in range(num_drivers):
                obj_terms.append(int(cost_int[bi, dk]) * y[bi, dk])
        for dk in range(num_drivers):
            obj_terms.append(mu_int * z[dk])
        for bi in range(num_batches):
            obj_terms.append(M_penalty * s[bi])

        model.minimize(sum(obj_terms))

        # ------------------------------------------------------------------
        # Buoc 5: Giai
        # ------------------------------------------------------------------
        solver = cp_model.CpSolver()
        # Gioi han thoi gian giai de dam bao phan hoi real-time
        solver.parameters.max_time_in_seconds = 30.0
        status = solver.solve(model)

        assignments = []
        unassigned_batches = []

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            for bi in range(num_batches):
                assigned = False
                for dk in range(num_drivers):
                    if solver.value(y[bi, dk]) == 1 and feasibility_matrix[bi, dk]:
                        driver = drivers[dk]
                        batch = clusters[bi]
                        assignments.append({
                            "driver": driver,
                            "batch": batch,
                            "route": routes[(bi, dk)],
                            "distance": distances[(bi, dk)],
                            "delay": delays[(bi, dk)],
                            "cost": float(cost_matrix[bi, dk]) + self.mu_activation,
                        })
                        assigned = True
                        break
                if not assigned:
                    unassigned_batches.append(clusters[bi])
        else:
            # Solver that bai hoac infeasible - fallback sang Heuristic
            status_name = solver.status_name(status)
            print(
                "[Warning] CP-SAT solver returned status '%s'. "
                "Falling back to greedy heuristic." % status_name
            )
            fallback = HeuristicBatchMatcher(self.lambda_delay, self.mu_activation)
            return fallback.solve(clusters, drivers)

        unassigned_orders = []
        for batch in unassigned_batches:
            unassigned_orders.extend(batch)

        return assignments, unassigned_orders
