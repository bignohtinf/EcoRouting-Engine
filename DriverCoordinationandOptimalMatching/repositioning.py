import numpy as np
try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class TimeExpandedRepositioner:
    """
    Bo giai bai toan Tai phan bo tai xe (Repositioning) dua tren Time-Expanded Transportation Problem LP.
    Dieu huong tai xe idle tu o luoi du thua sang o luoi khan hiem tai xe.
    """
    def __init__(self, num_grids=3, num_periods=2):
        self.G = num_grids
        self.T = num_periods

    def solve(self, supply_grid, demand_grid, cost_grid):
        """
        supply_grid: Ma tran nguon cung hien tai cua tai xe tai o g, thoi diem t (shape: G x T)
        demand_grid: Ma tran nhu cau don hang tai o g, thoi diem t (shape: G x T)
        cost_grid: Ma tran chi phi di chuyen giua o g va g' (shape: G x G)
        
        Tra ve: flow[g, g', t] la luong tai xe dieu phoi tu g -> g' o thoi diem t.
        """
        if not HAS_SCIPY:
            return self._solve_greedy(supply_grid, demand_grid, cost_grid)

        # Bien quyet dinh f_{g, g', t}: So luong xe dieu chuyen tu g -> g' tai thoi diem t
        # Tong so bien = G * G * T
        num_vars = self.G * self.G * self.T
        
        # Ham muc tieu: min sum_{g, g', t} cost_{g, g'} * f_{g, g', t}
        c = np.zeros(num_vars)
        for g in range(self.G):
            for gp in range(self.G):
                for t in range(self.T):
                    idx = g * self.G * self.T + gp * self.T + t
                    c[idx] = cost_grid[g, gp]
                    
        A_ub = []
        b_ub = []
        
        # Rang buoc 1: Tong luong tai xe roi khoi g tai thoi diem t khong vuot qua nguon cung tai do
        # sum_{gp} f_{g, gp, t} <= supply[g, t]   voi moi g, t
        for g in range(self.G):
            for t in range(self.T):
                row = np.zeros(num_vars)
                for gp in range(self.G):
                    idx = g * self.G * self.T + gp * self.T + t
                    row[idx] = 1.0
                A_ub.append(row)
                b_ub.append(supply_grid[g, t])
                
        # Rang buoc 2: Nhu cau thieu hut duoc dap ung toi da tai o gp, thoi diem t
        # sum_{g} f_{g, gp, t} >= max(0, demand[gp, t] - supply[gp, t])
        # Bien doi: -sum_{g} f_{g, gp, t} <= -deficit[gp, t]
        for gp in range(self.G):
            for t in range(self.T):
                deficit = max(0.0, demand_grid[gp, t] - supply_grid[gp, t])
                if deficit > 0:
                    row = np.zeros(num_vars)
                    for g in range(self.G):
                        idx = g * self.G * self.T + gp * self.T + t
                        row[idx] = -1.0
                    A_ub.append(row)
                    b_ub.append(-deficit)
                    
        if len(A_ub) == 0:
            return np.zeros((self.G, self.G, self.T))
            
        A_ub = np.vstack(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: f_{g, gp, t} >= 0
        bounds = [(0.0, None) for _ in range(num_vars)]
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        flow = np.zeros((self.G, self.G, self.T))
        if res.success:
            x = res.x
            for g in range(self.G):
                for gp in range(self.G):
                    for t in range(self.T):
                        idx = g * self.G * self.T + gp * self.T + t
                        flow[g, gp, t] = round(x[idx])
        else:
            return self._solve_greedy(supply_grid, demand_grid, cost_grid)
            
        return flow

    def _solve_greedy(self, supply_grid, demand_grid, cost_grid):
        """
        Fallback tham lam dieu phoi tu khu vuc du sang khu vuc thieu.
        """
        flow = np.zeros((self.G, self.G, self.T))
        for t in range(self.T):
            surplus = []
            deficits = []
            
            for g in range(self.G):
                net = supply_grid[g, t] - demand_grid[g, t]
                if net > 0:
                    surplus.append([g, net])
                elif net < 0:
                    deficits.append([g, -net])
                    
            # Dieu phoi tu thieu sang du
            for s_item in surplus:
                sg = s_item[0]
                s_val = s_item[1]
                for d_item in deficits:
                    dg = d_item[0]
                    d_val = d_item[1]
                    
                    if s_val <= 0 or d_val <= 0:
                        continue
                        
                    move_qty = min(s_val, d_val)
                    flow[sg, dg, t] = move_qty
                    s_val -= move_qty
                    d_item[1] -= move_qty
                    
        return flow
