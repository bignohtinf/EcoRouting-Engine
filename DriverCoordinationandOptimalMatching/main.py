import time
import os
import random
import numpy as np

# Thiet lap Agg cho matplotlib de chay headless mượt mà
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import cac module noi bo
from matching_solvers import HungarianMatcher, BertsekasAuctionMatcher
from fairness import DriverFairnessManager
from repositioning import TimeExpandedRepositioner

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2))
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def run_matching_simulation():
    print("=" * 80)
    print(" HIEU NANG MO PHONG DRIVER COORDINATION & OPTIMAL MATCHING ".center(80, "="))
    print("=" * 80)
    
    random.seed(200)
    np.random.seed(200)
    
    num_drivers = 8
    num_orders = 6
    
    # 1. Sinh du lieu gia cho doi tai xe (Driver profiles)
    center_lat, center_lon = 21.0285, 105.8522
    drivers = []
    current_incomes = []
    
    for i in range(num_drivers):
        drv = {
            'id': i + 1,
            'lat': center_lat + random.uniform(-0.02, 0.02),
            'lon': center_lon + random.uniform(-0.02, 0.02),
            'seats': random.choice([1, 4, 5, 7]),  # VinFast Bike, VF5, VF e34, VF8
            'class': random.choice([1, 2, 3]),     # 1: Standard Bike, 2: Standard SUV, 3: Premium
            'income': random.uniform(20.0, 180.0) # Thu nhap hien tai trong ca
        }
        drivers.append(drv)
        current_incomes.append(drv['income'])
        
    # 2. Sinh du lieu gia cho cac don hang dang cho (Orders pool)
    orders = []
    for j in range(num_orders):
        order = {
            'id': j + 1,
            'pu_lat': center_lat + random.uniform(-0.02, 0.02),
            'pu_lon': center_lon + random.uniform(-0.02, 0.02),
            'req_seats': random.choice([1, 2, 4]),
            'req_class': random.choice([1, 2]),
            'urgency': random.uniform(0.1, 0.9)  # Do khan cap don hang
        }
        orders.append(order)
        
    print(f"Khoi tao thanh cong:")
    print(f" - So luong tai xe (Bidders): {num_drivers}")
    print(f" - So luong don hang (Objects): {num_orders}")
    
    # 3. Tinh ma tran chi phi ghep cap da tieu chi (Multi-criteria cost matrix)
    # cost[i, j] = dist + w_urgency * urgency - w_compat * compat + w_fairness * fairness
    cost_matrix = np.zeros((num_drivers, num_orders))
    fairness_manager = DriverFairnessManager()
    fairness_penalties = fairness_manager.get_fairness_penalties(current_incomes)
    
    for i in range(num_drivers):
        for j in range(num_orders):
            d = drivers[i]
            o = orders[j]
            
            # Tinh khoang cach
            dist = haversine_dist(d['lat'], d['lon'], o['pu_lat'], o['pu_lon'])
            
            # Tinh tinh tuong thich (compatibility)
            # Xe phai du ghe va dung phan cap dich vu
            compat = 1.0 if d['seats'] >= o['req_seats'] and d['class'] >= o['req_class'] else 0.0
            
            # Tinh chi phi ghep cap
            cost = dist * 2.0 + (1.0 - o['urgency']) * 5.0 + (1.0 - compat) * 50.0 + fairness_penalties[i] * 0.1
            cost_matrix[i, j] = cost
            
    # 4. Chay va so sanh bo giai Hungarian vs Bertsekas Auction
    print("\n--- [Buoc 1] So sanh bo giai Hungarian vs Bertsekas Auction ---")
    
    # --- Hungarian solver ---
    hungarian = HungarianMatcher()
    t_start = time.time()
    h_rows, h_cols = hungarian.solve(cost_matrix)
    t_h = (time.time() - t_start) * 1000.0
    
    # --- Auction solver ---
    auction = BertsekasAuctionMatcher(eps=0.05, max_iters=500)
    t_start = time.time()
    a_rows, a_cols = auction.solve(cost_matrix)
    t_a = (time.time() - t_start) * 1000.0
    
    # Hien thi bang so sanh
    print("-" * 80)
    print(f"{'Thuat toan':<20} | {'Thoi gian (ms)':<15} | {'So cap ghep':<15} | {'Tong Chi phi'}")
    print("-" * 80)
    h_cost = sum(cost_matrix[r, c] for r, c in zip(h_rows, h_cols))
    a_cost = sum(cost_matrix[r, c] for r, c in zip(a_rows, a_cols))
    print(f"{'Hungarian':<20} | {t_h:<15.4f} | {len(h_rows):<15} | {h_cost:.2f}")
    print(f"{'Auction Bertsekas':<20} | {t_a:<15.4f} | {len(a_rows):<15} | {a_cost:.2f}")
    print("-" * 80)
    
    # 5. Tinh toan Gini Coefficient do luong Fairness
    print("\n--- [Buoc 2] Thong ke Driver Fairness va Gini Coefficient ---")
    gini_before = fairness_manager.calculate_gini(current_incomes)
    
    # Gia su sau ca, cac tai xe duoc ghep don se nhan them 50.0 thu nhap
    incomes_after_matching = list(current_incomes)
    for r in a_rows:
        incomes_after_matching[r] += 50.0
        
    gini_after = fairness_manager.calculate_gini(incomes_after_matching)
    print(f" - He so Gini truoc ghep don (Inequality): {gini_before:.4f}")
    print(f" - He so Gini sau ghep don ho tro Fairness: {gini_after:.4f}")
    print(f"   (He so Gini giam cho thay su cong bang thu nhap toan doi duoc cai thien)")
    
    # 6. Chay thu nghiem Tai phan bo (Repositioning)
    print("\n--- [Buoc 3] Mo phong Time-Expanded Repositioning cho tai xe ranh ---")
    repositioner = TimeExpandedRepositioner(num_grids=3, num_periods=2)
    
    # Giả lập ma trận cung-cầu lưới
    supply = np.array([
        [5, 4], # Grid 0 tai thoi diem t0, t1
        [1, 2], # Grid 1
        [10, 8] # Grid 2 (Du cung)
    ])
    
    demand = np.array([
        [2, 3], # Grid 0
        [8, 6], # Grid 1 (Thieu cung tram trong)
        [3, 4]  # Grid 2
    ])
    
    cost_grid = np.array([
        [0.0, 2.0, 4.0],
        [2.0, 0.0, 3.0],
        [4.0, 3.0, 0.0]
    ])
    
    flow = repositioner.solve(supply, demand, cost_grid)
    
    print("-" * 80)
    print("LUONG XE DIEU PHOI TAI PHAN BO (REPOSITIONING FLOWS):")
    print("-" * 80)
    for t in range(2):
        print(f"Chu ky Thoi gian t = {t}:")
        for g in range(3):
            for gp in range(3):
                if flow[g, gp, t] > 0:
                    print(f" - Di chuyen {flow[g, gp, t]:.0f} tai xe tu Grid {g} -> Grid {gp} (Chi phi/xe: {cost_grid[g, gp]})")
    print("-" * 80)
    
    # 7. Ve bieu do ket qua matching
    plt.figure(figsize=(10, 8))
    for d in drivers:
        plt.scatter(d['lon'], d['lat'], color='blue', marker='o', s=120)
        plt.text(d['lon'], d['lat'], f"Drv_{d['id']}\n(Class {d['class']})", fontsize=8, ha='center', va='bottom', color='blue')
        
    for o in orders:
        plt.scatter(o['pu_lon'], o['pu_lat'], color='red', marker='^', s=120)
        plt.text(o['pu_lon'], o['pu_lat'], f"Ord_{o['id']}\n(Class {o['req_class']})", fontsize=8, ha='center', va='top', color='red')
        
    # Ve ket noi cap ghep
    for r, c in zip(a_rows, a_cols):
        drv = drivers[r]
        ord = orders[c]
        plt.plot([drv['lon'], ord['pu_lon']], [drv['lat'], ord['pu_lat']], color='green', linestyle='--', linewidth=2)
        
    plt.title("KET QUA GHEP CAP TOI UU DRIVER - ORDER (AUCTION SOLVER)", weight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("matching_results.png", dpi=300)
    plt.close()
    print(f"\n[Success] Da xuat do thi ket qua matching ra file: {os.path.abspath('matching_results.png')}")

if __name__ == "__main__":
    run_matching_simulation()
