import time
import os
import random
import numpy as np

# Thiet lap Agg cho matplotlib de chay headless mượt mà
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import cac module noi bo
from vrptw_solver import VRPTWSolver, haversine_dist
from dvrp_rl import DVRPEnv, PPOAgent
from eta_predictor import STGNNETAPredictor


class MockOrder:
    def __init__(self, order_id, pu_lat, pu_lon, do_lat, do_lon, demand, req_time, tw_start, tw_end):
        self.id = order_id
        self.pu_lat = pu_lat
        self.pu_lon = pu_lon
        self.do_lat = do_lat
        self.do_lon = do_lon
        self.demand = demand
        self.req_time = req_time
        self.tw_start = tw_start
        self.tw_end = tw_end

class MockDriver:
    def __init__(self, driver_id, lat, lon, capacity=5):
        self.id = driver_id
        self.lat = lat
        self.lon = lon
        self.capacity = capacity

def generate_mock_vrp_data():
    """
    Sinh du lieu gia lap tai khu vuc Ha Noi.
    Center: Ho Hoan Kiem (21.0285, 105.8522)
    """
    random.seed(100)
    center_lat, center_lon = 21.0285, 105.8522
    
    orders = []
    # Sinh 8 don hang ban dau
    for i in range(1, 9):
        pu_lat = center_lat + random.uniform(-0.03, 0.03)
        pu_lon = center_lon + random.uniform(-0.03, 0.03)
        do_lat = pu_lat + random.uniform(-0.02, 0.02)
        do_lon = pu_lon + random.uniform(-0.02, 0.02)
        
        demand = random.choice([1, 2])
        req_time = random.uniform(0.0, 10.0)
        tw_start = req_time + random.uniform(5.0, 15.0)
        tw_end = tw_start + random.uniform(20.0, 40.0)
        
        orders.append(MockOrder(i, pu_lat, pu_lon, do_lat, do_lon, demand, req_time, tw_start, tw_end))
        
    drivers = [
        MockDriver(1, center_lat + 0.02, center_lon - 0.02, capacity=6),
        MockDriver(2, center_lat - 0.02, center_lon + 0.02, capacity=6)
    ]
    return orders, drivers

def plot_dynamic_vrp(assignments, orders, drivers, filename="dynamic_vrp_plot.png"):
    """
    Ve bieu do lo trinh hanh trinh dong cua cac xe.
    """
    plt.figure(figsize=(10, 8))
    
    # Ve cac diem lay/giao hang
    for o in orders:
        plt.scatter(o.pu_lon, o.pu_lat, color='green', marker='^', s=100, label='Pickup' if o.id == 1 else "")
        plt.scatter(o.do_lon, o.do_lat, color='red', marker='v', s=100, label='Delivery' if o.id == 1 else "")
        plt.text(o.pu_lon, o.pu_lat, f"PU_{o.id}", fontsize=9, ha='right', weight='bold')
        plt.text(o.do_lon, o.do_lat, f"DO_{o.id}", fontsize=9, ha='left', weight='bold')
        
    # Ve vi tri ban dau driver
    for d in drivers:
        plt.scatter(d.lon, d.lat, color='blue', marker='o', s=120, label='Driver Init Pos' if d.id == 1 else "")
        plt.text(d.lon, d.lat, f"Drv_{d.id}", fontsize=10, ha='center', va='bottom', color='blue', weight='bold')
        
    # Ve tuyen duong cho tung driver
    colors = ['#1f77b4', '#ff7f0e']
    for idx, (d_id, route) in enumerate(assignments.items()):
        drv = next(d for d in drivers if d.id == d_id)
        color = colors[idx % len(colors)]
        
        path_lons = [drv.lon]
        path_lats = [drv.lat]
        
        for node_idx in route:
            if node_idx <= len(orders):
                o = orders[node_idx - 1]
                path_lons.append(o.pu_lon)
                path_lats.append(o.pu_lat)
            else:
                o = orders[node_idx - 1 - len(orders)]
                path_lons.append(o.do_lon)
                path_lats.append(o.do_lat)
                
        plt.plot(path_lons, path_lats, color=color, linestyle='-', linewidth=2, 
                 label=f"Driver {d_id} Route ({len(route)} stops)")
        
        # Ve mui ten
        for i in range(len(path_lons) - 1):
            dx = (path_lons[i+1] - path_lons[i]) * 0.7
            dy = (path_lats[i+1] - path_lats[i]) * 0.7
            plt.annotate('', xy=(path_lons[i] + dx, path_lats[i] + dy), 
                         xytext=(path_lons[i], path_lats[i]),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
            
    plt.title("BAN DO QUY HOACH DINH TUYEN XE DONG - DYNAMIC VRP HANOI", fontsize=12, weight='bold', pad=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\n[Success] Da xuat ban do hanh trinh dong ra file: {os.path.abspath(filename)}")

def run_simulation():
    print("=" * 80)
    print(" HIEU NANG MO PHONG DYNAMIC VEHICLE ROUTING PROBLEM (DVRP) ".center(80, "="))
    print("=" * 80)
    
    # 1. Sinh du lieu gia
    orders, drivers = generate_mock_vrp_data()
    print(f"Khoi tao thanh cong: {len(orders)} don hang, {len(drivers)} tai xe")
    
    # 2. Khoi tao moi truong MDP va agent PPO
    print("\n--- [Buoc 1] Mo phong MDP va Tuyen chon hanh dong bang PPO (Reinforcement Learning) ---")
    env = DVRPEnv(drivers, orders, max_steps=10)
    obs_dim = len(drivers) * 3 + 2
    action_dim = len(drivers)
    agent = PPOAgent(obs_dim, action_dim)
    
    obs = env.reset()
    done = False
    step_count = 0
    rewards = []
    
    while not done:
        step_count += 1
        action, prob = agent.select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        obs = next_obs
        print(f" -> Step {step_count}: Reward = {reward:.2f} | Assigned count = {env.assigned_orders_count}")
        
    print(f"Ket thuc MDP: Tong Reward = {sum(rewards):.2f} | Quat trinh chay hoan tat.")
    
    # 3. Mo phong ST-GNN cho ETA Prediction
    print("\n--- [Buoc 2] Du bao ETA su dung mo hinh ST-GNN va Quantile Loss ---")
    predictor = STGNNETAPredictor(node_dim=4, edge_dim=2, time_dim=4)
    # Tinh ma tran gia lap features
    num_nodes = len(orders) * 2 + 1
    node_feats = np.random.randn(num_nodes, 4)
    adj = np.ones((num_nodes, num_nodes))
    edge_feats = np.random.randn(num_nodes, num_nodes, 2)
    
    # Chay thu du bao tu Depot (0) sang Pickup 1
    eta_preds = predictor.predict_eta(0, 1, 10.0, node_feats, adj, edge_feats)
    print("Ket qua du bao thoi gian di chuyen tu GNN (3 phan vi):")
    print(f" - Phan vi q10 (Som nhat): {eta_preds[0]:.2f} phut")
    print(f" - Phan vi q50 (Trung vi): {eta_preds[1]:.2f} phut")
    print(f" - Phan vi q90 (Bao thu - ket xe): {eta_preds[2]:.2f} phut")
    
    # 4. Giai VRPTW voi Subtour Elimination MTZ
    print("\n--- [Buoc 3] Toi uu hoa hanh trinh VRPTW (Miller-Tucker-Zemlin constraints) ---")
    solver = VRPTWSolver(speed_kmh=30.0, service_time=2.0)
    routes, total_dist = solver.solve(orders, drivers)
    
    print("-" * 80)
    print(f"KẾT QUẢ QUY HOẠCH ĐỊNH TUYẾN CUỐI CÙNG (Tong quang duong: {total_dist:.2f} km)")
    print("-" * 80)
    for d_id, route in routes.items():
        route_str = " -> ".join([f"PU_{node_idx}" if node_idx <= len(orders) else f"DO_{node_idx - len(orders)}" for node_idx in route])
        print(f"Driver {d_id}: Start -> {route_str} -> End")
    print("-" * 80)
    
    # 5. Ve lo trinh di chuyen
    plot_dynamic_vrp(routes, orders, drivers)

if __name__ == "__main__":
    run_simulation()
