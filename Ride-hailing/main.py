import time
import os
import random
import numpy as np

# Thiet lap Agg cho matplotlib de chay headless mượt mà
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import cac module noi bo
from shareability import ShareabilityNetwork, PassengerChoiceModel
from surge_pricing import SpatioTemporalSurger
from darp_solver import DARPSolver, haversine_dist

def print_banner(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def run_ride_hailing_simulation():
    print_banner("HE THONG DIEU PHOI & TOI UU HOA RIDE-HAILING CHO KHACH - XANH SM")
    
    random.seed(300)
    np.random.seed(300)
    
    # 1. Khởi tạo du lieu giả lập xe (Vehicles) va khach dat xe (Trips)
    center_lat, center_lon = 21.0285, 105.8522
    
    # Xe cap nhat theo các phan khuc VIP tiers: 1: Standard (VF5), 2: Plus (VFe34), 3: Premium (VF8)
    vehicles = [
        {'id': 1, 'lat': center_lat + 0.015, 'lon': center_lon - 0.015, 'class_tier': 1}, # Standard
        {'id': 2, 'lat': center_lat - 0.015, 'lon': center_lon + 0.015, 'class_tier': 2}, # Plus
        {'id': 3, 'lat': center_lat + 0.020, 'lon': center_lon + 0.020, 'class_tier': 3}  # Premium (VF8 VIP)
    ]
    
    trips = [
        {'id': 1, 'pu_lat': center_lat + 0.01, 'pu_lon': center_lon + 0.01, 'do_lat': center_lat + 0.03, 'do_lon': center_lon + 0.03, 'class_tier': 1},
        {'id': 2, 'pu_lat': center_lat + 0.012, 'pu_lon': center_lon + 0.008, 'do_lat': center_lat + 0.028, 'do_lon': center_lon + 0.032, 'class_tier': 1},
        {'id': 3, 'pu_lat': center_lat - 0.01, 'pu_lon': center_lon - 0.01, 'do_lat': center_lat - 0.03, 'do_lon': center_lon - 0.03, 'class_tier': 2},
        {'id': 4, 'pu_lat': center_lat + 0.02, 'pu_lon': center_lon + 0.02, 'do_lat': center_lat + 0.04, 'do_lon': center_lon + 0.04, 'class_tier': 3} # Premium order
    ]
    
    print(f"Khoi tao thanh cong:")
    print(f" - So luong xe cong nghe (VinFast fleet): {len(vehicles)}")
    print(f" - So luong khach hang dang cho xe (Trips): {len(trips)}")
    
    # 2. Dieu ap Surge Pricing (Spatio-Temporal Surge)
    print("\n--- [Buoc 1] Tinh toan Surge Pricing va Can bang Cung-Cau real-time ---")
    surger = SpatioTemporalSurger()
    
    # Gia su tai mot khu vuc o luoi dong khach, nhu cau demand = 12 don, cung avail = 5 xe
    surge_mult = surger.calculate_surge_multiplier(predicted_demand=12.0, available_supply=5.0)
    clearing_price = surger.calculate_market_clearing_price(D0=12.0, S0=5.0)
    
    print(f" - Mat do Cung-Cau o luoi: Demand = 12, Supply = 5")
    print(f" - He so nhan gia cuoc (Surge Multiplier): {surge_mult:.2f}x")
    print(f" - Gia cuoc can bang thi truong thuc te: {clearing_price:.2f} nghin VND (Base: 30 nghin VND)")
    
    # 3. Xay dung Shareability Network cho Ride-pooling
    print("\n--- [Buoc 2] Xay dung mang luoi chia se chuyen (Shareability Network) ---")
    pooling_network = ShareabilityNetwork(speed_kmh=30.0, max_detour_mins=8.0)
    edges = pooling_network.build_graph(trips)
    
    print(f"Ket qua phan tich Shareability Graph:")
    print(f" - Tim thay {len(edges)} canh lien ket co the ghep chuyen (pooling) hop le:")
    for edge in edges:
        print(f"   * Trip {edge['node1'] + 1} va Trip {edge['node2'] + 1} ghep chung tuyen {edge['option']}:")
        print(f"     + Tiet kiem hanh trinh: {edge['weight']:.2f} km")
        print(f"     + Thoi gian di vong: Khach 1 (+{edge['detour1']:.1f}m), Khach 2 (+{edge['detour2']:.1f}m)")
        
    # 4. Giai bai toan Dial-a-Ride Problem (DARP)
    print("\n--- [Buoc 3] Quy hoach dinh tuyen Dial-a-Ride Problem (DARP) voi VIP tiers ---")
    darp = DARPSolver(speed_kmh=30.0, max_ride_time_mins=15.0)
    assignments, unassigned = darp.solve_darp(trips, vehicles)
    
    print("-" * 80)
    print("KET QUA PHAN BO VA DINH TUYEN DIAL-A-RIDE (DARP):")
    print("-" * 80)
    for v_id, route in assignments.items():
        v = next(veh for veh in vehicles if veh['id'] == v_id)
        tier_names = {1: 'Standard', 2: 'Plus', 3: 'Premium VIP'}
        
        if not route:
            print(f"Xe {v_id} ({tier_names[v['class_tier']]}): Khong co lich trinh")
            continue
            
        route_str = " -> ".join([f"{node[0]}_{node[1]['id']}" for node in route])
        print(f"Xe {v_id} ({tier_names[v['class_tier']]}): Start -> {route_str} -> End")
        
    if unassigned:
        print(f" - Cac chuyen chua gán duoc: {', '.join([str(t['id']) for t in unassigned])}")
    print("-" * 80)
    
    # 5. Ve va xuat bieu do
    # Do thi 1: Bản đồ định tuyến DARP
    plt.figure(figsize=(10, 8))
    for v in vehicles:
        plt.scatter(v['lon'], v['lat'], color='blue', marker='o', s=120)
        plt.text(v['lon'], v['lat'], f"Drv_{v['id']}\n(Tier {v['class_tier']})", color='blue', fontsize=8, ha='center', va='bottom')
        
    for t in trips:
        plt.scatter(t['pu_lon'], t['pu_lat'], color='green', marker='^', s=100)
        plt.text(t['pu_lon'], t['pu_lat'], f"PU_{t['id']}", color='darkgreen', fontsize=9, ha='right', weight='bold')
        
        plt.scatter(t['do_lon'], t['do_lat'], color='red', marker='v', s=100)
        plt.text(t['do_lon'], t['do_lat'], f"DO_{t['id']}", color='darkred', fontsize=9, ha='left', weight='bold')
        
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, (v_id, route) in enumerate(assignments.items()):
        if not route:
            continue
        v = next(veh for veh in vehicles if veh['id'] == v_id)
        color = colors[idx % len(colors)]
        
        path_lons = [v['lon']]
        path_lats = [v['lat']]
        
        for node_type, trip in route:
            if node_type == 'PU':
                path_lons.append(trip['pu_lon'])
                path_lats.append(trip['pu_lat'])
            else:
                path_lons.append(trip['do_lon'])
                path_lats.append(trip['do_lat'])
                
        plt.plot(path_lons, path_lats, color=color, linestyle='-', linewidth=2, label=f"Driver {v_id} (Tier {v['class_tier']})")
        
        for i in range(len(path_lons) - 1):
            dx = (path_lons[i+1] - path_lons[i]) * 0.7
            dy = (path_lats[i+1] - path_lats[i]) * 0.7
            plt.annotate('', xy=(path_lons[i] + dx, path_lats[i] + dy), 
                         xytext=(path_lons[i], path_lats[i]),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
            
    plt.title("BAN DO QUY HOACH DIAL-A-RIDE (DARP) CO PHAN CAP VIP TIERS - XANH SM", weight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.savefig("ride_hailing_darp_map.png", dpi=300)
    plt.close()
    print(f"\n[Success] Da xuat ban do lo trinh cho khach ra file: {os.path.abspath('ride_hailing_darp_map.png')}")
    
    # Do thi 2: Shareability Network
    plt.figure(figsize=(8, 6))
    # Ve cac nut chuyến đi
    trip_x = [t['pu_lon'] for t in trips]
    trip_y = [t['pu_lat'] for t in trips]
    plt.scatter(trip_x, trip_y, color='darkorange', s=200, zorder=3)
    
    for t in trips:
        plt.text(t['pu_lon'], t['pu_lat'], f"Trip_{t['id']}\n(Tier {t['class_tier']})", ha='center', va='center', color='white', weight='bold', fontsize=8, zorder=4)
        
    # Ve cac canh shareable
    for edge in edges:
        n1 = trips[edge['node1']]
        n2 = trips[edge['node2']]
        plt.plot([n1['pu_lon'], n2['pu_lon']], [n1['pu_lat'], n2['pu_lat']], color='green', linestyle='-', linewidth=3, zorder=2, label='Feasible Pooling' if edge == edges[0] else "")
        plt.text((n1['pu_lon'] + n2['pu_lon'])/2.0, (n1['pu_lat'] + n2['pu_lat'])/2.0, f"Saved {edge['weight']:.2f}km", color='green', fontsize=8, weight='bold')
        
    plt.title("DO THI LIEN KET GHEP CHUYEN - SANTI SHAREABILITY NETWORK", weight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("shareability_network.png", dpi=300)
    plt.close()
    print(f"[Success] Da xuat do thi shareability network ra file: {os.path.abspath('shareability_network.png')}")

if __name__ == "__main__":
    run_ride_hailing_simulation()
