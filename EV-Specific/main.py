import time
import os
import numpy as np

# Thiet lap Agg cho matplotlib de chay headless mượt mà
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import cac module noi bo
from energy_model import EVEnergyModel
from evrp_solver import EVRPSolver, haversine_dist
from queuing_network import ChargingQueuingModel, ChargingFacilityPlanner

def print_banner(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def run_ev_simulation():
    print_banner("HE THONG MO PHONG TOI UU HOA XE DIEN (EV-SPECIFIC) - VINFAST")
    
    # 1. Sinh du lieu gia lập (Hanoi)
    center_lat, center_lon = 21.0285, 105.8522
    
    # Cac diem khach hang (stops)
    stops = [
        {'id': 1, 'lat': center_lat + 0.04, 'lon': center_lon + 0.04},
        {'id': 2, 'lat': center_lat + 0.08, 'lon': center_lon + 0.08},
        {'id': 3, 'lat': center_lat + 0.12, 'lon': center_lon + 0.12},
        {'id': 4, 'lat': center_lat + 0.02, 'lon': center_lon + 0.15}
    ]
    
    # Trạm sạc V-Green khả dụng
    charging_stations = [
        {'id': 1, 'lat': center_lat + 0.06, 'lon': center_lon + 0.06, 'max_power_kw': 150.0},
        {'id': 2, 'lat': center_lat + 0.10, 'lon': center_lon + 0.10, 'max_power_kw': 250.0}
    ]
    
    print(f"Khoi tao thanh cong:")
    print(f" - So luong diem khach hang: {len(stops)}")
    print(f" - So luong tram sac sieu nhanh V-Green: {len(charging_stations)}")
    
    # 2. Quy hoach dinh tuyen voi E-VRP-CS (SoC range safety limits)
    print("\n--- [Buoc 1] Lap lich trinh di chuyen va sac pin tu dong (E-VRP-CS) ---")
    solver = EVRPSolver(battery_cap_kwh=84.0, min_soc_pct=20.0, max_soc_pct=100.0)
    
    # Xuat phat tai Depot voi SoC ban dau thap (45% SoC) de kich hoat tu dong sac pin dọc duong
    start_lat, start_lon = center_lat, center_lon
    route, soc_history, charging_actions, total_dist = solver.solve_route_with_charging(
        start_lat, start_lon, stops, charging_stations, initial_soc_pct=45.0
    )
    
    print("-" * 80)
    print("HANH TRINH DINH TUYEN XE DIEN (CO CHEN TRAM SAC):")
    print("-" * 80)
    print(f" - Lo trinh: Start -> {' -> '.join(route)}")
    print(f" - Tong quang duong di chuyen: {total_dist:.2f} km")
    print(f" - SoC Bien thien: " + " -> ".join([f"{s:.1f}%" for s in soc_history]))
    
    if charging_actions:
        print("\nCHI TIET HANH DONG SAC PIN:");
        for act in charging_actions:
            print(f" - Sac tai Tram {act['station_id']}: {act['start_soc']:.1f}% -> {act['end_soc']:.1f}% trong {act['duration_mins']:.1f} phut (CC-CV)")
    print("-" * 80)
    
    # 3. Danh gia xep hang tai tram sac (M/M/c Queueing model)
    print("\n--- [Buoc 2] Mo phong hang doi cho tai tram sac (M/M/c Erlang C) ---")
    queuing = ChargingQueuingModel()
    # Gia su tram co 3 tru sac, 4 xe den/gio, moi xe sac 30 phut (service_rate = 2 xe/gio)
    prob_wait, W_q = queuing.calculate_waiting_time(arrival_rate_lambda=4.0, service_rate_mu=2.0, num_chargers_c=3)
    
    print(f" - So luong dau sac tai tram (c): 3 tru")
    print(f" - Xac suat xe phai xep hang doi sac: {prob_wait * 100.0:.2f}%")
    print(f" - Thoi gian cho doi trung binh cua EV: {W_q:.2f} phut")
    
    # 4. Quy hoach lap tram sac moi (Facility Location planning)
    print("\n--- [Buoc 3] Quy hoach lap tram sac moi trong khu do thi (Facility Location) ---")
    planner = ChargingFacilityPlanner()
    demand_points = [
        {'id': 1, 'lat': center_lat + 0.01, 'lon': center_lon + 0.01, 'load': 50},
        {'id': 2, 'lat': center_lat + 0.03, 'lon': center_lon + 0.03, 'load': 120},
        {'id': 3, 'lat': center_lat + 0.05, 'lon': center_lon + 0.02, 'load': 80}
    ]
    candidate_sites = [
        {'id': 1, 'lat': center_lat + 0.02, 'lon': center_lon + 0.02, 'fixed_cost': 40.0, 'cap_cost': 15.0},
        {'id': 2, 'lat': center_lat + 0.04, 'lon': center_lon + 0.04, 'fixed_cost': 60.0, 'cap_cost': 20.0}
    ]
    opened_sites, spent = planner.plan_charging_network(demand_points, candidate_sites, budget=90.0)
    print(f" - Quy hoach thanh cong: da mo {len(opened_sites)} tram sac moi")
    for site in opened_sites:
        print(f"   * Mo tram tai Candidate Site {site['id']} (Ngan sach tieu hao: {site['fixed_cost'] + site['cap_cost']*2:.1f} ty VND)")

    # 5. Ve va xuat bieu do
    # Do thi 1: Bản đồ lộ trình
    plt.figure(figsize=(10, 8))
    plt.scatter(center_lon, center_lat, color='blue', marker='o', s=150, label='Depot')
    for s in stops:
        plt.scatter(s['lon'], s['lat'], color='red', marker='v', s=100, label='Customer' if s['id'] == 1 else "")
        plt.text(s['lon'], s['lat'], f"Cust_{s['id']}", fontsize=9, ha='right', weight='bold')
    for cs in charging_stations:
        plt.scatter(cs['lon'], cs['lat'], color='green', marker='P', s=120, label='V-Green Station' if cs['id'] == 1 else "")
        plt.text(cs['lon'], cs['lat'], f"Station_{cs['id']}", fontsize=9, ha='left', color='darkgreen', weight='bold')
        
    # Ve line duong di
    curr_x, curr_y = center_lon, center_lat
    for node in route:
        if node.startswith("Cust_"):
            c_id = int(node.split("_")[1])
            target = next(s for s in stops if s['id'] == c_id)
            plt.plot([curr_x, target['lon']], [curr_y, target['lat']], color='purple', linestyle='-', linewidth=2)
            curr_x, curr_y = target['lon'], target['lat']
        elif node.startswith("CS_"):
            cs_id = int(node.split("_")[1])
            target = next(c for c in charging_stations if c['id'] == cs_id)
            plt.plot([curr_x, target['lon']], [curr_y, target['lat']], color='green', linestyle=':', linewidth=2.5)
            curr_x, curr_y = target['lon'], target['lat']
            
    plt.title("BAN DO LO TRINH DI CHUYEN & VE SAC PIN - E-VRP-CS", weight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.savefig("ev_route_map.png", dpi=300)
    plt.close()
    print(f"\n[Success] Da xuat ban do lo trinh sac ra file: {os.path.abspath('ev_route_map.png')}")
    
    # Do thi 2: Biến thiên SoC
    plt.figure(figsize=(9, 5))
    x_steps = list(range(len(soc_history)))
    plt.plot(x_steps, soc_history, color='darkorange', marker='s', linewidth=2.5, label='EV State-of-Charge')
    
    # Gan nhan cac nut vao bieu do
    labels = ["Start"]
    for node in route:
        labels.append(node)
        if node.startswith("CS_"):
            # Tram sac co hai moc SoC truoc/sau sạc
            labels.append(f"{node} (Charged)")
            
    # Chinh cac nhan cho vua khop soc_history
    if len(labels) < len(soc_history):
        labels.append("End")
    labels = labels[:len(soc_history)]
    
    plt.xticks(x_steps, labels, rotation=20, ha='right', fontsize=9)
    plt.axhline(y=20.0, color='red', linestyle='--', label='Min Safe SoC (20%)')
    plt.ylabel("SoC (%)", fontsize=11, weight='bold')
    plt.title("BIEN THIEN NANG LUONG PIN SOC (%) THEO HANH TRINH & TRAM SAC", weight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 105)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig("ev_soc_profile.png", dpi=300)
    plt.close()
    print(f"[Success] Da xuat bieu do bien thien SoC ra file: {os.path.abspath('ev_soc_profile.png')}")

if __name__ == "__main__":
    run_ev_simulation()
