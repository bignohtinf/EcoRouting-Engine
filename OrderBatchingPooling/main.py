import time
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from data_generator import generate_mock_data
from algorithms_heuristic import STDBSCAN, HeuristicBatchMatcher
from algorithms_milp import MILPBatchMatcher

def print_banner(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def display_results_table(assignments, unassigned_orders, title):
    print_banner(title)
    print(f"{'Driver ID':<10} | {'Active':<8} | {'Orders in Batch':<20} | {'Distance (km)':<13} | {'Delay (mins)':<12} | {'Route Sequence'}")
    print("-" * 100)
    
    total_dist = 0.0
    total_delay = 0.0
    active_drivers = 0
    
    for assign in assignments:
        drv = assign['driver']
        batch = assign['batch']
        route = assign['route']
        dist = assign['distance']
        delay = assign['delay']
        
        total_dist += dist
        total_delay += delay
        active_drivers += 1
        
        order_ids = ", ".join([str(o.id) for o in batch])
        route_str = " -> ".join([f"{node[0]}_{node[1]}" for node in route])
        
        print(f"Driver {drv.id:<3} | {'YES':<8} | {f'[{order_ids}]':<20} | {dist:<13.2f} | {delay:<12.2f} | Start -> {route_str}")
        
    print("-" * 100)
    print(f"TONG HOP CHI TIET:")
    print(f" - So tai xe duoc kich hoat: {active_drivers} / {active_drivers + len(assignments)}")
    print(f" - Tong quang duong di chuyen: {total_dist:.2f} km")
    print(f" - Tong thoi gian tre hen: {total_delay:.2f} phut")
    print(f" - So don hang chua duoc gan: {len(unassigned_orders)}")
    if unassigned_orders:
        print(f"   (Cac don chua gan ID: {', '.join([str(o.id) for o in unassigned_orders])})")

def plot_routes(assignments, orders, drivers, filename="route_plot.png"):
    if not HAS_MATPLOTLIB:
        print("\n[Notice] Do thi khong the xuat vi thieu thu vien 'matplotlib'. Vui long cai dat bang: pip install matplotlib")
        return
    plt.figure(figsize=(12, 10))
    
    for o in orders:
        plt.scatter(o.pu_lon, o.pu_lat, color='green', marker='^', s=100, label='Pickup' if o.id == 1 else "")
        plt.scatter(o.do_lon, o.do_lat, color='red', marker='v', s=100, label='Delivery' if o.id == 1 else "")
        plt.text(o.pu_lon, o.pu_lat, f"PU_{o.id}", fontsize=9, ha='right', color='darkgreen', weight='bold')
        plt.text(o.do_lon, o.do_lat, f"DO_{o.id}", fontsize=9, ha='left', color='darkred', weight='bold')
        
    for d in drivers:
        plt.scatter(d.lon, d.lat, color='blue', marker='o', s=120, label='Driver Init Pos' if d.id == 1 else "")
        plt.text(d.lon, d.lat, f"Drv_{d.id}", fontsize=10, ha='center', va='bottom', color='blue', weight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, assign in enumerate(assignments):
        drv = assign['driver']
        batch = assign['batch']
        route = assign['route']
        color = colors[idx % len(colors)]
        
        curr_lon, curr_lat = drv.lon, drv.lat
        
        path_lons = [curr_lon]
        path_lats = [curr_lat]
        
        for node_type, order_id in route:
            order = next(o for o in batch if o.id == order_id)
            if node_type == 'PU':
                path_lons.append(order.pu_lon)
                path_lats.append(order.pu_lat)
            else:
                path_lons.append(order.do_lon)
                path_lats.append(order.do_lat)
                
        plt.plot(path_lons, path_lats, color=color, linestyle='-', linewidth=2.5, 
                 label=f"Route Tai xe {drv.id} (Don {', '.join([str(o.id) for o in batch])})")
        
        for i in range(len(path_lons) - 1):
            dx = (path_lons[i+1] - path_lons[i]) * 0.7
            dy = (path_lats[i+1] - path_lats[i]) * 0.7
            plt.annotate('', xy=(path_lons[i] + dx, path_lats[i] + dy), 
                         xytext=(path_lons[i], path_lats[i]),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    plt.title("BAN DO TRUC QUAN HOA LO TRINH GHEP DON (ORDER BATCHING & POOLING) - HA NOI", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Kinh do (Longitude)", fontsize=11)
    plt.ylabel("Vi do (Latitude)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Success] Da xuat ban do truc quan hoa lo trinh ra file anh: {os.path.abspath(filename)}")

def run_simulation():
    print_banner("HE THONG MO PHONG ORDER BATCHING & POOLING - GREEN SM")
    
    num_orders = 15
    num_drivers = 4
    orders, drivers = generate_mock_data(num_orders=num_orders, num_drivers=num_drivers, seed=42)
    
    print(f"Khoi tao thanh cong:")
    print(f" - So luong don hang can giao: {len(orders)}")
    print(f" - So luong tai xe truc ca: {len(drivers)}")
    
    print("\n--- [Buoc 1] Tien hanh phan cum Spatiotemporal Clustering (ST-DBSCAN) ---")
    clustering = STDBSCAN(eps_s=2.5, eps_t=20.0, min_pts=1, alpha=0.4, beta=0.4, gamma=0.2)
    clusters = clustering.fit(orders)
    print(f"Phan cum hoan tat: gom duoc {len(clusters)} cum/batch don hang tu {len(orders)} don goc.")
    for idx, cluster in enumerate(clusters):
        ids = [o.id for o in cluster]
        print(f" - Cum {idx + 1}: Cac don hang {ids}")
        
    print("\n--- [Buoc 2A] Thuc hien toi uu hoa gan ghep bang mo hinh MILP (SciPy) ---")
    milp_solver = MILPBatchMatcher(lambda_delay=10.0, mu_activation=5.0)
    t_start = time.time()
    milp_assign, milp_unassigned = milp_solver.solve(clusters, drivers)
    t_milp = (time.time() - t_start) * 1000.0
    
    display_results_table(milp_assign, milp_unassigned, f"KET QUA TOI UU HOA MILP (Thoi gian tinh toan: {t_milp:.2f}ms)")
    
    print("\n--- [Buoc 2B] Thuc hien toi uu hoa gan ghep bang Heuristic (Greedy Matching) ---")
    heuristic_solver = HeuristicBatchMatcher(lambda_delay=10.0, mu_activation=5.0)
    t_start = time.time()
    heur_assign, heur_unassigned = heuristic_solver.solve(clusters, drivers)
    t_heur = (time.time() - t_start) * 1000.0
    
    display_results_table(heur_assign, heur_unassigned, f"KET QUA TOI UU HOA HEURISTIC (Thoi gian tinh toan: {t_heur:.2f}ms)")
    
    if milp_assign:
        try:
            plot_routes(milp_assign, orders, drivers, filename="route_visualization.png")
        except Exception as e:
            print(f"\n[Warning] Khong the xuat do thi truc quan hoa: {e}")

if __name__ == "__main__":
    run_simulation()
