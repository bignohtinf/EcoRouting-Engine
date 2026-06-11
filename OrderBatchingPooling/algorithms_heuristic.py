import math
from data_generator import haversine_distance

class STDBSCAN:
    """
    Spatiotemporal DBSCAN (ST-DBSCAN) cho Phân cụm đơn hàng.
    Khoảng cách kết hợp giữa khoảng cách địa lý pickup, khoảng cách địa lý delivery,
    và chênh lệch thời gian đặt đơn.
    """
    def __init__(self, eps_s=2.0, eps_t=15.0, min_pts=1, alpha=0.4, beta=0.4, gamma=0.2):
        """
        eps_s: Ngưỡng khoảng cách không gian (km)
        eps_t: Ngưỡng chênh lệch thời gian (phút)
        min_pts: Số điểm tối thiểu để tạo thành cụm (thường là 1 hoặc 2 cho batching đơn lẻ)
        alpha, beta, gamma: Hệ số trọng số cho hàm khoảng cách tổng hợp
        """
        self.eps_s = eps_s
        self.eps_t = eps_t
        self.min_pts = min_pts
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def calculate_distance(self, o1, o2):
        """
        Tính khoảng cách không-thời gian tổng hợp giữa hai đơn hàng.
        """
        d_pu = haversine_distance(o1.pu_lat, o1.pu_lon, o2.pu_lat, o2.pu_lon)
        d_do = haversine_distance(o1.do_lat, o1.do_lon, o2.do_lat, o2.do_lon)
        d_time = abs(o1.req_time - o2.req_time)
        
        # Hàm khoảng cách chuẩn hóa/kết hợp theo trọng số
        return self.alpha * d_pu + self.beta * d_do + self.gamma * d_time

    def range_query(self, orders, index):
        """
        Tìm tất cả các đơn hàng 'gần' đơn hàng đang xét trong ngưỡng không-thời gian.
        """
        target = orders[index]
        neighbors = []
        for i, o in enumerate(orders):
            # Điều kiện không gian: cả pickup và delivery đều không quá xa
            d_pu = haversine_distance(target.pu_lat, target.pu_lon, o.pu_lat, o.pu_lon)
            d_do = haversine_distance(target.do_lat, target.do_lon, o.do_lat, o.do_lon)
            d_time = abs(target.req_time - o.req_time)
            
            if d_pu <= self.eps_s and d_do <= self.eps_s and d_time <= self.eps_t:
                neighbors.append(i)
        return neighbors

    def fit(self, orders):
        """
        Thực hiện phân cụm các đơn hàng.
        Trả về danh sách các cụm (mỗi cụm là danh sách các Order).
        """
        n = len(orders)
        labels = [-1] * n  # -1 nghĩa là chưa phân cụm/noise
        cluster_id = 0
        
        for i in range(n):
            if labels[i] != -1:
                continue
                
            neighbors = self.range_query(orders, i)
            
            if len(neighbors) < self.min_pts:
                # Đánh dấu là nhiễu tạm thời
                labels[i] = -2  # Noise
            else:
                # Tạo cụm mới
                labels[i] = cluster_id
                
                # Mở rộng cụm
                queue = list(neighbors)
                if i in queue:
                    queue.remove(i)
                    
                idx = 0
                while idx < len(queue):
                    neighbor_idx = queue[idx]
                    
                    if labels[neighbor_idx] == -2:
                        labels[neighbor_idx] = cluster_id # Đơn nhiễu trở thành thành viên cụm
                        
                    elif labels[neighbor_idx] == -1:
                        labels[neighbor_idx] = cluster_id
                        new_neighbors = self.range_query(orders, neighbor_idx)
                        if len(new_neighbors) >= self.min_pts:
                            for nn in new_neighbors:
                                if nn not in queue:
                                    queue.append(nn)
                    idx += 1
                cluster_id += 1
                
        # Gom nhóm kết quả
        clusters = {}
        for i, label in enumerate(labels):
            if label < 0:
                # Mỗi đơn lẻ không thuộc cụm nào sẽ tạo thành 1 cụm riêng (Single-order batch)
                cluster_key = f"noise_{i}"
            else:
                cluster_key = f"cluster_{label}"
                
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(orders[i])
            
        return list(clusters.values())


class GreedyInsertionRouter:
    """
    Tối ưu hóa định tuyến (Routing) cho một Batch đơn hàng.
    Áp dụng thuật toán Greedy Insertion để tìm lịch trình Pickup và Delivery tốt nhất.
    """
    def __init__(self, speed_kmh=30.0, service_time_mins=2.0):
        self.speed = speed_kmh
        self.service_time = service_time_mins # Thời gian dừng tại mỗi điểm lấy/giao (phút)

    def solve_route(self, batch_orders, start_lat, start_lon, start_time, max_capacity):
        """
        Tìm tuyến đường tối ưu cho tài xế bắt đầu từ (start_lat, start_lon) lúc start_time.
        Quy tắc:
        - Phải đón (Pickup) trước khi giao (Delivery) cho mỗi đơn hàng.
        - Số lượng hàng trên xe tại mỗi thời điểm không được vượt quá max_capacity.
        - Tối ưu hóa tổng thời gian và kiểm tra ràng buộc Time Windows.
        """
        # Node cấu trúc: Type ('PU' hoặc 'DO'), Order, Lat, Lon
        nodes_to_visit = []
        for o in batch_orders:
            nodes_to_visit.append({'type': 'PU', 'order': o, 'lat': o.pu_lat, 'lon': o.pu_lon})
            nodes_to_visit.append({'type': 'DO', 'order': o, 'lat': o.do_lat, 'lon': o.do_lon})
            
        route = []
        current_lat, current_lon = start_lat, start_lon
        current_time = start_time
        
        visited_orders_pu = set()
        visited_orders_do = set()
        
        current_load = 0
        total_distance = 0.0
        total_delay = 0.0
        
        # Bắt đầu chèn từng điểm một cách tham lam
        while len(route) < len(nodes_to_visit):
            best_next = None
            best_cost = float('inf')
            best_dist = 0.0
            best_arrival_time = 0.0
            best_delay = 0.0
            
            for node in nodes_to_visit:
                # Kiểm tra xem điểm này đã đi chưa
                node_id = (node['type'], node['order'].id)
                if node_id in route:
                    continue
                
                # Ràng buộc Precedence: Giao hàng (DO) chỉ được thực hiện khi đã lấy hàng (PU)
                if node['type'] == 'DO' and node['order'].id not in visited_orders_pu:
                    continue
                
                # Ràng buộc Capacity: Nếu là Pickup, xem có quá tải xe không
                if node['type'] == 'PU' and current_load + node['order'].demand > max_capacity:
                    continue
                
                # Tính khoảng cách di chuyển từ vị trí hiện tại
                dist = haversine_distance(current_lat, current_lon, node['lat'], node['lon'])
                travel_time = (dist / self.speed) * 60.0 # phút
                arrival_time = current_time + travel_time
                
                # Tính phạt trễ cửa sổ thời gian (Time Windows)
                # Chỉ kiểm tra phạt trễ tại điểm giao hàng (Delivery)
                delay = 0.0
                if node['type'] == 'DO':
                    if arrival_time > node['order'].tw_end:
                        delay = arrival_time - node['order'].tw_end
                    # Nếu đến sớm hơn tw_start, xe phải chờ (không phạt nhưng tăng thời gian)
                    arrival_time = max(arrival_time, node['order'].tw_start)
                
                # Hàm chi phí tham lam: ưu tiên khoảng cách ngắn + giảm thiểu trễ hẹn
                cost = dist * 1.0 + delay * 2.0
                
                if cost < best_cost:
                    best_cost = cost
                    best_next = node
                    best_dist = dist
                    best_arrival_time = arrival_time
                    best_delay = delay
            
            if best_next is None:
                # Không tìm thấy điểm đi tiếp hợp lệ (ví dụ bị kẹt do Capacity)
                return None, float('inf'), float('inf'), False
            
            # Cập nhật trạng thái sau khi quyết định chọn best_next
            node_id = (best_next['type'], best_next['order'].id)
            route.append(node_id)
            
            if best_next['type'] == 'PU':
                visited_orders_pu.add(best_next['order'].id)
                current_load += best_next['order'].demand
            else:
                visited_orders_do.add(best_next['order'].id)
                current_load -= best_next['order'].demand
                
            total_distance += best_dist
            current_time = best_arrival_time + self.service_time
            total_delay += best_delay
            current_lat, current_lon = best_next['lat'], best_next['lon']
            
        return route, total_distance, total_delay, True


class HeuristicBatchMatcher:
    """
    Bộ giải thuật toán Heuristic để ghép tài xế với các batch đơn hàng.
    Không phụ thuộc vào thư viện bên ngoài. Phù hợp cho tính toán real-time tần suất cao.
    """
    def __init__(self, lambda_delay=15.0, mu_activation=5.0):
        self.lambda_delay = lambda_delay  # Hệ số phạt trễ thời gian (phút)
        self.mu_activation = mu_activation # Chi phí kích hoạt tài xế mới
        self.router = GreedyInsertionRouter()

    def solve(self, clusters, drivers):
        """
        Ghép cặp tài xế và các cụm đơn hàng bằng thuật toán Greedy Matching cải tiến.
        """
        unassigned_batches = list(clusters)
        available_drivers = list(drivers)
        
        assignments = [] # Lưu các cặp (Driver, Batch, Route, Distance, Delay, Cost)
        
        # Với mỗi Driver, tìm cụm (batch) tốt nhất để gán
        while len(available_drivers) > 0 and len(unassigned_batches) > 0:
            best_match = None
            best_cost = float('inf')
            
            for driver in available_drivers:
                for batch in unassigned_batches:
                    # Chạy Router để tính lộ trình cho tài xế này với cụm này
                    route, dist, delay, feasible = self.router.solve_route(
                        batch_orders=batch,
                        start_lat=driver.lat,
                        start_lon=driver.lon,
                        start_time=0.0,
                        max_capacity=driver.capacity
                    )
                    
                    if not feasible:
                        continue
                    
                    # Tính tổng chi phí của batch này
                    cost = dist + self.lambda_delay * delay + self.mu_activation
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_match = {
                            'driver': driver,
                            'batch': batch,
                            'route': route,
                            'distance': dist,
                            'delay': delay,
                            'cost': cost
                        }
            
            if best_match is None:
                # Không tìm thêm được ghép cặp hợp lệ nào
                break
                
            assignments.append(best_match)
            available_drivers.remove(best_match['driver'])
            unassigned_batches.remove(best_match['batch'])
            
        # Các batch còn lại chưa được gán (nếu thiếu tài xế)
        unassigned_orders = []
        for batch in unassigned_batches:
            unassigned_orders.extend(batch)
            
        return assignments, unassigned_orders
