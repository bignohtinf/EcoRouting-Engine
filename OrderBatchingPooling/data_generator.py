import random
import math

class Order:
    def __init__(self, order_id, pu_lat, pu_lon, do_lat, do_lon, demand, req_time, tw_start, tw_end):
        self.id = order_id
        self.pu_lat = pu_lat
        self.pu_lon = pu_lon
        self.do_lat = do_lat
        self.do_lon = do_lon
        self.demand = demand          # Sức nặng/Số lượng phần ăn (ví dụ: 1, 2, 3)
        self.req_time = req_time      # Thời điểm đặt đơn (phút kể từ mốc 0)
        self.tw_start = tw_start      # Cửa sổ thời gian nhận/giao bắt đầu
        self.tw_end = tw_end          # Cửa sổ thời gian nhận/giao kết thúc

    def __repr__(self):
        return (f"Order(ID={self.id}, Demand={self.demand}, Req={self.req_time:.1f}m, "
                f"TW=[{self.tw_start:.1f}, {self.tw_end:.1f}]m)")

class Driver:
    def __init__(self, driver_id, lat, lon, capacity=5, speed_kmh=30):
        self.id = driver_id
        self.lat = lat
        self.lon = lon
        self.capacity = capacity      # Sức chứa tối đa của thùng chứa (xe máy/ô tô điện)
        self.speed_kmh = speed_kmh    # Tốc độ di chuyển trung bình trong đô thị (km/h)

    def __repr__(self):
        return f"Driver(ID={self.id}, Pos=({self.lat:.4f}, {self.lon:.4f}), Cap={self.capacity})"

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Tính khoảng cách mặt cầu Haversine giữa 2 điểm (vĩ độ, kinh độ) theo đơn vị km.
    """
    R = 6371.0 # Bán kính Trái Đất trung bình (km)
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def generate_mock_data(num_orders=20, num_drivers=5, seed=42):
    """
    Giả lập dữ liệu các đơn hàng và tài xế tại khu vực Hà Nội.
    Tâm bản đồ: Hồ Hoàn Kiếm (21.0285, 105.8522)
    Bán kính phân bổ: khoảng 5-7 km.
    """
    random.seed(seed)
    center_lat, center_lon = 21.0285, 105.8522
    
    orders = []
    drivers = []
    
    # 1. Sinh dữ liệu các đơn hàng
    for i in range(1, num_orders + 1):
        # Điểm Pickup (lấy đồ) xung quanh khu vực các nhà hàng ẩm thực
        pu_lat = center_lat + random.uniform(-0.04, 0.04)
        pu_lon = center_lon + random.uniform(-0.04, 0.04)
        
        # Điểm Delivery (giao đồ) xa hơn một chút đại diện cho khu dân cư/văn phòng
        do_lat = pu_lat + random.uniform(-0.03, 0.03)
        do_lon = pu_lon + random.uniform(-0.03, 0.03)
        
        demand = random.choice([1, 2, 3])
        req_time = random.uniform(0, 30) # đơn nổ trong vòng 30 phút đầu ca
        
        # Cửa sổ thời gian giao hàng (ví dụ: cần giao trong vòng 30 - 60 phút từ lúc đặt)
        # a_i = req_time + thời gian di chuyển ước lượng + offset
        direct_dist = haversine_distance(pu_lat, pu_lon, do_lat, do_lon)
        est_travel_time = (direct_dist / 30.0) * 60.0 # phút (với tốc độ 30 km/h)
        
        tw_start = req_time + est_travel_time + random.uniform(5, 15)
        tw_end = tw_start + random.uniform(20, 45) # thời gian đệm cho khách nhận đồ
        
        orders.append(Order(
            order_id=i,
            pu_lat=pu_lat, pu_lon=pu_lon,
            do_lat=do_lat, do_lon=do_lon,
            demand=demand,
            req_time=req_time,
            tw_start=tw_start,
            tw_end=tw_end
        ))
        
    # 2. Sinh dữ liệu các tài xế
    for j in range(1, num_drivers + 1):
        # Vị trí tài xế rải rác xung quanh trung tâm
        d_lat = center_lat + random.uniform(-0.05, 0.05)
        d_lon = center_lon + random.uniform(-0.05, 0.05)
        
        drivers.append(Driver(
            driver_id=j,
            lat=d_lat, lon=d_lon,
            capacity=random.choice([4, 5, 6])
        ))
        
    return orders, drivers

if __name__ == "__main__":
    # Chạy thử sinh dữ liệu
    orders, drivers = generate_mock_data(5, 2)
    print("--- DEMO GENERATED DATA ---")
    print(f"Total simulated orders: {len(orders)}")
    for o in orders:
        print(f" - {o}")
    print(f"\nTotal simulated drivers: {len(drivers)}")
    for d in drivers:
        print(f" - {d}")
