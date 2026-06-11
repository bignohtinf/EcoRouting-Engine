import numpy as np

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2))
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

class DARPSolver:
    """
    Bo giai bai toan Dial-a-Ride Problem (DARP) cho Ride-hailing.
    Rang buoc:
    - Precedence: don truoc, tra sau
    - Pairing: don va tra tren cung 1 xe
    - Max ride time: gioi han thoi gian hanh khach tren xe
    - Lexicographic Tier Compatibility: Khach VIP (Premium) chi duoc di xe Premium (e.g. VF8)
    """
    def __init__(self, speed_kmh=30.0, max_ride_time_mins=15.0):
        self.speed = speed_kmh
        self.max_ride_time = max_ride_time_mins

    def solve_darp(self, trips, vehicles):
        """
        trips: Danh sach cac trip request (id, pu_lat, pu_lon, do_lat, do_lon, class_tier [1: Standard, 2: Plus, 3: Premium])
        vehicles: Danh sach xe (id, lat, lon, class_tier)
        Tra ve: assignments {vehicle_id: [node_sequence]}, unassigned
        """
        assignments = {v['id']: [] for v in vehicles}
        unassigned_trips = list(trips)
        
        # Toi uu hoa phan cap Lexicographic (VIP Tier): giai tu hang cao xuong hang thap
        # Sap xep don hang: Premium (3) -> Plus (2) -> Standard (1)
        unassigned_trips.sort(key=lambda t: t['class_tier'], reverse=True)
        
        for trip in list(unassigned_trips):
            best_vehicle = None
            best_cost = float('inf')
            best_route = []
            
            # Quet qua tat ca cac xe phu hop hang xe (VIP Tier compatibility)
            for v in vehicles:
                # Ràng buộc Tier: Xe phai co hang >= hang yeu cau cua khach
                if v['class_tier'] < trip['class_tier']:
                    continue
                    
                # Thu chen tuyen duong don/tra
                curr_route = assignments[v['id']]
                
                # Check feasibility va chi phi
                feasible, new_route, cost = self._try_insert_trip(v, curr_route, trip)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_vehicle = v
                    best_route = new_route
                    
            if best_vehicle is not None:
                assignments[best_vehicle['id']] = best_route
                unassigned_trips.remove(trip)
                
        return assignments, unassigned_trips

    def _try_insert_trip(self, vehicle, current_route, trip):
        """
        Thu chen diem pickup va dropoff cua trip moi vao route hien tai cua xe.
        Dam bao rang buoc Precedence (don truoc tra sau) va Max Ride Time.
        """
        trip_id = trip['id']
        n = len(current_route)
        
        best_cost = float('inf')
        best_seq = []
        
        # Duyet qua cac vi tri chen pickup (i) va dropoff (j) voi i <= j
        # Node format: (node_type ['PU', 'DO'], trip_dict)
        nodes = []
        for node in current_route:
            nodes.append(node)
            
        pu_node = ('PU', trip)
        do_node = ('DO', trip)
        
        for i in range(len(nodes) + 1):
            for j in range(i, len(nodes) + 1):
                # Tao chuoi moi sau khi chen
                temp_route = list(nodes)
                temp_route.insert(i, pu_node)
                temp_route.insert(j + 1, do_node) # j + 1 de luon đung sau pickup
                
                # Kiem tra cac rang buoc
                feasible, cost = self._check_route_feasibility(vehicle, temp_route)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_seq = temp_route
                    
        if best_cost < float('inf'):
            return True, best_seq, best_cost
        return False, [], float('inf')

    def _check_route_feasibility(self, vehicle, route):
        """
        Kiểm tra cac rang buoc:
        - Max Ride Time cua tung hanh khach
        - Sức chứa ghe xe (capacity - gia su xe 4 cho cho DARP ghep)
        """
        curr_lat, curr_lon = vehicle['lat'], vehicle['lon']
        curr_time = 0.0
        
        # Tracking thoi gian pickup cua tung trip de check ride time
        pickup_times = {}
        total_dist = 0.0
        current_load = 0
        
        for node_type, trip in route:
            # Tinh khoang cach va thoi gian di chuyen
            target_lat = trip['pu_lat'] if node_type == 'PU' else trip['do_lat']
            target_lon = trip['pu_lon'] if node_type == 'PU' else trip['do_lon']
            
            d = haversine_dist(curr_lat, curr_lon, target_lat, target_lon)
            travel_time = (d / self.speed) * 60.0
            
            curr_time += travel_time
            total_dist += d
            
            if node_type == 'PU':
                pickup_times[trip['id']] = curr_time
                current_load += 1
            else:
                # Dropoff: Kiem tra precedence va max ride time
                if trip['id'] not in pickup_times:
                    return False, float('inf') # Delivery truoc Pickup -> Infeasible
                    
                ride_time = curr_time - pickup_times[trip['id']]
                if ride_time > self.max_ride_time:
                    return False, float('inf') # Vi pham Max ride time
                    
                current_load -= 1
                
            if current_load > 3: # Gioi han ghe ngoi
                return False, float('inf')
                
            curr_lat, curr_lon = target_lat, target_lon
            
        return True, total_dist
