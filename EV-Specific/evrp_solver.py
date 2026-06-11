import numpy as np
from energy_model import EVEnergyModel

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2))
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

class EVRPSolver:
    """
    Bo quy hoach dinh tuyen cho xe dien (E-VRP-CS).
    Tich hop SoC continuity va tu dong chen tram sac V-Green khi pin xuong duoi nguong an toan.
    """
    def __init__(self, battery_cap_kwh=84.0, min_soc_pct=15.0, max_soc_pct=100.0, speed_kmh=30.0):
        self.battery_cap = battery_cap_kwh
        self.min_soc = min_soc_pct
        self.max_soc = max_soc_pct
        self.speed = speed_kmh
        self.energy_model = EVEnergyModel()

    def solve_route_with_charging(self, start_lat, start_lon, stops, charging_stations, initial_soc_pct=90.0):
        """
        stops: Danh sach cac diem khach hang (id, lat, lon)
        charging_stations: Danh sach cac trạm sac (id, lat, lon, max_power_kw)
        Tra ve: route (chuoi cac diem di qua), soc_history (bien thien SoC), charging_actions (lich sac), total_dist
        """
        route = []
        soc_history = [initial_soc_pct]
        charging_actions = []
        
        curr_lat, curr_lon = start_lat, start_lon
        curr_soc = initial_soc_pct
        total_dist = 0.0
        
        unvisited = list(stops)
        
        while unvisited:
            # Tim diem khach hang gan nhat
            best_stop = None
            best_dist = float('inf')
            for s in unvisited:
                d = haversine_dist(curr_lat, curr_lon, s['lat'], s['lon'])
                if d < best_dist:
                    best_dist = d
                    best_stop = s
                    
            if best_stop is None:
                break
                
            # Uoc tinh tieu thu nang luong den diem khach hang tiep theo
            duration_h = best_dist / self.speed
            kwh_consumed = self.energy_model.calculate_consumption(
                dist_km=best_dist,
                elevation_gain_m=0.0,
                speed_kmh=self.speed,
                load_kg=80.0, # gia su load hanh khach mac dinh
                duration_h=duration_h
            )
            soc_dropped_pct = (kwh_consumed / self.battery_cap) * 100.0
            
            # Neu SoC sau khi di se rot duoi muc an toan (min_soc) -> PHAI SAC PIN TRUOC
            if curr_soc - soc_dropped_pct < self.min_soc:
                # Tim trạm sac gan nhat de re vao sạc
                best_station = None
                best_station_dist = float('inf')
                for cs in charging_stations:
                    d_to_cs = haversine_dist(curr_lat, curr_lon, cs['lat'], cs['lon'])
                    if d_to_cs < best_station_dist:
                        best_station_dist = d_to_cs
                        best_station = cs
                        
                if best_station is not None:
                    # 1. Di chuyen đen trạm sạc
                    route.append(f"CS_{best_station['id']}")
                    total_dist += best_station_dist
                    
                    # Tieu thu nang luong den tram sac
                    kwh_to_cs = self.energy_model.calculate_consumption(
                        dist_km=best_station_dist,
                        elevation_gain_m=0.0,
                        speed_kmh=self.speed,
                        load_kg=80.0,
                        duration_h=best_station_dist / self.speed
                    )
                    curr_soc -= (kwh_to_cs / self.battery_cap) * 100.0
                    soc_history.append(curr_soc)
                    
                    # 2. Thuc hien hanh dong sạc (sạc den 85% SoC de tiet kiem thoi gian va giam suy hao)
                    target_soc = 85.0
                    charge_time = self.energy_model.get_cccv_charging_time(
                        start_soc_pct=curr_soc,
                        end_soc_pct=target_soc,
                        max_power_kw=best_station['max_power_kw'],
                        battery_cap_kwh=self.battery_cap
                    )
                    
                    charging_actions.append({
                        'station_id': best_station['id'],
                        'start_soc': curr_soc,
                        'end_soc': target_soc,
                        'duration_mins': charge_time
                    })
                    
                    curr_soc = target_soc
                    soc_history.append(curr_soc)
                    
                    # Cap nhat vi tri sang vi tri trạm sạc
                    curr_lat, curr_lon = best_station['lat'], best_station['lon']
                    
                    # Tinh toan lai quang duong va tieu thu tu tram sac den best_stop
                    best_dist = haversine_dist(curr_lat, curr_lon, best_stop['lat'], best_stop['lon'])
                    duration_h = best_dist / self.speed
                    kwh_consumed = self.energy_model.calculate_consumption(
                        dist_km=best_dist,
                        elevation_gain_m=0.0,
                        speed_kmh=self.speed,
                        load_kg=80.0,
                        duration_h=duration_h
                    )
                    soc_dropped_pct = (kwh_consumed / self.battery_cap) * 100.0
                    
            # Thuc hien di chuyen den diem khach hang best_stop
            route.append(f"Cust_{best_stop['id']}")
            total_dist += best_dist
            curr_soc -= soc_dropped_pct
            soc_history.append(curr_soc)
            
            # Cap nhat vi tri hien tai
            curr_lat, curr_lon = best_stop['lat'], best_stop['lon']
            unvisited.remove(best_stop)
            
        return route, soc_history, charging_actions, total_dist
