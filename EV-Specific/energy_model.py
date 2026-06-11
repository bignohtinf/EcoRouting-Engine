import numpy as np

class EVEnergyModel:
    """
    Mo hinh tieu thu nang luong, duong cong sac CC-CV, va suy hao pin (DoD) cua xe dien VinFast.
    """
    def __init__(self, alpha=0.15, beta=0.05, gamma=0.0001, delta=0.01, eps_hvac=0.5):
        """
        alpha: Tieu thu co ban theo quang duong (kWh/km)
        beta: Năng luong leo doc (kWh/met do cao chenh lech)
        gamma: Luc can khi dong hoc (phu thuoc vao binh phuong toc do)
        delta: Anh huong cua tai trong (kWh/kg/km)
        eps_hvac: Tieu thu dieu hoa nhiet do theo thoi gian (kWh/gio)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eps_hvac = eps_hvac

    def calculate_consumption(self, dist_km, elevation_gain_m, speed_kmh, load_kg, duration_h):
        """
        Cong thuc tieu thu nang luong phi tuyen tai Section 5.1:
        h_ij = alpha * d_ij + beta * delta_h + gamma * v^2 * d_ij + delta * m_load * d_ij + eps_HVAC
        """
        h_base = self.alpha * dist_km
        h_elevation = self.beta * (elevation_gain_m / 100.0) # quy doi tram met do cao
        h_aerodynamic = self.gamma * (speed_kmh ** 2) * dist_km
        h_load = self.delta * (load_kg / 100.0) * dist_km
        h_hvac = self.eps_hvac * duration_h
        
        total_kwh = h_base + h_elevation + h_aerodynamic + h_load + h_hvac
        return max(0.05, total_kwh) # Luon tieu thu mot luong nang luong toi thieu

    def get_cccv_charging_time(self, start_soc_pct, end_soc_pct, max_power_kw=150.0, battery_cap_kwh=84.0):
        """
        Mo phong duong cong sac phi tuyen CC-CV (Section 5.3):
        - Giai doan CC (Constant Current): Sac nhanh den 80% SoC voi cong suat tối đa.
        - Giai doan CV (Constant Voltage): Sac cham dan khi SoC > 80%.
        """
        if start_soc_pct >= end_soc_pct:
            return 0.0
            
        time_mins = 0.0
        current_soc = start_soc_pct
        step = 1.0 # Tinh toan tung phan tram SoC
        
        while current_soc < end_soc_pct:
            # Tinh luong dien nang can nap cho 1% SoC
            kwh_needed = (step / 100.0) * battery_cap_kwh
            
            # Tinh cong suat sac thuc te theo CC-CV
            if current_soc < 80.0:
                # CC Phase: Cong suat sac dat tối đa
                power = max_power_kw
            else:
                # CV Phase: Cong suat sac giam dan theo dang phi tuyen
                ratio = (100.0 - current_soc) / 20.0 # ratio ve 0 khi gan day
                power = max_power_kw * (ratio ** 2)
                power = max(5.0, power) # Cong suat toi thieu duy tri dong nho
                
            # Thoi gian de nap 1% SoC (gio -> phut)
            duration_mins = (kwh_needed / power) * 60.0
            time_mins += duration_mins
            current_soc += step
            
        return time_mins

    def calculate_degradation_cost(self, dod_pct, battery_cost_usd=8000.0):
        """
        Tinh toan chi phi hao mon pin theo do sau xa pin Depth of Discharge (DoD) (Section 5.5):
        L_cycle(DoD) = A * e^{-B * DoD}
        """
        dod = dod_pct / 100.0
        if dod <= 0:
            return 0.0
            
        # Cac tham so Arrhenius cho pin Li-ion VinFast
        A = 6000.0
        B = 2.1
        
        # So chu ky toi da dat duoc tai muc DoD nay
        max_cycles = A * np.exp(-B * dod)
        
        # Chi phi khau hao pin tren moi chu ky sac-xa o muc DoD nay
        cycle_cost = battery_cost_usd / max_cycles
        return cycle_cost
