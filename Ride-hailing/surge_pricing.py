import numpy as np

class SpatioTemporalSurger:
    """
    Toi uu hoa gia cuoc dong (Dynamic Surge Pricing) va can bang cung cau thoi gian thuc.
    """
    def __init__(self, kappa=1.2, base_fare=30.0):
        self.kappa = kappa        # He so dieu chinh muc do nhay cam cua surge
        self.base_fare = base_fare  # Gia cuoc goc (nghin VND)

    def calculate_surge_multiplier(self, predicted_demand, available_supply):
        """
        Tinh he so surge theo o luoi va thoi gian (Section 6.5):
        rho = 1 + kappa * max(0, (D_hat - S_hat) / S_hat)
        """
        if available_supply <= 0:
            # Thieu cung tram trong, set muc surge toi da (cap)
            return 3.0
            
        excess_ratio = (predicted_demand - available_supply) / available_supply
        surge_mult = 1.0 + self.kappa * max(0.0, excess_ratio)
        
        # Gioi han nguong surge tren de khach khong bo di (cap o muc 2.8)
        return min(2.8, surge_mult)

    def calculate_market_clearing_price(self, D0, S0, epsilon_D=-0.8, epsilon_S=0.4):
        """
        Tinh gia cuoc can bang thi truong (Market Clearing Price - Section 6.5.1):
        P* = (D0 / S0) ^ (1 / (epsilon_S - epsilon_D))
        epsilon_D: Do co gian cau theo gia (< 0, thuong la -0.8)
        epsilon_S: Do co gian cung theo gia (> 0, thuong la 0.4)
        """
        if S0 <= 0 or D0 <= 0:
            return self.base_fare
            
        power = 1.0 / (epsilon_S - epsilon_D)
        ratio = D0 / S0
        opt_price_mult = (ratio) ** power
        
        clearing_price = self.base_fare * opt_price_mult
        return clearing_price
