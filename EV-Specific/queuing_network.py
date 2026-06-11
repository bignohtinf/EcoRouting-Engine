import numpy as np
import math

class ChargingQueuingModel:
    """
    Mo phong hang doi M/M/c tai tram sac (Erlang C) theo dung cong thuc Section 5.4.
    """
    def calculate_waiting_time(self, arrival_rate_lambda, service_rate_mu, num_chargers_c):
        """
        arrival_rate_lambda: So xe EV den tram moi gio (lambda)
        service_rate_mu: So xe duoc sac xong moi gio cua 1 tru sac (mu)
        num_chargers_c: So tru sac tai tram (c)
        Tra ve: Prob(Wait) (Xac suat phai cho doi), W_q (Thoi gian cho trung binh - phut)
        """
        c = num_chargers_c
        mu = service_rate_mu
        lam = arrival_rate_lambda
        
        # Tinh traffic intensity: rho = lambda / (c * mu)
        rho = lam / (c * mu)
        if rho >= 1.0:
            # Qua tai tram sac, hang doi vo han
            return 1.0, 999.0
            
        # Tinh Erlang C probability
        # Phan mau so va tu so cua Erlang C
        term1 = ((c * rho) ** c) / (math.factorial(c) * (1.0 - rho))
        
        sum_terms = 0.0
        for n in range(c):
            sum_terms += ((c * rho) ** n) / math.factorial(n)
            
        prob_wait = term1 / (sum_terms + term1)
        
        # Thoi gian cho trung binh trong hang doi: W_q = Prob(Wait) / (mu * (c - lambda / mu))
        # Quy doi ra phut (* 60)
        W_q = (prob_wait / (mu * (c - (lam / mu)))) * 60.0
        
        return prob_wait, W_q


class ChargingFacilityPlanner:
    """
    Bo giai bai toan Facility Location cho Tram sac (Section 5.6):
    Quy hoach vi tri lap tram sac va so luong dau sac toi uu duoi nguon ngan sach cho truoc.
    """
    def plan_charging_network(self, demand_points, candidate_sites, budget=100.0):
        """
        demand_points: Danh sach diem nhu cau (id, lat, lon, load)
        candidate_sites: Danh sach vi tri ung vien lap tram (id, lat, lon, fixed_cost, cap_cost)
        Tra ve: sites_to_open (cac tram mo), total_cost
        """
        # Giai thuat Heuristic tham lam toi uu hoa do phu (Greedy Facility Location)
        opened_sites = []
        remaining_budget = budget
        
        # Tinh toan khoang cach tu cac diem nhu cau den cac tram ung vien
        site_coverage = {cs['id']: [] for cs in candidate_sites}
        for cs in candidate_sites:
            for dp in demand_points:
                dist = haversine_dist(cs['lat'], cs['lon'], dp['lat'], dp['lon'])
                # Neu nam trong ban kinh phuc vu 3km
                if dist <= 3.0:
                    site_coverage[cs['id']].append(dp)
                    
        # Sap xep cac site theo chi phi-hieu qua: So luong demand phuc vu tren 1 unit chi phi mo
        candidate_sites = list(candidate_sites)
        
        while remaining_budget > 0 and candidate_sites:
            best_site = None
            best_score = -1.0
            
            for cs in candidate_sites:
                cost = cs['fixed_cost'] + cs['cap_cost'] * 2 # mac dinh lap 2 tru sac truoc
                if cost <= remaining_budget:
                    # Score = tong demand load bao phu / cost
                    total_load = sum(dp['load'] for dp in site_coverage[cs['id']])
                    score = total_load / cost
                    if score > best_score:
                        best_score = score
                        best_site = cs
                        
            if best_site is None:
                break
                
            opened_sites.append(best_site)
            remaining_budget -= (best_site['fixed_cost'] + best_site['cap_cost'] * 2)
            candidate_sites.remove(best_site)
            
        return opened_sites, budget - remaining_budget

def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2))
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
