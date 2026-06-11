import numpy as np

class DriverFairnessManager:
    """
    Quan ly tinh cong bang cho doi tai xe (Driver Fairness).
    Trien khai cac thuoc do: Gini Coefficient, Lexicographic Max-Min, va Nash Bargaining.
    """
    def __init__(self, target_income_bar=100.0):
        self.target_income_bar = target_income_bar

    def calculate_gini(self, incomes):
        """
        Tinh he so Gini cua thu nhap doi tai xe trong ca (Cong thuc Section 4.5):
        Gini = sum_{k, k'} |R_k - R_k'| / (2 * m * sum_k R_k)
        """
        incomes = np.array(incomes, dtype=np.float64)
        m = len(incomes)
        if m <= 1:
            return 0.0
            
        sum_income = np.sum(incomes)
        if sum_income == 0:
            return 0.0
            
        # Tinh tong chenh lech tuyet doi giua tat ca cac cap tai xe
        diff_matrix = np.abs(incomes[:, np.newaxis] - incomes[np.newaxis, :])
        gini = np.sum(diff_matrix) / (2 * m * sum_income)
        return gini

    def get_fairness_penalties(self, current_incomes):
        """
        Hinh thuc hoa Lexicographic Max-Min:
        fairness(d_k) = max(0, avg_R - R_k^current)
        Tuyen chon uu tien cho cac tai xe dang co thu nhap thap hon trung binh ca.
        """
        current_incomes = np.array(current_incomes, dtype=np.float64)
        if len(current_incomes) == 0:
            return []
            
        avg_income = np.mean(current_incomes)
        
        # Phat lon hon doi voi tai xe dang kiem duoc it tien hon
        penalties = []
        for inc in current_incomes:
            penalty = max(0.0, avg_income - inc)
            penalties.append(penalty)
            
        return np.array(penalties)

    def calculate_nash_product(self, incomes, reference_incomes=None):
        """
        Nash Bargaining Solution:
        Max Tich_{k} (R_k - R_k^0)
        """
        incomes = np.array(incomes, dtype=np.float64)
        m = len(incomes)
        if reference_incomes is None:
            reference_incomes = np.zeros(m)
            
        # Tinh hieu so thu nhap so voi muc tham chieu
        surplus = incomes - reference_incomes
        # Gia tri am nghia la tai xe kiem it hon tham chieu, lay max voi 0
        surplus = np.maximum(0.0, surplus)
        
        # Tich so Nash
        return np.prod(surplus)
