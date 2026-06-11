import numpy as np

class QPLEXIndividualAgent:
    """
    Mang hanh vi ca nhan cua tung Agent xe dien trong QPLEX.
    Dung de du doan Individual Utility Q_i = V_i(o_i) + A_i(o_i, a_i).
    """
    def __init__(self, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Trong so cho V_i (State Value) va A_i (Advantage)
        self.W_v = np.random.randn(obs_dim, 1) * 0.01
        self.W_a = np.random.randn(obs_dim, action_dim) * 0.01

    def forward(self, obs):
        """
        Tra ve Q_values cho tat ca cac hanh dong cua Agent.
        """
        V = np.dot(obs, self.W_v)
        A = np.dot(obs, self.W_a)
        
        # Chuan hoa Advantage: A(o, a) = A(o, a) - mean(A(o, a'))
        A_mean = np.mean(A, axis=-1, keepdims=True)
        Q = V + (A - A_mean)
        return Q, V, A - A_mean


class QPLEXMixer:
    """
    Mang tron QPLEX (Dueling Bipartite Value Factorisation Mixer).
    Tinh toan Q_tot = V_tot(s) + sum_i lambda_i(s, a) * A_i(o_i, a_i).
    """
    def __init__(self, num_agents, state_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        
        # Hypernetwork de sinh ra cac he so lambda_i(s, a) > 0
        self.W_lambda = np.random.randn(state_dim, num_agents) * 0.01
        # Hypernetwork de sinh ra global V_tot(s)
        self.W_v_tot = np.random.randn(state_dim, 1) * 0.01

    def forward(self, individual_values, individual_advantages, global_state, selected_actions):
        """
        Factorize Q_tot theo công thuc QPLEX:
        Q_tot = V_tot + sum_i (lambda_i * A_i_selected)
        """
        # 1. Tinh global V_tot
        V_tot = np.dot(global_state, self.W_v_tot)[0]
        
        # 2. Sinh trong so lambda_i > 0 thong qua ham mu (exp) de dam bao tinh nghiem ngat
        lambda_weights = np.exp(np.dot(global_state, self.W_lambda))
        # Chuan hoa lambda
        lambda_weights = lambda_weights / np.sum(lambda_weights)
        
        # 3. Tinh tong loi the gop: A_tot = sum_i lambda_i * A_i[selected_action]
        A_tot = 0.0
        for i in range(self.num_agents):
            a_idx = selected_actions[i]
            a_val = individual_advantages[i][a_idx]
            A_tot += lambda_weights[i] * a_val
            
        Q_tot = V_tot + A_tot
        return Q_tot, V_tot, lambda_weights
