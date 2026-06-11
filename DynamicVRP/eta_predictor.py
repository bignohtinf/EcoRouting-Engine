import numpy as np

class STGNNETAPredictor:
    """
    Mo phong mang do thi ST-GNN cho du bao ETA (Message Passing) 
    ket hop Quantile Loss de du doan cac truong hop kẹt xe nang ne (Heavy-tailed congestion).
    """
    def __init__(self, node_dim=8, edge_dim=2, time_dim=4):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        
        # Khoi tao ma tran trong so cho cac tang Message Passing va MLP output
        self.W_msg = np.random.randn(node_dim * 2 + edge_dim, node_dim) * 0.1
        self.W_node = np.random.randn(node_dim * 2, node_dim) * 0.1
        
        # MLP de ra 3 phan vi: q0.1 (Som nhat), q0.5 (Trung vi), q0.9 (Tre nhat - bao thu)
        self.W_out = np.random.randn(node_dim * 2 + edge_dim + time_dim, 3) * 0.1

    def _message_passing(self, node_features, adj_matrix, edge_features):
        """
        Message Passing theo dung cong thuc duoc neu tai Section 3.3:
        m_{ij} = phi(h_i, h_j, e_{ij})
        h_i^{new} = psi(h_i, sum(m_{ij}))
        """
        num_nodes = len(node_features)
        new_features = np.zeros_like(node_features)
        
        for i in range(num_nodes):
            m_agg = np.zeros(self.node_dim)
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    # Hop nhat feature de lam tin nhan: [h_i || h_j || e_{ij}]
                    e_ij = edge_features[i, j]
                    concat_feat = np.concatenate([node_features[i], node_features[j], e_ij])
                    
                    # Truyen thong tin phi(concat_feat)
                    msg = np.tanh(np.dot(concat_feat, self.W_msg))
                    m_agg += msg
                    
            # Cap nhat feature cua node i bang ham psi: [h_i || m_agg]
            concat_update = np.concatenate([node_features[i], m_agg])
            new_features[i] = np.tanh(np.dot(concat_update, self.W_node))
            
        return new_features

    def predict_eta(self, u_idx, v_idx, req_time, node_features, adj_matrix, edge_features):
        """
        Du doan thoi gian di chuyen giua nut u va nut v:
        tau_hat = MLP([h_u || h_v || e_{uv} || t_emb])
        """
        # Chay 2 tang Message Passing
        h = self._message_passing(node_features, adj_matrix, edge_features)
        h = self._message_passing(h, adj_matrix, edge_features)
        
        # Lay dac trung nut u, v, canh uv
        h_u = h[u_idx]
        h_v = h[v_idx]
        e_uv = edge_features[u_idx, v_idx]
        
        # Time embedding don gian hoa
        t_emb = np.array([np.sin(req_time), np.cos(req_time), req_time / 60.0, 1.0])
        
        # Concat: [h_u || h_v || e_uv || t_emb]
        mlp_input = np.concatenate([h_u, h_v, e_uv, t_emb])
        
        # Du doan cac phan vi (som, trung binh, tre)
        out = np.dot(mlp_input, self.W_out)
        
        # Dam bao cac phan vi luon duong va tang dan (monotonicity)
        q10 = max(0.5, out[0])
        q50 = q10 + max(0.5, out[1])
        q90 = q50 + max(0.5, out[2])
        
        return np.array([q10, q50, q90])

    def compute_losses(self, predictions, actual_eta):
        """
        Tinh toan Huber Loss va Quantile Loss dung cong thuc Section 3.3.
        predictions: array chua [q10, q50, q90]
        actual_eta: float thoi gian thuc te
        """
        # 1. Huber Loss (cho q50 lam ETA chinh)
        diff = actual_eta - predictions[1]
        delta = 2.0 # nguong Huber
        if abs(diff) <= delta:
            huber_loss = 0.5 * (diff ** 2)
        else:
            huber_loss = delta * (abs(diff) - 0.5 * delta)
            
        # 2. Quantile Loss cho {0.1, 0.5, 0.9}
        quantiles = [0.1, 0.5, 0.9]
        q_loss = 0.0
        for i, q in enumerate(quantiles):
            error = actual_eta - predictions[i]
            loss_i = max(q * error, (q - 1) * error)
            q_loss += loss_i
            
        return huber_loss, q_loss
