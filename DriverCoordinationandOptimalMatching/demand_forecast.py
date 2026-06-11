"""
demand_forecast.py
==================
Spatio-temporal demand forecasting via ConvLSTM (temporal) + GAT (spatial).

Architecture
------------
history_features  (T, num_nodes, input_dim)
    --> ConvLSTMCell (unrolled T steps per node)  --> h_T  (num_nodes, hidden_dim)
    --> GATLayer                                  --> h'   (num_nodes, out_dim)
    --> MLP head                                  --> demand_hat (num_nodes, forecast_horizon)

All math implemented in pure NumPy for portability.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utility activations
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _tanh(x):
    return np.tanh(np.clip(x, -30, 30))


def _relu(x):
    return np.maximum(0.0, x)


def _leaky_relu(x, alpha=0.2):
    return np.where(x >= 0, x, alpha * x)


def _softmax_rows(x):
    """Row-wise softmax for a 2-D array."""
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / (exp_x.sum(axis=-1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------------
# A1. Road-network graph container
# ---------------------------------------------------------------------------

class SpatioTemporalGraph:
    """
    Road network graph: nodes = grid cells, edges = adjacency.

    Parameters
    ----------
    num_nodes         : int   Number of spatial nodes (grid cells).
    node_features_dim : int   Feature dimension per node.
    edge_dim          : int   Edge attribute dimension (stored but not used
                              in the simplified GAT below).
    """

    def __init__(self, num_nodes: int, node_features_dim: int = 8, edge_dim: int = 2):
        self.num_nodes = num_nodes
        self.node_features_dim = node_features_dim
        self.edge_dim = edge_dim
        # Default: no edges
        self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    def set_adjacency(self, adj_matrix: np.ndarray):
        """
        Set the adjacency matrix.

        Parameters
        ----------
        adj_matrix : np.ndarray, shape (num_nodes, num_nodes)
            Binary or weighted adjacency.  Self-loops are added internally.
        """
        adj_matrix = np.asarray(adj_matrix, dtype=np.float32)
        if adj_matrix.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(
                f"Expected adj_matrix shape ({self.num_nodes},{self.num_nodes}), "
                f"got {adj_matrix.shape}"
            )
        self.adj_matrix = adj_matrix

    def get_adjacency_with_self_loops(self) -> np.ndarray:
        """Return adjacency with self-loops added (used in GAT)."""
        return self.adj_matrix + np.eye(self.num_nodes, dtype=np.float32)

    @classmethod
    def grid_graph(cls, rows: int, cols: int, node_features_dim: int = 8) -> "SpatioTemporalGraph":
        """Convenience constructor: build a rows×cols grid graph."""
        num_nodes = rows * cols
        g = cls(num_nodes=num_nodes, node_features_dim=node_features_dim)
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        adj[idx, nr * cols + nc] = 1.0
        g.set_adjacency(adj)
        return g


# ---------------------------------------------------------------------------
# A2. Graph Attention Layer
# ---------------------------------------------------------------------------

class GATLayer:
    """
    Graph Attention Layer (Veličković et al. 2018) — pure NumPy.

    For each node i:
        e_ij  = LeakyReLU(a^T [ W·h_i || W·h_j ])
        α_ij  = softmax_j(e_ij)   (only over neighbours + self)
        h_i'  = σ( Σ_j α_ij · W · h_j )   (averaged over heads)

    Parameters
    ----------
    in_dim    : int   Input feature dimension per node.
    out_dim   : int   Output feature dimension per node.
    num_heads : int   Number of attention heads.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = max(1, out_dim // num_heads)

        rng = np.random.default_rng(seed=42)
        scale_W = np.sqrt(2.0 / (in_dim + self.head_dim))
        scale_a = np.sqrt(2.0 / (2 * self.head_dim))

        # W: (num_heads, head_dim, in_dim)
        self.W = rng.normal(0, scale_W, (num_heads, self.head_dim, in_dim)).astype(np.float32)
        # a: (num_heads, 2 * head_dim)  — attention vector per head
        self.a = rng.normal(0, scale_a, (num_heads, 2 * self.head_dim)).astype(np.float32)

    def forward(self, node_features: np.ndarray, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        node_features : np.ndarray, shape (num_nodes, in_dim)
        adj_matrix    : np.ndarray, shape (num_nodes, num_nodes)
                        Should include self-loops for proper attention.

        Returns
        -------
        h_out : np.ndarray, shape (num_nodes, out_dim)
        """
        node_features = np.asarray(node_features, dtype=np.float32)
        adj_matrix = np.asarray(adj_matrix, dtype=np.float32)
        num_nodes = node_features.shape[0]

        # Add self-loops if not present
        adj_with_self = np.clip(adj_matrix + np.eye(num_nodes, dtype=np.float32), 0, 1)

        head_outputs = []  # collect per-head results

        for h in range(self.num_heads):
            # Linear transform: (num_nodes, head_dim)
            Wh = node_features @ self.W[h].T  # (num_nodes, head_dim)

            # Attention logits e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
            # Broadcast: tile Wh_i (N,1,d) and Wh_j (1,N,d) -> (N,N,2d)
            Wh_i = Wh[:, np.newaxis, :]  # (N, 1, head_dim)
            Wh_j = Wh[np.newaxis, :, :]  # (1, N, head_dim)
            concat = np.concatenate([
                np.broadcast_to(Wh_i, (num_nodes, num_nodes, self.head_dim)),
                np.broadcast_to(Wh_j, (num_nodes, num_nodes, self.head_dim)),
            ], axis=-1)  # (N, N, 2*head_dim)

            e = _leaky_relu(concat @ self.a[h])  # (N, N)

            # Mask out non-neighbours (set to -inf before softmax)
            mask = adj_with_self == 0
            e[mask] = -1e9

            alpha = _softmax_rows(e)  # (N, N)

            # Aggregate: h'_i = σ(Σ_j α_ij · Wh_j)
            h_prime = _relu(alpha @ Wh)  # (N, head_dim)
            head_outputs.append(h_prime)

        # Concatenate heads and project to out_dim
        h_cat = np.concatenate(head_outputs, axis=-1)  # (N, num_heads*head_dim)

        # Linear projection to out_dim (lazy: slice or zero-pad)
        actual_dim = h_cat.shape[-1]
        if actual_dim >= self.out_dim:
            h_out = h_cat[:, :self.out_dim]
        else:
            h_out = np.pad(h_cat, ((0, 0), (0, self.out_dim - actual_dim)))

        return h_out.astype(np.float32)


# ---------------------------------------------------------------------------
# A3. ConvLSTM Cell (spatial conv replaced by linear for tractability)
# ---------------------------------------------------------------------------

class ConvLSTMCell:
    """
    Simplified ConvLSTM cell — pure NumPy.

    Gate equations (using linear layers instead of spatial conv):
        [i, f, g, o] = [σ, σ, tanh, σ](W_x · x_t + W_h · h_{t-1} + b)
        c_t = f ⊙ c_{t-1} + i ⊙ g
        h_t = o ⊙ tanh(c_t)

    Parameters
    ----------
    input_dim  : int   Dimension of input vector x_t.
    hidden_dim : int   Dimension of hidden state h_t.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        rng = np.random.default_rng(seed=0)
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        # Combined weight matrices for all four gates (4 * hidden_dim outputs)
        self.W_x = rng.normal(0, scale, (4 * hidden_dim, input_dim)).astype(np.float32)
        self.W_h = rng.normal(0, scale, (4 * hidden_dim, hidden_dim)).astype(np.float32)
        self.b = np.zeros(4 * hidden_dim, dtype=np.float32)
        # Initialise forget gate bias to 1 for better gradient flow
        self.b[hidden_dim:2 * hidden_dim] = 1.0

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        """
        Parameters
        ----------
        x_t    : np.ndarray, shape (input_dim,)
        h_prev : np.ndarray, shape (hidden_dim,)
        c_prev : np.ndarray, shape (hidden_dim,)

        Returns
        -------
        h_t : np.ndarray, shape (hidden_dim,)
        c_t : np.ndarray, shape (hidden_dim,)
        """
        x_t = np.asarray(x_t, dtype=np.float32).ravel()
        h_prev = np.asarray(h_prev, dtype=np.float32).ravel()
        c_prev = np.asarray(c_prev, dtype=np.float32).ravel()

        gates = self.W_x @ x_t + self.W_h @ h_prev + self.b  # (4*hidden_dim,)

        hd = self.hidden_dim
        i_gate = _sigmoid(gates[0 * hd: 1 * hd])
        f_gate = _sigmoid(gates[1 * hd: 2 * hd])
        g_gate = _tanh(gates[2 * hd: 3 * hd])
        o_gate = _sigmoid(gates[3 * hd: 4 * hd])

        c_t = f_gate * c_prev + i_gate * g_gate
        h_t = o_gate * _tanh(c_t)

        return h_t, c_t

    def zero_state(self):
        """Return (h, c) zero initial states."""
        return (np.zeros(self.hidden_dim, dtype=np.float32),
                np.zeros(self.hidden_dim, dtype=np.float32))


# ---------------------------------------------------------------------------
# A4. Demand Forecaster
# ---------------------------------------------------------------------------

class DemandForecaster:
    """
    Spatio-temporal demand forecaster.

    Architecture
    ------------
    1. ConvLSTMCell  unrolled over T timesteps independently per node.
       After T steps the hidden state h_T (num_nodes, hidden_dim) summarises
       the temporal pattern.
    2. GATLayer      propagates information across the road-network graph
       using attention-weighted neighbour aggregation.
    3. MLP head      maps each node's feature to forecast_horizon scalars.

    Parameters
    ----------
    num_nodes        : int  Number of spatial grid nodes.
    input_dim        : int  Feature dimension per node per timestep.
    hidden_dim       : int  Hidden dimension for ConvLSTM.
    forecast_horizon : int  Number of future time steps to predict.
    """

    def __init__(
        self,
        num_nodes: int = 16,
        input_dim: int = 8,
        hidden_dim: int = 32,
        forecast_horizon: int = 2,
    ):
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon

        # Shared ConvLSTM cell (weights shared across nodes)
        self.lstm_cell = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim)

        # GAT layer: hidden_dim -> gat_out_dim
        self.gat_out_dim = hidden_dim
        self.gat = GATLayer(in_dim=hidden_dim, out_dim=self.gat_out_dim, num_heads=4)

        # MLP head: gat_out_dim -> hidden_dim -> forecast_horizon
        rng = np.random.default_rng(seed=1)
        self.mlp_W1 = rng.normal(0, 0.01, (hidden_dim, self.gat_out_dim)).astype(np.float32)
        self.mlp_b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.mlp_W2 = rng.normal(0, 0.01, (forecast_horizon, hidden_dim)).astype(np.float32)
        self.mlp_b2 = np.zeros(forecast_horizon, dtype=np.float32)

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(self, history_features: np.ndarray, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        history_features : np.ndarray, shape (T, num_nodes, input_dim)
            Historical demand/feature snapshots.
        adj_matrix       : np.ndarray, shape (num_nodes, num_nodes)

        Returns
        -------
        demand_hat : np.ndarray, shape (num_nodes, forecast_horizon)
            Predicted demand per node per future time step.
        """
        history_features = np.asarray(history_features, dtype=np.float32)
        adj_matrix = np.asarray(adj_matrix, dtype=np.float32)

        T = history_features.shape[0]
        num_nodes = history_features.shape[1]

        if num_nodes != self.num_nodes:
            raise ValueError(
                f"history_features has {num_nodes} nodes, "
                f"forecaster expects {self.num_nodes}"
            )

        # ---- Step 1: ConvLSTM temporal encoding ----
        # Process each node independently with a shared LSTM cell
        h_states = np.zeros((num_nodes, self.hidden_dim), dtype=np.float32)
        c_states = np.zeros((num_nodes, self.hidden_dim), dtype=np.float32)

        for t in range(T):
            x_t = history_features[t]  # (num_nodes, input_dim)
            for n in range(num_nodes):
                h_n, c_n = self.lstm_cell.forward(x_t[n], h_states[n], c_states[n])
                h_states[n] = h_n
                c_states[n] = c_n

        # h_states: (num_nodes, hidden_dim)

        # ---- Step 2: GAT spatial aggregation ----
        adj_with_self = np.clip(adj_matrix + np.eye(num_nodes, dtype=np.float32), 0, 1)
        gat_out = self.gat.forward(h_states, adj_with_self)  # (num_nodes, gat_out_dim)

        # ---- Step 3: MLP head ----
        # gat_out -> MLP -> demand_hat
        mlp_h = _relu(gat_out @ self.mlp_W1.T + self.mlp_b1)  # (num_nodes, hidden_dim)
        demand_hat = mlp_h @ self.mlp_W2.T + self.mlp_b2       # (num_nodes, forecast_horizon)

        # Clamp to non-negative demand
        demand_hat = np.maximum(0.0, demand_hat)

        return demand_hat.astype(np.float32)

    def predict(self, history_features: np.ndarray, adj_matrix: np.ndarray) -> np.ndarray:
        """Alias for forward(); public prediction API."""
        return self.forward(history_features, adj_matrix)


# ---------------------------------------------------------------------------
# Quick smoke test (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows, cols = 4, 4
    num_nodes = rows * cols
    T = 6  # history timesteps
    input_dim = 8
    forecast_horizon = 2

    # Build grid graph
    graph = SpatioTemporalGraph.grid_graph(rows=rows, cols=cols, node_features_dim=input_dim)

    # Random history features
    rng = np.random.default_rng(42)
    history = rng.uniform(0, 10, (T, num_nodes, input_dim)).astype(np.float32)

    # Build forecaster
    forecaster = DemandForecaster(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=32,
        forecast_horizon=forecast_horizon,
    )

    demand_hat = forecaster.predict(history, graph.adj_matrix)
    print(f"demand_hat shape: {demand_hat.shape}")   # (16, 2)
    print(f"Sample output (node 0): {demand_hat[0]}")
    print("demand_forecast.py OK")
