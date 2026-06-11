"""
qmix.py
=======
QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL
(Rashid et al. 2018 — https://arxiv.org/abs/1803.11485)

Architecture
------------
Individual agents
    Q_i(o_i, a)  —  standard DQN-style network per agent (shared weights optional)

Mixing network
    Input  : [Q_1, ..., Q_m]  (agent Q-values for chosen actions)
             global state s
    Output : Q_tot  (scalar)
    Monotonicity: ∂Q_tot/∂Q_i ≥ 0 enforced by using |W| (abs of weights).

    Q_tot = W2 · ELU( W1 · [Q_i] + b1 ) + b2
    where  W1(s) = |hypernetwork_1(s)|
           W2(s) = |hypernetwork_2(s)|
           b1(s) = hypernetwork_b1(s)   (unbounded bias)
           b2(s) = ELU(hypernetwork_b2(s))  (scalar bias with nonlinearity)

All implemented in pure NumPy.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x >= 0, x, alpha * (np.exp(np.clip(x, -30, 0)) - 1.0))


# ---------------------------------------------------------------------------
# Individual Agent Q-Network
# ---------------------------------------------------------------------------

class QMIXAgent:
    """
    Individual agent Q-network: observation → Q-values for all actions.

    Architecture: Linear → ReLU → Linear → Q(o, a) for all a.

    Parameters
    ----------
    obs_dim    : int  Observation dimension for this agent.
    action_dim : int  Number of discrete actions.
    hidden_dim : int  Width of the hidden layer.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        rng = np.random.default_rng(seed=42)
        self.W1 = rng.normal(0, np.sqrt(2.0 / obs_dim),
                             (hidden_dim, obs_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim),
                             (action_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(action_dim, dtype=np.float32)

        # Collect all parameters as a flat vector for gradient-based updates
        self._param_shapes = [self.W1.shape, self.b1.shape,
                               self.W2.shape, self.b2.shape]

    # ------------------------------------------------------------------

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)

        Returns
        -------
        q_values : np.ndarray, shape (action_dim,)
        """
        obs = np.asarray(obs, dtype=np.float32).ravel()
        h = _relu(self.W1 @ obs + self.b1)
        q = self.W2 @ h + self.b2
        return q

    def get_action(self, obs: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        obs     : np.ndarray, shape (obs_dim,)
        epsilon : float  Exploration probability.

        Returns
        -------
        action : int
        """
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.action_dim))
        q_values = self.forward(obs)
        return int(np.argmax(q_values))

    # ------------------------------------------------------------------
    # Parameter helpers (used by trainer for finite-diff gradient)
    # ------------------------------------------------------------------

    def get_params(self) -> np.ndarray:
        """Return all parameters as a flat 1-D array."""
        return np.concatenate([p.ravel() for p in
                               [self.W1, self.b1, self.W2, self.b2]])

    def set_params(self, flat_params: np.ndarray):
        """Set all parameters from a flat 1-D array."""
        idx = 0
        for attr, shape in zip(['W1', 'b1', 'W2', 'b2'], self._param_shapes):
            size = int(np.prod(shape))
            setattr(self, attr, flat_params[idx: idx + size].reshape(shape).astype(np.float32))
            idx += size


# ---------------------------------------------------------------------------
# QMIX Mixing Network
# ---------------------------------------------------------------------------

class QMIXMixer:
    """
    Mixing network: [Q_1, ..., Q_m] + global_state → Q_tot.

    Hypernetworks generate the weights and biases of a 2-layer monotone mixer:
        W1(s) = |hyper1(s)|   shape: (embed_dim, num_agents)
        b1(s) =  hyper_b1(s)  shape: (embed_dim,)
        W2(s) = |hyper2(s)|   shape: (1, embed_dim)
        b2(s) = ELU(hyper_b2(s)) scalar

    Forward:
        Q_tot = W2 · ELU( W1 · Q_i + b1 ) + b2

    Parameters
    ----------
    num_agents : int  Number of agents (= m).
    state_dim  : int  Global state dimension.
    embed_dim  : int  Hidden size of the mixing layer.
    """

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 32):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        rng = np.random.default_rng(seed=7)

        def _init(shape):
            return rng.normal(0, 0.01, shape).astype(np.float32)

        # Hypernetwork 1: state → W1 weights  (embed_dim × num_agents)
        self.hyper1_W = _init((embed_dim * num_agents, state_dim))
        self.hyper1_b = np.zeros(embed_dim * num_agents, dtype=np.float32)

        # Hypernetwork bias 1: state → b1  (embed_dim,)
        self.hyper_b1_W = _init((embed_dim, state_dim))
        self.hyper_b1_b = np.zeros(embed_dim, dtype=np.float32)

        # Hypernetwork 2: state → W2 weights  (embed_dim,)
        self.hyper2_W = _init((embed_dim, state_dim))
        self.hyper2_b = np.zeros(embed_dim, dtype=np.float32)

        # Hypernetwork bias 2: state → scalar b2
        self.hyper_b2_W = _init((1, state_dim))
        self.hyper_b2_b = np.zeros(1, dtype=np.float32)

        # Collect parameter shapes for flat-vector interface
        self._all_params_attrs = [
            'hyper1_W', 'hyper1_b',
            'hyper_b1_W', 'hyper_b1_b',
            'hyper2_W', 'hyper2_b',
            'hyper_b2_W', 'hyper_b2_b',
        ]

    # ------------------------------------------------------------------

    def _hypernetwork(self, state: np.ndarray, W: np.ndarray,
                      output_shape: tuple) -> np.ndarray:
        """
        Generic hypernetwork: linear projection of state.

        Parameters
        ----------
        state        : np.ndarray, shape (state_dim,)
        W            : np.ndarray, shape (out_size, state_dim)
        output_shape : tuple  Desired output shape.

        Returns
        -------
        output : np.ndarray, shape output_shape
        """
        out = W @ state.astype(np.float32)
        return out.reshape(output_shape)

    def forward(self, agent_qs: np.ndarray, global_state: np.ndarray) -> float:
        """
        Compute Q_tot.

        Parameters
        ----------
        agent_qs     : np.ndarray, shape (num_agents,)
                       Q-values of each agent for their chosen action.
        global_state : np.ndarray, shape (state_dim,)

        Returns
        -------
        q_tot : float  Scalar total Q-value.
        """
        agent_qs = np.asarray(agent_qs, dtype=np.float32).ravel()
        global_state = np.asarray(global_state, dtype=np.float32).ravel()

        m = self.num_agents
        e = self.embed_dim

        # -- Layer 1 --
        # W1: (embed_dim, num_agents) — non-negative via abs()
        W1_flat = self._hypernetwork(global_state, self.hyper1_W,
                                     (e * m,)) + self.hyper1_b
        W1 = np.abs(W1_flat).reshape(e, m)

        b1 = self._hypernetwork(global_state, self.hyper_b1_W, (e,)) + self.hyper_b1_b

        hidden = _elu(W1 @ agent_qs + b1)  # (embed_dim,)

        # -- Layer 2 --
        # W2: (1, embed_dim) — non-negative via abs()
        W2_flat = self._hypernetwork(global_state, self.hyper2_W, (e,)) + self.hyper2_b
        W2 = np.abs(W2_flat).reshape(1, e)

        b2_raw = self._hypernetwork(global_state, self.hyper_b2_W, (1,)) + self.hyper_b2_b
        b2 = _elu(b2_raw)  # scalar with ELU nonlinearity

        q_tot = float((W2 @ hidden + b2)[0])
        return q_tot

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def get_params(self) -> np.ndarray:
        return np.concatenate([getattr(self, a).ravel()
                               for a in self._all_params_attrs])

    def set_params(self, flat_params: np.ndarray):
        idx = 0
        for attr in self._all_params_attrs:
            arr = getattr(self, attr)
            size = arr.size
            setattr(self, attr,
                    flat_params[idx: idx + size].reshape(arr.shape).astype(np.float32))
            idx += size


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    num_agents = 3
    obs_dim = 8
    action_dim = 5
    state_dim = 16
    hidden_dim = 64
    embed_dim = 32

    agents = [QMIXAgent(obs_dim, action_dim, hidden_dim) for _ in range(num_agents)]
    mixer = QMIXMixer(num_agents, state_dim, embed_dim)

    rng = np.random.default_rng(0)
    obs_batch = rng.uniform(-1, 1, (num_agents, obs_dim)).astype(np.float32)
    state = rng.uniform(-1, 1, state_dim).astype(np.float32)

    agent_qs = []
    actions = []
    for i, agent in enumerate(agents):
        q = agent.forward(obs_batch[i])
        a = agent.get_action(obs_batch[i], epsilon=0.1)
        agent_qs.append(q[a])
        actions.append(a)
        print(f"Agent {i}: action={a}, Q(o,a)={q[a]:.4f}")

    q_tot = mixer.forward(np.array(agent_qs), state)
    print(f"Q_tot = {q_tot:.4f}")

    # Verify monotonicity: increasing Q_i should increase Q_tot
    agent_qs_up = np.array(agent_qs) + 1.0
    q_tot_up = mixer.forward(agent_qs_up, state)
    assert q_tot_up >= q_tot - 1e-4, f"Monotonicity violated: {q_tot_up:.4f} < {q_tot:.4f}"
    print(f"Monotonicity check passed: Q_tot_up={q_tot_up:.4f} >= Q_tot={q_tot:.4f}")
    print("qmix.py OK")
