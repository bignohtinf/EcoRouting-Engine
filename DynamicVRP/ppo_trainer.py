"""
ppo_trainer.py -- PPO with Generalized Advantage Estimation (GAE), NumPy only.

Architecture:
  NumpyMLP       : 2-layer MLP (input -> hidden(64) -> hidden(64) -> output), ReLU activations
  PPOActor       : policy network outputting action probabilities via softmax
  PPOCritic      : value network outputting V(s) scalar
  PPOTrainer     : full PPO loop with GAE, numerical gradient (finite differences)
                   for actor and critic, SGD with momentum for parameter updates
"""

import numpy as np
from collections import deque


class NumpyMLP:
    """2-layer MLP: input -> hidden(64) -> hidden(64) -> output, ReLU activation."""

    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        scale3 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.standard_normal((input_dim, hidden_dim)).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.standard_normal((hidden_dim, hidden_dim)).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = rng.standard_normal((hidden_dim, output_dim)).astype(np.float32) * scale3
        self.b3 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=np.float32))
        h1 = self._relu(x @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        return h2 @ self.W3 + self.b3

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = [
            np.asarray(p, dtype=np.float32) for p in params
        ]

    def _relu(self, x):
        return np.maximum(0.0, x)

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + 1e-12)

    def _flat_params(self):
        return np.concatenate([p.ravel() for p in self.get_params()]).astype(np.float32)

    def _set_flat_params(self, flat):
        params = self.get_params()
        idx = 0
        new_params = []
        for p in params:
            size = p.size
            new_params.append(flat[idx: idx + size].reshape(p.shape).astype(np.float32))
            idx += size
        self.set_params(new_params)


class PPOActor(NumpyMLP):
    """Policy network: obs -> action probabilities (softmax output)."""

    def __init__(self, obs_dim, action_dim, hidden_dim=64, seed=42):
        super().__init__(obs_dim, hidden_dim, action_dim, seed=seed)
        self.action_dim = action_dim

    def get_probs(self, obs):
        logits = self.forward(obs)
        return self._softmax(logits)

    def get_action(self, obs):
        """Sample action. Returns (action_idx: int, log_prob: float)."""
        probs = self.get_probs(obs)
        action = int(np.random.choice(len(probs), p=probs))
        log_prob = float(np.log(probs[action] + 1e-12))
        return action, log_prob


class PPOCritic(NumpyMLP):
    """Value network: obs -> V(s) scalar."""

    def __init__(self, obs_dim, hidden_dim=64, seed=43):
        super().__init__(obs_dim, hidden_dim, 1, seed=seed)

    def get_value(self, obs):
        return float(self.forward(obs)[0])


class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))

    def size(self):
        return len(self.rewards)

    def as_arrays(self):
        return (
            np.array(self.obs, dtype=np.float32),
            np.array(self.actions, dtype=np.int32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
        )


class SGDMomentum:
    """SGD with momentum for a NumpyMLP."""

    def __init__(self, model, lr=3e-4, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.velocity = np.zeros_like(model._flat_params(), dtype=np.float32)

    def step(self, grad):
        grad = np.asarray(grad, dtype=np.float32)
        self.velocity = self.momentum * self.velocity - self.lr * grad
        flat = self.model._flat_params()
        self.model._set_flat_params(flat + self.velocity)


class PPOTrainer:
    """
    PPO trainer (NumPy only).
    - GAE for advantage estimation
    - Finite differences numerical gradient for actor (PPO clipped loss)
    - Finite differences numerical gradient for critic (MSE loss)
    - SGD with momentum
    """

    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99,
                 lam_gae=0.95, clip_eps=0.2, epochs=4, batch_size=32,
                 hidden_dim=64, fd_eps=1e-4, seed=42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.lam_gae = lam_gae
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.fd_eps = fd_eps

        self.actor = PPOActor(obs_dim, action_dim, hidden_dim=hidden_dim, seed=seed)
        self.critic = PPOCritic(obs_dim, hidden_dim=hidden_dim, seed=seed + 1)
        self.actor_opt = SGDMomentum(self.actor, lr=lr)
        self.critic_opt = SGDMomentum(self.critic, lr=lr)

    def compute_gae(self, rewards, values, dones, next_value):
        """
        Generalized Advantage Estimation (GAE-lambda).

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = sum_{l=0}^{T-t-1} (gamma * lam_gae)^l * delta_{t+l}
        Normalize: A_t <- (A_t - mean(A)) / (std(A) + eps)

        Returns (advantages, returns) both shape (T,).
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            nv = next_value * (1.0 - dones[t]) if t == T - 1 else values[t + 1] * (1.0 - dones[t])
            delta = rewards[t] + self.gamma * nv - values[t]
            advantages[t] = delta + self.gamma * self.lam_gae * (1.0 - dones[t]) * last_gae
            last_gae = advantages[t]

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Discounted returns as critic targets
        returns = np.zeros(T, dtype=np.float32)
        R = float(next_value)
        for t in reversed(range(T)):
            R = rewards[t] + self.gamma * R * (1.0 - dones[t])
            returns[t] = R

        return advantages, returns

    def ppo_loss(self, obs_batch, act_batch, old_log_probs, advantages):
        """
        PPO clipped surrogate loss (scalar, to be minimized).

        r_t(theta) = pi_theta(a|s) / pi_theta_old(a|s)
        L = -mean( min( r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t ) )
        """
        n = len(obs_batch)
        total = 0.0
        for i in range(n):
            probs = self.actor.get_probs(obs_batch[i])
            new_lp = float(np.log(probs[act_batch[i]] + 1e-12))
            ratio = float(np.exp(new_lp - float(old_log_probs[i])))
            adv = float(advantages[i])
            clipped = float(np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps))
            total += min(ratio * adv, clipped * adv)
        return -total / max(n, 1)

    def _mean_entropy(self, obs_batch):
        total = 0.0
        for obs in obs_batch:
            probs = self.actor.get_probs(obs)
            total -= float(np.sum(probs * np.log(probs + 1e-12)))
        return total / max(len(obs_batch), 1)

    def _actor_numerical_grad(self, obs_batch, act_batch, old_log_probs, advantages):
        """
        Gradient of ppo_loss w.r.t. actor params via central finite differences:
            g_i ~ (L(theta + eps*e_i) - L(theta - eps*e_i)) / (2*eps)
        Uses a mini-batch subsample to keep cost tractable.
        """
        flat0 = self.actor._flat_params().copy()
        n_params = len(flat0)
        grad = np.zeros(n_params, dtype=np.float32)

        n = len(obs_batch)
        if n > self.batch_size:
            idx = np.random.choice(n, self.batch_size, replace=False)
            obs_s, act_s, olp_s, adv_s = obs_batch[idx], act_batch[idx], old_log_probs[idx], advantages[idx]
        else:
            obs_s, act_s, olp_s, adv_s = obs_batch, act_batch, old_log_probs, advantages

        eps = self.fd_eps
        for i in range(n_params):
            p_plus = flat0.copy(); p_plus[i] += eps
            self.actor._set_flat_params(p_plus)
            Lp = self.ppo_loss(obs_s, act_s, olp_s, adv_s)

            p_minus = flat0.copy(); p_minus[i] -= eps
            self.actor._set_flat_params(p_minus)
            Lm = self.ppo_loss(obs_s, act_s, olp_s, adv_s)

            grad[i] = float((Lp - Lm) / (2.0 * eps))

        self.actor._set_flat_params(flat0)
        return grad

    def _critic_grad(self, obs_batch, returns):
        """
        Gradient of MSE loss w.r.t. critic params via central finite differences:
            L_critic = mean( (R_t - V(s_t))^2 )
            g_i ~ (L(theta + eps*e_i) - L(theta - eps*e_i)) / (2*eps)
        """
        flat0 = self.critic._flat_params().copy()
        n_params = len(flat0)
        grad = np.zeros(n_params, dtype=np.float32)

        n = len(obs_batch)
        if n > self.batch_size:
            idx = np.random.choice(n, self.batch_size, replace=False)
            obs_s, ret_s = obs_batch[idx], returns[idx]
        else:
            obs_s, ret_s = obs_batch, returns

        def mse(flat_p):
            self.critic._set_flat_params(flat_p)
            total = 0.0
            for i in range(len(obs_s)):
                v = self.critic.get_value(obs_s[i])
                total += (float(ret_s[i]) - v) ** 2
            return total / max(len(obs_s), 1)

        eps = self.fd_eps
        for i in range(n_params):
            p_plus = flat0.copy(); p_plus[i] += eps
            p_minus = flat0.copy(); p_minus[i] -= eps
            grad[i] = float((mse(p_plus) - mse(p_minus)) / (2.0 * eps))

        self.critic._set_flat_params(flat0)
        return grad

    def update(self, rollout_buffer):
        """
        Perform PPO update over stored rollout.
        Returns dict with keys: actor_loss, critic_loss, entropy.
        """
        obs_arr, act_arr, old_lp_arr, rew_arr, val_arr, done_arr = rollout_buffer.as_arrays()
        next_value = self.critic.get_value(obs_arr[-1]) * (1.0 - float(done_arr[-1]))
        advantages, returns = self.compute_gae(rew_arr, val_arr, done_arr, next_value)

        actor_losses = []
        critic_losses = []

        for _epoch in range(self.epochs):
            actor_grad = self._actor_numerical_grad(obs_arr, act_arr, old_lp_arr, advantages)
            self.actor_opt.step(actor_grad)
            actor_losses.append(self.ppo_loss(obs_arr, act_arr, old_lp_arr, advantages))

            critic_grad = self._critic_grad(obs_arr, returns)
            self.critic_opt.step(critic_grad)
            c_loss = float(np.mean(
                [(float(returns[i]) - self.critic.get_value(obs_arr[i])) ** 2
                 for i in range(len(obs_arr))]
            ))
            critic_losses.append(c_loss)

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": self._mean_entropy(obs_arr),
        }

    def train_episode(self, env):
        """
        Run one full episode, collect rollout, call update.
        Returns dict: total_reward, steps, actor_loss, critic_loss, entropy.
        """
        buffer = RolloutBuffer()
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            obs_arr = np.asarray(obs, dtype=np.float32)
            action, log_prob = self.actor.get_action(obs_arr)
            value = self.critic.get_value(obs_arr)
            next_obs, reward, done, _ = env.step(action)
            buffer.add(obs_arr, action, log_prob, reward, value, float(done))
            total_reward += reward
            steps += 1
            obs = next_obs

        if buffer.size() == 0:
            return {"total_reward": 0.0, "steps": 0,
                    "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        info = self.update(buffer)
        return {"total_reward": total_reward, "steps": steps, **info}

    def train(self, env, n_episodes=200):
        """
        Run n_episodes; return list of per-episode metric dicts.
        Prints progress every 20 episodes.
        """
        history = []
        reward_window = deque(maxlen=20)

        for ep in range(1, n_episodes + 1):
            info = self.train_episode(env)
            history.append(info)
            reward_window.append(info["total_reward"])

            if ep % 20 == 0 or ep == 1:
                avg_r = float(np.mean(reward_window))
                print(
                    f"[PPO] Episode {ep:4d}/{n_episodes} | "
                    f"avg_reward={avg_r:8.3f} | "
                    f"actor_loss={info['actor_loss']:.4f} | "
                    f"critic_loss={info['critic_loss']:.4f} | "
                    f"entropy={info['entropy']:.4f}"
                )

        return history


if __name__ == '__main__':
    from dvrp_rl import DVRPEnv
    from main import generate_mock_vrp_data

    print("=" * 60)
    print("PPO Trainer -- smoke test (5 episodes)")
    print("=" * 60)

    orders, drivers = generate_mock_vrp_data()
    env = DVRPEnv(drivers, orders, max_steps=10)

    # obs_dim matches DVRPEnv._get_observation():
    # len(drivers) * 3 (lat, lon, load_ratio) + 2 (num_active_orders, time_step_ratio)
    obs_dim = len(drivers) * 3 + 2
    action_dim = len(drivers)

    trainer = PPOTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        lam_gae=0.95,
        clip_eps=0.2,
        epochs=2,
        batch_size=8,
        hidden_dim=64,
        fd_eps=1e-4,
    )

    history = trainer.train(env, n_episodes=5)

    print("\nTraining complete. Episode summaries:")
    for i, h in enumerate(history, 1):
        print(
            f"  Ep {i}: reward={h['total_reward']:.2f}, "
            f"steps={h['steps']}, "
            f"actor_loss={h['actor_loss']:.4f}, "
            f"critic_loss={h['critic_loss']:.4f}"
        )
    print("Smoke test passed.")
