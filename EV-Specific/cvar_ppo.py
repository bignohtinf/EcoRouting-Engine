"""
cvar_ppo.py — CVaR-PPO for EV Range Anxiety (Section 5.6).
"""
import math
import numpy as np


def _relu(x):
    return np.maximum(0.0, x)

def _relu_grad(x):
    return (x > 0).astype(float)

def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


class NumpyMLP:
    def __init__(self, in_dim, hidden_dim, out_dim, lr=3e-4, seed=0):
        rng = np.random.default_rng(seed)
        s1 = math.sqrt(2.0 / in_dim)
        s2 = math.sqrt(2.0 / hidden_dim)
        s3 = math.sqrt(2.0 / hidden_dim)
        self.W1 = rng.normal(0, s1, (hidden_dim, in_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, s2, (hidden_dim, hidden_dim))
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.normal(0, s3, (out_dim, hidden_dim))
        self.b3 = np.zeros(out_dim)
        self.lr = lr
        self._cache = {}

    def forward(self, x):
        h1 = _relu(self.W1 @ x + self.b1)
        h2 = _relu(self.W2 @ h1 + self.b2)
        out = self.W3 @ h2 + self.b3
        self._cache = {'x': x, 'h1': h1, 'h2': h2}
        return out

    def backward(self, grad_out):
        c = self._cache
        h1, h2, x = c['h1'], c['h2'], c['x']
        dW3 = np.outer(grad_out, h2)
        db3 = grad_out.copy()
        dh2 = self.W3.T @ grad_out
        dh2_pre = dh2 * _relu_grad(self.W2 @ h1 + self.b2)
        dW2 = np.outer(dh2_pre, h1)
        db2 = dh2_pre.copy()
        dh1 = self.W2.T @ dh2_pre
        dh1_pre = dh1 * _relu_grad(self.W1 @ x + self.b1)
        dW1 = np.outer(dh1_pre, x)
        db1 = dh1_pre.copy()
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


class CVaRPPOTrainer:
    """
    CVaR-PPO for EV range anxiety (Section 5.6).

    Objective: L_CVaR-PPO(theta) = L_PPO(theta) - kappa * CVaR_alpha[min_{t<=T} e_k(t)]

    Reward shaping: r_t <- r_t - eta * 1[e_k(t) < E_safe] * ((E_safe - e_k(t))/E_max)^2

    CVaR_alpha[X] = min_v { v + 1/(1-alpha) * E[max(X-v, 0)] }  (Rockafellar-Uryasev)
    Applied to minimise the alpha-CVaR of minimum SoC across episode.
    """

    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99,
                 clip_eps=0.2, kappa=0.5, alpha=0.9, e_safe_pct=15.0,
                 eta=2.0, hidden_dim=64, seed=42, n_episodes=300):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.kappa = kappa
        self.alpha = alpha
        self.e_safe_pct = e_safe_pct
        self.eta = eta
        self.actor = NumpyMLP(obs_dim, hidden_dim, action_dim, lr=lr, seed=seed)
        self.critic = NumpyMLP(obs_dim, hidden_dim, 1, lr=lr, seed=seed + 1)
        self._rng = np.random.default_rng(seed)

    def compute_cvar_loss(self, min_soc_trajectory):
        """
        CVaR_alpha of minimum SoC values (Rockafellar-Uryasev).
        Returns float: CVaR value (higher = safer tail).
        """
        X = np.asarray(min_soc_trajectory, dtype=float)
        if len(X) == 0:
            return 0.0
        v_grid = np.linspace(float(X.min()), float(X.max()), 200)
        best = np.inf
        for v in v_grid:
            cvar = v + 1.0 / (1.0 - self.alpha) * float(np.mean(np.maximum(X - v, 0.0)))
            if cvar < best:
                best = cvar
        return float(best)

    def compute_shaped_reward(self, base_reward, current_soc, e_max):
        """
        r_t <- r_t - eta * 1[e_k(t) < E_safe] * ((E_safe - e_k(t)) / E_max)^2
        """
        if current_soc < self.e_safe_pct:
            penalty = self.eta * ((self.e_safe_pct - current_soc) / e_max) ** 2
            return base_reward - penalty
        return base_reward

    def _sample_action(self, obs):
        logits = self.actor.forward(obs)
        probs = _softmax(logits)
        action = int(self._rng.choice(self.action_dim, p=probs))
        return action, float(probs[action])

    def _value(self, obs):
        return float(self.critic.forward(obs)[0])

    def train_episode(self, env):
        """
        Collect one episode and perform a PPO update step.

        Returns dict: {total_reward, ppo_loss, cvar_loss, min_soc, range_violations}
        """
        obs = np.asarray(env.reset(), dtype=float)
        observations, actions, old_log_probs = [], [], []
        rewards_shaped, values, soc_values = [], [], []
        total_reward = 0.0
        range_violations = 0
        done = False

        while not done:
            action, prob = self._sample_action(obs)
            val = self._value(obs)
            result = env.step(action)
            if len(result) == 4:
                next_obs, reward, done, info = result
            else:
                next_obs, reward, done, truncated, info = result
                done = done or truncated
            info = info or {}
            soc = float(info.get('soc', 50.0))
            soc_values.append(soc)
            if soc < self.e_safe_pct:
                range_violations += 1
            shaped_r = self.compute_shaped_reward(float(reward), soc, 100.0)
            observations.append(obs.copy())
            actions.append(action)
            old_log_probs.append(math.log(max(prob, 1e-8)))
            rewards_shaped.append(shaped_r)
            values.append(val)
            total_reward += float(reward)
            obs = np.asarray(next_obs, dtype=float)

        T = len(rewards_shaped)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards_shaped[t] + self.gamma * G
            returns[t] = G
        advantages = returns - np.array(values)
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - advantages.mean()) / adv_std

        total_ppo_loss = 0.0
        for t in range(T):
            obs_t = observations[t]
            a_t = actions[t]
            adv_t = float(advantages[t])
            ret_t = float(returns[t])
            old_lp = old_log_probs[t]
            logits = self.actor.forward(obs_t)
            probs = _softmax(logits)
            log_p = math.log(max(probs[a_t], 1e-8))
            ratio = math.exp(log_p - old_lp)
            clip_ratio = max(1.0 - self.clip_eps, min(1.0 + self.clip_eps, ratio))
            ppo_obj = min(ratio * adv_t, clip_ratio * adv_t)
            total_ppo_loss += ppo_obj
            grad_log_p = np.zeros(self.action_dim)
            grad_log_p[a_t] = 1.0
            grad_log_p -= probs
            scale = min(ratio, clip_ratio) * adv_t
            grad_logits = scale * grad_log_p
            soc_t = soc_values[t] if t < len(soc_values) else self.e_safe_pct
            if soc_t < self.e_safe_pct:
                cvar_grad_scale = self.kappa * (self.e_safe_pct - soc_t) / 100.0
                grad_logits -= cvar_grad_scale * grad_log_p
            self.actor.backward(-grad_logits)
            v_pred = self.critic.forward(obs_t)[0]
            td_err = float(v_pred) - ret_t
            self.critic.backward(np.array([2.0 * td_err]))

        min_soc = float(min(soc_values)) if soc_values else self.e_safe_pct
        cvar_val = self.compute_cvar_loss([min_soc])

        return {
            'total_reward': total_reward,
            'ppo_loss': total_ppo_loss / max(T, 1),
            'cvar_loss': float(self.kappa * cvar_val),
            'min_soc': min_soc,
            'range_violations': range_violations,
        }

    def train(self, env_factory, n_episodes=300):
        """
        Train CVaR-PPO for n_episodes.

        Parameters
        ----------
        env_factory : callable() -> gym-compatible environment
        n_episodes  : int

        Returns list of per-episode metric dicts.
        """
        history = []
        window_min_socs = []
        for ep in range(n_episodes):
            env = env_factory()
            metrics = self.train_episode(env)
            window_min_socs.append(metrics['min_soc'])
            if len(window_min_socs) > 50:
                window_min_socs = window_min_socs[-50:]
            metrics['rolling_cvar'] = self.compute_cvar_loss(window_min_socs)
            metrics['episode'] = ep + 1
            history.append(metrics)
            if (ep + 1) % 50 == 0:
                avg_r = float(np.mean([h['total_reward'] for h in history[-50:]]))
                avg_cvar = float(np.mean([h['rolling_cvar'] for h in history[-50:]]))
                print(f"  Ep {ep+1:4d} | avg_r={avg_r:.3f} | CVaR_min_soc={avg_cvar:.2f}% "
                      f"| range_viol={metrics['range_violations']}")
        return history


class _ToyEVEnv:
    """Minimal toy EV env for self-test."""
    def __init__(self, obs_dim=6, action_dim=3):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._rng = np.random.default_rng(0)

    def reset(self):
        self.soc = float(self._rng.uniform(40, 90))
        self.step_count = 0
        return self._obs()

    def _obs(self):
        return np.array([
            self.soc / 100.0, self.step_count / 20.0,
            float(self._rng.uniform()), float(self._rng.uniform()),
            float(self._rng.uniform()), float(self._rng.uniform()),
        ])

    def step(self, action):
        if action == 0:
            self.soc -= self._rng.uniform(3, 8)
        elif action == 1:
            self.soc += self._rng.uniform(8, 15)
        else:
            self.soc -= self._rng.uniform(0.5, 2)
        self.soc = float(np.clip(self.soc, 0, 100))
        self.step_count += 1
        reward = -abs(self.soc - 60.0) / 100.0
        done = self.step_count >= 20
        return self._obs(), reward, done, {'soc': self.soc}


if __name__ == '__main__':
    trainer = CVaRPPOTrainer(obs_dim=6, action_dim=3)
    history = trainer.train(lambda: _ToyEVEnv(), n_episodes=100)
    f = history[-1]
    print(f"Final: reward={f['total_reward']:.3f}, min_soc={f['min_soc']:.1f}%, "
          f"cvar_loss={f['cvar_loss']:.4f}, range_viol={f['range_violations']}")
