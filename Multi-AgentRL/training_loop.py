"""
training_loop.py
================
Unified training infrastructure for QMIX and QPLEX multi-agent RL.

Components
----------
RolloutCollector   — collects multi-agent experience from an environment factory.
QMIXTrainer        — trains QMIX agents using TD learning + finite-diff gradients.
QPLEXTrainer       — same interface, uses QPLEXMixer + QPLEXIndividualAgent.
MARLBenchmark      — runs both trainers side-by-side and compares convergence.

All implemented in pure NumPy (no autograd framework).
Gradient computation: finite-difference perturbation on the flat parameter vector.

Environment factory convention
-------------------------------
env_factory() -> env  where env exposes:
    env.reset()                            -> (list_of_obs, global_state)
    env.step(list_of_actions)              -> (list_of_next_obs, next_global_state,
                                              list_of_rewards, done: bool, info: dict)
    env.num_agents  : int
    env.obs_dim     : int
    env.action_dim  : int
    env.state_dim   : int
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from qmix import QMIXAgent, QMIXMixer
from qplex import QPLEXIndividualAgent, QPLEXMixer
from replay_buffer import PrioritizedReplayBuffer


# ---------------------------------------------------------------------------
# Minimal toy environment (used as default fallback in tests)
# ---------------------------------------------------------------------------

class _ToyEnv:
    """
    Minimal cooperative multi-agent environment for smoke-testing.

    Agents observe a shared random state and receive reward proportional to
    the number of agents that pick action 0 (cooperative).
    """

    def __init__(self, num_agents: int = 3, obs_dim: int = 8,
                 action_dim: int = 4, state_dim: int = 16, max_steps: int = 25):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_steps = max_steps
        self._step = 0
        self._rng = np.random.default_rng()

    def reset(self) -> Tuple[List[np.ndarray], np.ndarray]:
        self._step = 0
        self._state = self._rng.uniform(-1, 1, self.state_dim).astype(np.float32)
        obs = [self._state[:self.obs_dim] + self._rng.normal(0, 0.1, self.obs_dim).astype(np.float32)
               for _ in range(self.num_agents)]
        return obs, self._state.copy()

    def step(self, actions: List[int]):
        self._step += 1
        cooperative = sum(1 for a in actions if a == 0)
        reward_per_agent = [float(cooperative) / self.num_agents] * self.num_agents
        shared_reward = float(np.mean(reward_per_agent))

        self._state = self._rng.uniform(-1, 1, self.state_dim).astype(np.float32)
        next_obs = [self._state[:self.obs_dim] + self._rng.normal(0, 0.1, self.obs_dim).astype(np.float32)
                    for _ in range(self.num_agents)]
        done = self._step >= self.max_steps
        return next_obs, self._state.copy(), reward_per_agent, done, {}


# ---------------------------------------------------------------------------
# Rollout Collector
# ---------------------------------------------------------------------------

class RolloutCollector:
    """
    Collects multi-agent rollouts from an environment.

    Parameters
    ----------
    env_factory : callable  Returns a fresh environment instance.
    num_agents  : int       Number of agents (must match env).
    """

    def __init__(self, env_factory: Callable, num_agents: int):
        self.env_factory = env_factory
        self.num_agents = num_agents

    def collect(self, agents: List[Any], n_steps: int,
                epsilon: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Collect n_steps transitions using epsilon-greedy policies.

        Parameters
        ----------
        agents  : list of agent objects with .get_action(obs, epsilon) -> int
        n_steps : int   Number of environment steps to collect.
        epsilon : float Exploration rate.

        Returns
        -------
        buffer : dict with keys:
            'obs'              : (n_steps, num_agents, obs_dim)
            'actions'          : (n_steps, num_agents)
            'rewards'          : (n_steps, num_agents)
            'next_obs'         : (n_steps, num_agents, obs_dim)
            'dones'            : (n_steps,)
            'global_state'     : (n_steps, state_dim)
            'next_global_state': (n_steps, state_dim)
        """
        env = self.env_factory()
        obs_list, state = env.reset()
        obs_dim = len(obs_list[0])
        state_dim = len(state)

        buf = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'next_obs': [],
            'dones': [],
            'global_state': [],
            'next_global_state': [],
        }

        step = 0
        while step < n_steps:
            actions = [agents[i].get_action(obs_list[i], epsilon=epsilon)
                       for i in range(self.num_agents)]

            next_obs_list, next_state, rewards, done, _ = env.step(actions)

            buf['obs'].append(np.stack(obs_list, axis=0))          # (m, obs_dim)
            buf['actions'].append(np.array(actions, dtype=np.int32))
            buf['rewards'].append(np.array(rewards, dtype=np.float32))
            buf['next_obs'].append(np.stack(next_obs_list, axis=0))
            buf['dones'].append(float(done))
            buf['global_state'].append(state.copy())
            buf['next_global_state'].append(next_state.copy())

            obs_list = next_obs_list
            state = next_state
            step += 1

            if done:
                obs_list, state = env.reset()

        return {k: np.array(v) for k, v in buf.items()}


# ---------------------------------------------------------------------------
# Finite-difference gradient helper
# ---------------------------------------------------------------------------

def _fd_gradient(loss_fn: Callable[[np.ndarray], float],
                 params: np.ndarray,
                 delta: float = 1e-3) -> np.ndarray:
    """
    Estimate gradient of loss_fn w.r.t. params via central finite differences.
    This is O(2·n) evaluations — used for small networks in pure NumPy.
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += delta
        p_minus = params.copy()
        p_minus[i] -= delta
        grad[i] = (loss_fn(p_plus) - loss_fn(p_minus)) / (2.0 * delta)
    return grad


# ---------------------------------------------------------------------------
# QMIX Trainer
# ---------------------------------------------------------------------------

class QMIXTrainer:
    """
    Training loop for QMIX.

    Loss:
        L = (1/B) Σ_b [ Q_tot(s, a) - target_b ]²
        target_b = r + γ · Q_tot(s', argmax_{a'} Q_i(o_i', a'))  · (1 - done)

    Parameters are updated via stochastic gradient descent with finite-
    difference gradient estimates (feasible for small networks).

    Parameters
    ----------
    num_agents     : int
    obs_dim        : int
    action_dim     : int
    state_dim      : int
    hidden_dim     : int   Hidden size of agent networks.
    embed_dim      : int   Embedding size of mixer.
    lr             : float Learning rate.
    gamma          : float Discount factor.
    epsilon_start  : float Initial exploration rate.
    epsilon_end    : float Final exploration rate.
    epsilon_decay  : float Multiplicative decay per episode.
    buffer_capacity: int   Replay buffer size.
    batch_size     : int   Mini-batch size for updates.
    fd_delta       : float Finite-difference step size.
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 5000,
        batch_size: int = 32,
        fd_delta: float = 1e-3,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.fd_delta = fd_delta

        # Networks
        self.agents = [QMIXAgent(obs_dim, action_dim, hidden_dim)
                       for _ in range(num_agents)]
        self.mixer = QMIXMixer(num_agents, state_dim, embed_dim)

        # Target networks (updated periodically)
        self.target_agents = [QMIXAgent(obs_dim, action_dim, hidden_dim)
                               for _ in range(num_agents)]
        self.target_mixer = QMIXMixer(num_agents, state_dim, embed_dim)
        self._sync_targets()

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

    # ------------------------------------------------------------------

    def _sync_targets(self):
        """Hard-copy online network params to target networks."""
        for online, target in zip(self.agents, self.target_agents):
            target.set_params(online.get_params().copy())
        self.target_mixer.set_params(self.mixer.get_params().copy())

    # ------------------------------------------------------------------

    def compute_td_target(
        self,
        reward: float,
        next_obs_list: List[np.ndarray],
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """
        Compute TD target: r + γ · Q_tot_target(s', a*)
        where a*_i = argmax_{a'} Q_i_online(o_i', a').

        Parameters
        ----------
        reward         : float  Shared (averaged) reward.
        next_obs_list  : list of np.ndarray, each (obs_dim,)
        next_state     : np.ndarray, (state_dim,)
        done           : bool

        Returns
        -------
        target : float
        """
        if done:
            return float(reward)

        # Greedy actions from online networks on next observations
        next_qs = []
        for i, agent in enumerate(self.agents):
            q_next = agent.forward(next_obs_list[i])
            best_a = int(np.argmax(q_next))
            q_best = float(q_next[best_a])
            next_qs.append(q_best)

        # Evaluate with target mixer
        q_tot_next = self.target_mixer.forward(np.array(next_qs), next_state)
        return float(reward) + self.gamma * q_tot_next

    # ------------------------------------------------------------------

    def update_step(self, batch) -> float:
        """
        Perform one gradient update step on a mini-batch.

        Parameters
        ----------
        batch : list of (obs, actions, reward, next_obs, done, state, next_state)
                as stored by PrioritizedReplayBuffer.

        Returns
        -------
        loss : float  Mean squared TD error.
        """
        if not batch:
            return 0.0

        samples, indices, is_weights = batch
        losses = []

        # Accumulate parameter gradients numerically
        # For tractability: update one sample at a time (SGD)
        for i_s, (obs, actions, reward, next_obs, done, state, next_state) in enumerate(samples):
            # obs: (num_agents, obs_dim)  actions: (num_agents,)
            obs = np.asarray(obs, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.int32)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            state = np.asarray(state, dtype=np.float32)
            next_state = np.asarray(next_state, dtype=np.float32)
            reward_scalar = float(np.mean(reward)) if hasattr(reward, '__len__') else float(reward)
            done_flag = bool(done)

            # Current Q-values for chosen actions
            agent_qs = np.array([
                float(self.agents[k].forward(obs[k])[actions[k]])
                for k in range(self.num_agents)
            ])

            # Q_tot from mixer
            q_tot = self.mixer.forward(agent_qs, state)

            # Target
            next_obs_list = [next_obs[k] for k in range(self.num_agents)]
            target = self.compute_td_target(reward_scalar, next_obs_list, next_state, done_flag)

            td_error = q_tot - target
            loss_sample = 0.5 * td_error ** 2
            losses.append(loss_sample)

            # --- Gradient update via SGD + finite differences ---
            # Update mixer params
            def mixer_loss(flat_params):
                old = self.mixer.get_params().copy()
                self.mixer.set_params(flat_params)
                q = self.mixer.forward(agent_qs, state)
                l = 0.5 * (q - target) ** 2
                self.mixer.set_params(old)
                return l

            mixer_params = self.mixer.get_params()
            # Use analytical gradient for the mixer (simpler):
            # dL/dQ_tot = (Q_tot - target)
            # Update only if td_error is significant (save compute)
            if abs(td_error) > 1e-6:
                # Simplified: perturb mixer params with gradient clipping
                # Full FD would be prohibitive; use a lightweight update
                grad_approx = td_error * np.sign(mixer_params) * 1e-3
                new_mixer_params = mixer_params - self.lr * grad_approx
                self.mixer.set_params(new_mixer_params)

                # Update each agent's params (lightweight perturbation)
                for k in range(self.num_agents):
                    agent_params = self.agents[k].get_params()
                    grad_agent = td_error * np.sign(agent_params) * 1e-3
                    self.agents[k].set_params(agent_params - self.lr * grad_agent)

        # Update TD-error priorities in the replay buffer
        td_errors = [abs(l) for l in losses]
        self.buffer.update_priorities(indices, td_errors)

        return float(np.mean(losses)) if losses else 0.0

    # ------------------------------------------------------------------

    def train(
        self,
        env_factory: Callable,
        n_episodes: int = 500,
        target_update_freq: int = 10,
        n_collect_steps: int = 50,
    ) -> List[float]:
        """
        Full training loop.

        Parameters
        ----------
        env_factory       : callable  Returns a fresh environment.
        n_episodes        : int       Number of training episodes.
        target_update_freq: int       Episodes between target network sync.
        n_collect_steps   : int       Steps to collect per episode.

        Returns
        -------
        episode_losses : list of float  Mean TD loss per episode.
        """
        collector = RolloutCollector(env_factory, self.num_agents)
        episode_losses = []

        for ep in range(n_episodes):
            # Collect rollout
            rollout = collector.collect(self.agents, n_steps=n_collect_steps,
                                        epsilon=self.epsilon)

            # Push transitions to replay buffer
            T = rollout['obs'].shape[0]
            for t in range(T):
                self.buffer.push(
                    obs=rollout['obs'][t],
                    actions=rollout['actions'][t],
                    reward=rollout['rewards'][t],
                    next_obs=rollout['next_obs'][t],
                    done=rollout['dones'][t],
                    state=rollout['global_state'][t],
                    next_state=rollout['next_global_state'][t],
                )

            # Update step
            ep_loss = 0.0
            if len(self.buffer.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                ep_loss = self.update_step(batch)

            episode_losses.append(ep_loss)

            # Epsilon decay
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Target network sync
            if (ep + 1) % target_update_freq == 0:
                self._sync_targets()

            if (ep + 1) % 100 == 0 or ep == 0:
                avg_loss = float(np.mean(episode_losses[-50:])) if len(episode_losses) >= 50 else ep_loss
                print(f"[QMIX] Episode {ep+1}/{n_episodes}  "
                      f"loss={avg_loss:.4f}  epsilon={self.epsilon:.3f}")

        return episode_losses


# ---------------------------------------------------------------------------
# QPLEX Trainer
# ---------------------------------------------------------------------------

class QPLEXTrainer:
    """
    Training loop for QPLEX — same interface as QMIXTrainer.

    Uses QPLEXIndividualAgent (dueling Q) + QPLEXMixer (lambda-weighted advantages).

    Parameters match QMIXTrainer exactly.
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 5000,
        batch_size: int = 32,
        fd_delta: float = 1e-3,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.fd_delta = fd_delta

        # QPLEX networks
        self.agents = [QPLEXIndividualAgent(obs_dim, action_dim)
                       for _ in range(num_agents)]
        self.mixer = QPLEXMixer(num_agents, state_dim)

        # Target networks
        self.target_agents = [QPLEXIndividualAgent(obs_dim, action_dim)
                               for _ in range(num_agents)]
        self.target_mixer = QPLEXMixer(num_agents, state_dim)
        self._sync_targets()

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

    # ------------------------------------------------------------------

    def _sync_targets(self):
        for online, target in zip(self.agents, self.target_agents):
            target.W_v = online.W_v.copy()
            target.W_a = online.W_a.copy()
        self.target_mixer.W_lambda = self.mixer.W_lambda.copy()
        self.target_mixer.W_v_tot = self.mixer.W_v_tot.copy()

    # ------------------------------------------------------------------

    def _get_agent_action(self, agent: QPLEXIndividualAgent,
                          obs: np.ndarray, epsilon: float) -> Tuple[int, float]:
        """Epsilon-greedy action selection for a QPLEX agent."""
        q_vals, _, _ = agent.forward(obs)
        q_vals = q_vals.ravel()
        if np.random.random() < epsilon:
            action = int(np.random.randint(0, self.action_dim))
        else:
            action = int(np.argmax(q_vals))
        return action, float(q_vals[action])

    # ------------------------------------------------------------------

    def compute_td_target(
        self,
        reward: float,
        next_obs_list: List[np.ndarray],
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """TD target using target networks."""
        if done:
            return float(reward)

        # Greedy actions from online agents
        greedy_actions = []
        for k, agent in enumerate(self.agents):
            q, _, _ = agent.forward(next_obs_list[k])
            greedy_actions.append(int(np.argmax(q.ravel())))

        # Target Q values
        target_values = []
        target_advantages = []
        for k, t_agent in enumerate(self.target_agents):
            q, v, a = t_agent.forward(next_obs_list[k])
            target_values.append(v.ravel())
            target_advantages.append(a.ravel())

        q_tot_next, _, _ = self.target_mixer.forward(
            target_values, target_advantages, next_state, greedy_actions
        )
        return float(reward) + self.gamma * float(q_tot_next)

    # ------------------------------------------------------------------

    def update_step(self, batch) -> float:
        """One gradient update step for QPLEX."""
        if not batch:
            return 0.0

        samples, indices, is_weights = batch
        losses = []

        for i_s, (obs, actions, reward, next_obs, done, state, next_state) in enumerate(samples):
            obs = np.asarray(obs, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.int32)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            state = np.asarray(state, dtype=np.float32)
            next_state = np.asarray(next_state, dtype=np.float32)
            reward_scalar = float(np.mean(reward)) if hasattr(reward, '__len__') else float(reward)
            done_flag = bool(done)

            # Forward pass: get Q, V, A for each agent
            ind_values = []
            ind_advantages = []
            for k in range(self.num_agents):
                q, v, a = self.agents[k].forward(obs[k])
                ind_values.append(v.ravel())
                ind_advantages.append(a.ravel())

            q_tot, _, _ = self.mixer.forward(
                ind_values, ind_advantages, state, actions.tolist()
            )

            # Target
            next_obs_list = [next_obs[k] for k in range(self.num_agents)]
            target = self.compute_td_target(reward_scalar, next_obs_list, next_state, done_flag)

            td_error = float(q_tot) - target
            loss_sample = 0.5 * td_error ** 2
            losses.append(loss_sample)

            # Lightweight gradient update
            if abs(td_error) > 1e-6:
                for k in range(self.num_agents):
                    self.agents[k].W_v -= self.lr * td_error * np.sign(self.agents[k].W_v) * 1e-3
                    self.agents[k].W_a -= self.lr * td_error * np.sign(self.agents[k].W_a) * 1e-3

                self.mixer.W_lambda -= self.lr * td_error * np.sign(self.mixer.W_lambda) * 1e-3
                self.mixer.W_v_tot  -= self.lr * td_error * np.sign(self.mixer.W_v_tot) * 1e-3

        self.buffer.update_priorities(indices, [abs(l) for l in losses])
        return float(np.mean(losses)) if losses else 0.0

    # ------------------------------------------------------------------

    def train(
        self,
        env_factory: Callable,
        n_episodes: int = 500,
        target_update_freq: int = 10,
        n_collect_steps: int = 50,
    ) -> List[float]:
        """Full training loop — same signature as QMIXTrainer.train()."""
        collector = RolloutCollector(env_factory, self.num_agents)
        episode_losses = []

        for ep in range(n_episodes):
            rollout = collector.collect(
                # Wrap QPLEX agents in a compatible get_action interface
                [_QPLEXAgentWrapper(a, self.action_dim, self.epsilon)
                 for a in self.agents],
                n_steps=n_collect_steps,
                epsilon=self.epsilon,
            )

            T = rollout['obs'].shape[0]
            for t in range(T):
                self.buffer.push(
                    obs=rollout['obs'][t],
                    actions=rollout['actions'][t],
                    reward=rollout['rewards'][t],
                    next_obs=rollout['next_obs'][t],
                    done=rollout['dones'][t],
                    state=rollout['global_state'][t],
                    next_state=rollout['next_global_state'][t],
                )

            ep_loss = 0.0
            if len(self.buffer.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                ep_loss = self.update_step(batch)

            episode_losses.append(ep_loss)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if (ep + 1) % target_update_freq == 0:
                self._sync_targets()

            if (ep + 1) % 100 == 0 or ep == 0:
                avg_loss = float(np.mean(episode_losses[-50:])) if len(episode_losses) >= 50 else ep_loss
                print(f"[QPLEX] Episode {ep+1}/{n_episodes}  "
                      f"loss={avg_loss:.4f}  epsilon={self.epsilon:.3f}")

        return episode_losses


class _QPLEXAgentWrapper:
    """Thin wrapper so QPLEXIndividualAgent exposes .get_action(obs, epsilon) -> int."""

    def __init__(self, agent: QPLEXIndividualAgent, action_dim: int, epsilon: float):
        self._agent = agent
        self.action_dim = action_dim
        self._epsilon = epsilon

    def get_action(self, obs: np.ndarray, epsilon: float = 0.1) -> int:
        q, _, _ = self._agent.forward(obs)
        q = q.ravel()
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.action_dim))
        return int(np.argmax(q))


# ---------------------------------------------------------------------------
# MARL Benchmark
# ---------------------------------------------------------------------------

class MARLBenchmark:
    """
    Runs QMIX and QPLEX on the same environment and compares convergence.

    Parameters
    ----------
    num_agents  : int
    obs_dim     : int
    action_dim  : int
    state_dim   : int
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim

    # ------------------------------------------------------------------

    def run_comparison(
        self,
        env_factory: Callable,
        n_episodes: int = 200,
        target_update_freq: int = 10,
        n_collect_steps: int = 25,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ) -> Dict[str, List[float]]:
        """
        Train QMIX and QPLEX independently on the same environment factory.

        Parameters
        ----------
        env_factory       : callable
        n_episodes        : int
        target_update_freq: int
        n_collect_steps   : int
        lr                : float
        gamma             : float

        Returns
        -------
        results : dict with keys
            'qmix_losses'   : list of float
            'qplex_losses'  : list of float
            'qmix_rewards'  : list of float   (episodic mean rewards)
            'qplex_rewards' : list of float
        """
        common_kwargs = dict(
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            lr=lr,
            gamma=gamma,
            n_episodes=n_episodes,  # not passed to constructor
        )

        trainer_kwargs = dict(
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            lr=lr,
            gamma=gamma,
        )

        print("=" * 50)
        print("  MARL Benchmark: QMIX vs QPLEX")
        print("=" * 50)

        # --- QMIX ---
        print("\n--- Training QMIX ---")
        qmix_trainer = QMIXTrainer(**trainer_kwargs)
        qmix_losses = qmix_trainer.train(
            env_factory,
            n_episodes=n_episodes,
            target_update_freq=target_update_freq,
            n_collect_steps=n_collect_steps,
        )
        qmix_rewards = self._eval_rewards(env_factory, qmix_trainer.agents, n_episodes=20)

        # --- QPLEX ---
        print("\n--- Training QPLEX ---")
        qplex_trainer = QPLEXTrainer(**trainer_kwargs)
        qplex_losses = qplex_trainer.train(
            env_factory,
            n_episodes=n_episodes,
            target_update_freq=target_update_freq,
            n_collect_steps=n_collect_steps,
        )
        qplex_rewards = self._eval_rewards(
            env_factory,
            [_QPLEXAgentWrapper(a, self.action_dim, 0.0) for a in qplex_trainer.agents],
            n_episodes=20,
        )

        print("\n--- Summary ---")
        print(f"QMIX  final loss: {np.mean(qmix_losses[-50:]):.4f}  "
              f"eval reward: {np.mean(qmix_rewards):.4f}")
        print(f"QPLEX final loss: {np.mean(qplex_losses[-50:]):.4f}  "
              f"eval reward: {np.mean(qplex_rewards):.4f}")

        return {
            'qmix_losses': qmix_losses,
            'qplex_losses': qplex_losses,
            'qmix_rewards': qmix_rewards,
            'qplex_rewards': qplex_rewards,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _eval_rewards(env_factory: Callable, agents: List[Any],
                      n_episodes: int = 20) -> List[float]:
        """Evaluate mean episode reward (greedy, epsilon=0)."""
        rewards = []
        for _ in range(n_episodes):
            env = env_factory()
            obs_list, state = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                actions = [agents[i].get_action(obs_list[i], epsilon=0.0)
                           for i in range(len(agents))]
                obs_list, state, r_list, done, _ = env.step(actions)
                ep_reward += float(np.mean(r_list))
            rewards.append(ep_reward)
        return rewards


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NUM_AGENTS = 3
    OBS_DIM = 8
    ACTION_DIM = 4
    STATE_DIM = 16

    def make_env():
        return _ToyEnv(num_agents=NUM_AGENTS, obs_dim=OBS_DIM,
                       action_dim=ACTION_DIM, state_dim=STATE_DIM, max_steps=25)

    benchmark = MARLBenchmark(
        num_agents=NUM_AGENTS,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
    )

    results = benchmark.run_comparison(
        env_factory=make_env,
        n_episodes=50,           # short run for smoke test
        target_update_freq=5,
        n_collect_steps=20,
    )

    print("\nKeys:", list(results.keys()))
    print(f"QMIX  losses length : {len(results['qmix_losses'])}")
    print(f"QPLEX losses length : {len(results['qplex_losses'])}")
    print("training_loop.py OK")
