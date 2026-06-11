import random
import numpy as np

class PrioritizedReplayBuffer:
    """
    Bo dem trai nghiem uu tien (Prioritized Experience Replay - PER) cho Multi-Agent RL.
    Luu tru va trich xuat cac mau chuyen dich co do uu tien ty le voi do lon sai so TD-error.
    """
    def __init__(self, capacity=5000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Muc do anh huong cua do uu tien (0: ngau nhien, 1: uu tien tuyet doi)
        self.beta = beta    # He so dieu chinh truong hop Importance Sampling bias correction
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, obs, actions, reward, next_obs, done, state, next_state):
        """
        Them mau chuyen dich moi vao bo dem.
        Do uu tien ban dau luon duoc set o muc lon nhat de dam bao tat ca cac mau deu duoc hoc it nhat 1 lan.
        """
        max_p = max(self.priorities) if self.priorities else 1.0
        
        transition = (obs, actions, reward, next_obs, done, state, next_state)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_p)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_p
            
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """
        Trich mau tu bo dem dua tren xac suat ty le voi do uu tien:
        P(i) = p_i^alpha / sum_k p_k^alpha
        """
        if len(self.buffer) == 0:
            return []
            
        prios = np.array(self.priorities, dtype=np.float32)
        scaled_prios = prios ** self.alpha
        probs = scaled_prios / np.sum(scaled_prios)
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Tinh Importance Sampling weights de hieu chinh thien kien (bias correction)
        # w_i = (N * P(i))^-beta / max_k w_k
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= np.max(weights)
        
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, td_errors):
        """
        Cap nhat do uu tien sau khi lay mau va tinh xong TD-errors trong buoc train.
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6 # Tranh do uu tien bang 0
