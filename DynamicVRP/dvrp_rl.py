import numpy as np
import random

class DVRPEnv:
    """
    Moi truong MDP (Markov Decision Process) cho bai toan Dinh tuyen Xe dong (Dynamic VRP).
    """
    def __init__(self, drivers, orders, max_steps=20):
        self.drivers = list(drivers)
        self.all_orders = list(orders)
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.current_step = 0
        self.active_orders = []
        self.assigned_orders_count = 0
        self.total_cost = 0.0
        
        # Dat lai trang thai cho cac driver
        self.driver_states = []
        for d in self.drivers:
            self.driver_states.append({
                'id': d.id,
                'lat': d.lat,
                'lon': d.lon,
                'capacity': d.capacity,
                'current_load': 0,
                'route': []
            })
            
        # Chia cac don hang thanh cac don da biet truoc (static) va cac don nocloud (dynamic)
        # 50% đơn hàng đầu tiên biết từ trước, 50% đơn sau xuất hiện ngẫu nhiên theo bước thời gian
        split_idx = len(self.all_orders) // 2
        self.static_orders = self.all_orders[:split_idx]
        self.dynamic_pool = self.all_orders[split_idx:]
        
        self.active_orders.extend(self.static_orders)
        
        # Bieu dien trang thai state vector: [driver_1_lat, driver_1_lon, ..., num_active_orders, time_step]
        return self._get_observation()

    def _get_observation(self):
        obs = []
        for ds in self.driver_states:
            obs.extend([ds['lat'], ds['lon'], ds['current_load'] / ds['capacity']])
        obs.append(len(self.active_orders))
        obs.append(self.current_step / self.max_steps)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """
        action: index cua tai xe duoc chon de phan bo cho don hang dau tien trong danh sach active_orders
        """
        self.current_step += 1
        reward = 0.0
        
        if len(self.active_orders) > 0 and len(self.driver_states) > 0:
            target_order = self.active_orders[0]
            selected_driver_idx = action % len(self.driver_states)
            drv = self.driver_states[selected_driver_idx]
            
            # Kiem tra rang buoc suc chua
            if drv['current_load'] + target_order.demand <= drv['capacity']:
                drv['current_load'] += target_order.demand
                drv['route'].append(target_order.id)
                self.assigned_orders_count += 1
                
                # Reward duong khi hoan thanh gan
                reward += 10.0
                # Cost di chuyen uoc luong (tru vao reward)
                from vrptw_solver import haversine_dist
                dist = haversine_dist(drv['lat'], drv['lon'], target_order.pu_lat, target_order.pu_lon)
                reward -= dist * 1.5
                
                # Cap nhat vi tri tai xe đen diem lay hang
                drv['lat'], drv['lon'] = target_order.pu_lat, target_order.pu_lon
                self.total_cost += dist
                self.active_orders.pop(0)
            else:
                # Phai phat neu vi pham suc chua (capacity violation)
                reward -= 5.0
                
        # Nhan cac don hang moi ngau nhien tu pool dynamic theo thoi gian
        if self.dynamic_pool and random.random() < 0.4:
            new_o = self.dynamic_pool.pop(0)
            self.active_orders.append(new_o)
            
        done = (self.current_step >= self.max_steps) or (not self.active_orders and not self.dynamic_pool)
        obs = self._get_observation()
        
        return obs, reward, done, {}


class PPOAgent:
    """
    Trien khai thuat toan PPO (Proximal Policy Optimization) voi GAE (Generalized Advantage Estimator).
    Thiet ke tuong thich cao voi ca NumPy ma khong can bat buoc PyTorch nhung van ho tro logic GAE chuan.
    """
    def __init__(self, input_dim, action_dim, gamma=0.99, lam=0.95, clip_eps=0.2):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam  # lambda dung cho GAE
        self.clip_eps = clip_eps
        
        # Khoi tao trong so mang neural phan phoi Policy & Value bang NumPy
        self.W_policy = np.random.randn(input_dim, action_dim) * 0.01
        self.W_value = np.random.randn(input_dim, 1) * 0.01

    def select_action(self, obs):
        # Softmax policy
        logits = np.dot(obs, self.W_policy)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        action = np.random.choice(self.action_dim, p=probs)
        return action, probs[action]

    def compute_gae(self, rewards, values, next_values, dones):
        """
        Cong thuc tinh Generalized Advantage Estimator (GAE) da duoc tuyen bo trong Section 3.2.
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            last_gae = advantages[t]
            
        # Chuan hoa GAE de lam vung vang variance
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages

    def train_step(self, obs_batch, action_batch, advantage_batch, old_prob_batch):
        """
        Thuc hien mot buoc cap nhat chinh sach PPO tuong tu clip objective:
        r_t(theta) = prob / old_prob
        L^{PPO} = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]
        """
        # Day la code gia lap cap nhat gradient tuong thich cho PPO
        # Trong thuc te, ta se goi optimizer de cap nhat W_policy va W_value
        for i in range(len(obs_batch)):
            obs = obs_batch[i]
            act = action_batch[i]
            adv = advantage_batch[i]
            old_p = old_prob_batch[i]
            
            # Gradient ascent tren surrogate objective
            logits = np.dot(obs, self.W_policy)
            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)
            new_p = probs[act]
            
            ratio = new_p / (old_p + 1e-8)
            
            # Policy gradient cap nhat
            if 1 - self.clip_eps < ratio < 1 + self.clip_eps:
                grad = obs * adv
                self.W_policy[:, act] += 0.001 * grad
                
            # Value function cap nhat (MSE)
            val_pred = np.dot(obs, self.W_value)[0]
            val_err = (adv + val_pred) - val_pred
            self.W_value += 0.001 * (obs.reshape(-1, 1) * val_err)
