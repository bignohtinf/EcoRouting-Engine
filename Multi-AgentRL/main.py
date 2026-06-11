import time
import os
import random
import numpy as np

# Thiet lap Agg cho matplotlib de chay headless mượt mà
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import cac module noi bo
from qplex import QPLEXIndividualAgent, QPLEXMixer
from replay_buffer import PrioritizedReplayBuffer

def print_banner(title):
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)

def run_qplex_training_simulation():
    print_banner("HE THONG HUAN LUYEN HOP TAC DA TAC TU (MARL QPLEX) - GREEN SM")
    
    random.seed(400)
    np.random.seed(400)
    
    num_agents = 3
    obs_dim = 6
    action_dim = 4 # 4 hanh dong di chuyen / cho khach o luoi
    state_dim = 12 # Global state
    
    # 1. Khoi tao cac agent ca nhan va global QPLEX mixer
    agents = [QPLEXIndividualAgent(obs_dim, action_dim) for _ in range(num_agents)]
    mixer = QPLEXMixer(num_agents, state_dim)
    
    # Khoi tao bo dem trai nghiem PER
    replay_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    print(f"Khoi tao mo hinh QPLEX thanh cong:")
    print(f" - So luong tac tu xe dien (Agents): {num_agents}")
    print(f" - Mạng individual Q-networks va Centralized Mixing Network da san sang")
    
    # 2. Vong lap huan luyen gia lap (Cooperative Grid Matching Game)
    epochs = 50
    batch_size = 8
    
    reward_history = []
    td_error_history = []
    v_tot_history = []
    q_tot_history = []
    
    print("\n--- Bat dau qua trinh huan luyen hop tac quy mo lon (QPLEX) ---")
    print(f"{'Epoch':<8} | {'Avg Reward':<12} | {'Avg TD-Error':<15} | {'Global V_tot':<12} | {'Global Q_tot'}")
    print("-" * 75)
    
    for epoch in range(1, epochs + 1):
        epoch_rewards = []
        epoch_errors = []
        
        # Chay 5 steps trong moi epoch
        for step in range(5):
            # Sinh random observations cho tung agent
            obs_list = [np.random.randn(obs_dim) for _ in range(num_agents)]
            global_state = np.random.randn(state_dim)
            
            # Cac agent chon hanh dong phi tap trung (Decentralized Execution)
            actions = []
            q_values_list = []
            v_list = []
            a_list = []
            
            for i, agent in enumerate(agents):
                q_vals, v, a = agent.forward(obs_list[i])
                # Epsilon-greedy action selection
                if random.random() < 0.1:
                    act = random.randint(0, action_dim - 1)
                else:
                    act = np.argmax(q_vals)
                actions.append(act)
                q_values_list.append(q_vals)
                v_list.append(v[0])
                a_list.append(a)
                
            # Môi truong phan phoi phan thuong hop tac (Cooperative Reward)
            # Gia su phan thuong tang khi cac agent lua chon hanh dong khac nhau (tranh trung o luoi)
            unique_actions = len(set(actions))
            reward = float(unique_actions) * 4.0 - 2.0 # Cang phoi hop khac nhau thuong cang cao
            epoch_rewards.append(reward)
            
            # Sinh next state
            next_obs_list = [np.random.randn(obs_dim) for _ in range(num_agents)]
            next_global_state = np.random.randn(state_dim)
            done = False
            
            # Dua trai nghiem vao bo dem PER
            replay_buffer.push(obs_list, actions, reward, next_obs_list, done, global_state, next_global_state)
            
        # Bat dau train khi du lich su
        if len(replay_buffer.buffer) >= batch_size:
            samples, indices, weights = replay_buffer.sample(batch_size)
            
            batch_td_errors = []
            for sample in samples:
                s_obs, s_act, s_rew, s_next_obs, s_done, s_state, s_next_state = sample
                
                # Tinh toan Q_tot tu QPLEX mixer cho trang thai hien tai
                s_v_list = []
                s_a_list = []
                for i in range(num_agents):
                    _, v, a = agents[i].forward(s_obs[i])
                    s_v_list.append(v[0])
                    s_a_list.append(a)
                    
                q_tot, v_tot, _ = mixer.forward(s_v_list, s_a_list, s_state, s_act)
                
                # Tinh target Q_tot (su dung next state)
                s_next_v_list = []
                s_next_a_list = []
                s_next_act = []
                for i in range(num_agents):
                    next_q, v, a = agents[i].forward(s_next_obs[i])
                    s_next_v_list.append(v[0])
                    s_next_a_list.append(a)
                    s_next_act.append(np.argmax(next_q))
                    
                q_tot_next, _, _ = mixer.forward(s_next_v_list, s_next_a_list, s_next_state, s_next_act)
                target = s_rew + 0.99 * q_tot_next * (1 - s_done)
                
                # TD-error
                td_error = target - q_tot
                batch_td_errors.append(td_error)
                
                # Gradient update gia lap cho model trong batch huan luyen
                for i in range(num_agents):
                    agents[i].W_v += 0.0001 * s_obs[i].reshape(-1, 1) * td_error
                    agents[i].W_a[:, s_act[i]] += 0.0001 * s_obs[i] * td_error
                mixer.W_v_tot += 0.0001 * s_state.reshape(-1, 1) * td_error
                
            # Cap nhat do uu tien trong PER
            replay_buffer.update_priorities(indices, batch_td_errors)
            avg_td = np.mean(np.abs(batch_td_errors))
            epoch_errors.append(avg_td)
            
            # Ghi lai thong tin
            v_tot_history.append(v_tot)
            q_tot_history.append(q_tot)
        else:
            epoch_errors.append(0.0)
            v_tot_history.append(0.0)
            q_tot_history.append(0.0)
            
        avg_reward = np.mean(epoch_rewards)
        avg_error = np.mean(epoch_errors)
        reward_history.append(avg_reward)
        td_error_history.append(avg_error)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:<8} | {avg_reward:<12.2f} | {avg_error:<15.4f} | {v_tot_history[-1]:<12.4f} | {q_tot_history[-1]:.4f}")
            
    print("-" * 75)
    print("Huan luyen hoan tat. Tien hanh xuat do thi bao cao...")
    
    # 3. Ve va xuat bieu do
    # Do thi 1: Reward Convergence
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, epochs + 1), reward_history, color='dodgerblue', linewidth=2, marker='o', markersize=4, label='Cooperative Episode Reward')
    plt.xlabel("Training Epochs", fontsize=11, weight='bold')
    plt.ylabel("Reward Value", fontsize=11, weight='bold')
    plt.title("QPLEX COOPERATIVE REWARD CONVERGENCE", weight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("qplex_cooperative_reward.png", dpi=300)
    plt.close()
    print(f"\n[Success] Da xuat do thi hoi tu reward ra file: {os.path.abspath('qplex_cooperative_reward.png')}")
    
    # Do thi 2: Dueling Factorization Verification
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(q_tot_history)), q_tot_history, color='forestgreen', linewidth=2, label='Global Q_tot (Joint Utility)')
    plt.plot(range(len(v_tot_history)), v_tot_history, color='darkviolet', linestyle='--', linewidth=1.8, label='Global V_tot (State Value)')
    plt.xlabel("Gradient Update Steps", fontsize=11, weight='bold')
    plt.ylabel("Utility Value", fontsize=11, weight='bold')
    plt.title("QPLEX ADVANTAGE FACTORIZATION VERIFICATION (IGM)", weight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("qplex_advantage_factorization.png", dpi=300)
    plt.close()
    print(f"[Success] Da xuat do thi phan ra QPLEX ra file: {os.path.abspath('qplex_advantage_factorization.png')}")

if __name__ == "__main__":
    run_qplex_training_simulation()
