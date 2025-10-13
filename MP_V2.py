import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
from collections import namedtuple, deque
from multiprocessing import Process, Queue, Value, Manager
import time
import os
import matplotlib.pyplot as plt
import multiprocessing as mp

# ------------------------
# Q-Network
# ------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ------------------------
# Replay Buffer
# ------------------------
Transition = namedtuple('Transition', ('observation', 'action', 'next_observation', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))
        obs = torch.tensor(np.array(batch.observation), dtype=torch.float32)
        actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
        next_obs = torch.tensor(np.array(batch.next_observation), dtype=torch.float32)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)

# ------------------------
# Worker Agent
# ------------------------
# ------------------------
# Worker Agent
# ------------------------
def worker_process(worker_id, policy_weights_queue, experience_queue, stop_flag, global_epsilon, episode_rewards):
    env = gym.make('CartPole-v1')
    local_net = QNetwork(4, 2)
    obs, _ = env.reset()
    episode_reward = 0
    step_count = 0

    while not stop_flag.value:
        # Sync weights if available (safe against empty queue)
        try:
            while True:
                weights = policy_weights_queue.get_nowait()
                local_net.load_state_dict(weights)
        except Exception:
            pass  # queue.Empty or any sync race

        epsilon = global_epsilon.value
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = local_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = int(torch.argmax(q_vals))

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        try:
            experience_queue.put_nowait((obs, action, next_obs, reward, done))
        except Exception:
            pass  # queue full, skip this step

        episode_reward += reward
        obs = next_obs

        if done:
            episode_rewards.append(episode_reward)
            obs, _ = env.reset()
            episode_reward = 0

        step_count += 1
        if step_count % 1000 == 0:
            time.sleep(0.01)

    env.close()


# ------------------------
# Plot results
# ------------------------
def plot_results(avg_rewards):
    x = [i * 2000 for i in range(len(avg_rewards))]
    plt.plot(x, avg_rewards)
    plt.xlabel("Training steps")
    plt.ylabel("Average Reward (last 25 episodes)")
    plt.title("Parallel DQN on CartPole-v1")
    plt.grid()
    plt.show()

# ------------------------
# Main DQN loop
# ------------------------
# ------------------------
# Main DQN loop
# ------------------------
def MP_DQN(
    total_steps=20000,
    n_agents=4,
    batch_size=64,
    capacity=10000,
    gamma=0.99,
    learning_rate=1e-3,
    target_update=500
):
    state_dim = 4
    action_dim = 2
    policy_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(capacity)

    mp.set_start_method("spawn", force=True)
    manager = Manager()
    stop_flag = manager.Value('b', False)
    global_epsilon = manager.Value('d', 1.0)
    episode_rewards = manager.list()

    policy_weights_queue = mp.Queue()
    experience_queue = mp.Queue(maxsize=5000)

    # Start worker processes
    processes = []
    for wid in range(n_agents):
        p = Process(target=worker_process,
                    args=(wid, policy_weights_queue, experience_queue, stop_flag, global_epsilon, episode_rewards))
        p.start()
        processes.append(p)

    step = 0
    plot_avg_rewards = []
    last_sync = 0

    while step < total_steps:
        if not experience_queue.empty():
            obs, act, next_obs, rew, done = experience_queue.get()
            buffer.push(obs, act, next_obs, rew, done)
            step += 1

        if len(buffer) >= batch_size:
            obs, actions, rewards, next_obs, dones = buffer.sample(batch_size)
            with torch.no_grad():
                next_q = target_net(next_obs).max(1, keepdim=True)[0]
                td_target = rewards + gamma * (1 - dones) * next_q
            current_q = policy_net(obs).gather(1, actions)
            loss = nn.MSELoss()(current_q, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Target net + policy sync
        if step - last_sync >= target_update:
            target_net.load_state_dict(policy_net.state_dict())
            last_sync = step
            weights = policy_net.state_dict()
            for _ in range(n_agents):
                policy_weights_queue.put(weights)

        # Global epsilon decay
        if global_epsilon.value > 0.01:
            global_epsilon.value *= 0.9995

        # Track average rewards
        if step % 2000 == 0 and step > 0:
            last_25 = list(episode_rewards)[-25:]
            avg_reward = np.mean(last_25) if last_25 else 0
            plot_avg_rewards.append(avg_reward)
            print(f"Step {step}/{total_steps}, Avg Reward: {avg_reward:.2f}, Epsilon: {global_epsilon.value:.3f}")
            if avg_reward == 500.0:
                break

        time.sleep(0.001)

    # Stop workers safely
    stop_flag.value = True
    # Drain queue to prevent workers blocking
    while not experience_queue.empty():
        try:
            experience_queue.get_nowait()
        except Exception:
            break

    print("Training finished. Shutting down workers...")
    for p in processes:
        p.join(timeout=5)
    print("All processes joined successfully.")
    return plot_avg_rewards, policy_net


# ------------------------
# Run + Save + Plot
# ------------------------
if __name__ == "__main__":
    avg_rewards, network = MP_DQN(
        total_steps=100000,
        n_agents=4,
        batch_size=64,
        capacity=10000,
        learning_rate=1e-3
    )

    # --- Save model before stopping ---
    os.makedirs("weights", exist_ok=True)
    model_path = "weights/MP_DQN2_v2.pth"
    torch.save(network.state_dict(), model_path)

    plot_results(avg_rewards)

