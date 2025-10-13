import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
import os
from collections import namedtuple
from multiprocessing import Process, Manager
import time
import matplotlib.pyplot as plt

# ------------------------
# Q-Network
# ------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ------------------------
# Replay Memory
# ------------------------
Transition = namedtuple('Transition', ('observation', 'action', 'next_observation', 'reward', 'done'))

class ReplayMemoryMP:
    def __init__(self, capacity, manager):
        self.memory = manager.list()
        self.capacity = capacity

    def push(self, *args):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(list(self.memory), batch_size)

    def __len__(self):
        return len(self.memory)

# ------------------------
# Action selection
# ------------------------
def pick_action(epsilon, policy_net, obs, env):
    if random.random() < epsilon:
        return env.action_space.sample()
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(obs_tensor)
        return int(torch.argmax(q_values).item())

# ------------------------
# Loss
# ------------------------
def compute_loss(current_q, td_target):
    return nn.MSELoss()(current_q, td_target)

# ------------------------
# Sample batch
# ------------------------
def sample_batch(replay_buffer, batch_size):
    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))

    obs = torch.stack([torch.tensor(o, dtype=torch.float32) for o in batch.observation])
    next_obs = torch.stack([torch.tensor(o, dtype=torch.float32) for o in batch.next_observation])
    actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
    dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

    return obs, actions, rewards, next_obs, dones

# ------------------------
# Agent process
# ------------------------
def run_agent(replay_buffer, policy_net, epsilon, stop_flag, episode_rewards):
    pole_length = random.uniform(0.4, 1.8)  # Different for each agent
    env = gym.make('CartPole-v1')
    env.unwrapped.length = pole_length  # Set pole length
    obs, _ = env.reset()
    terminated = truncated = False
    ep_reward = 0

    while not stop_flag.value:
        action = pick_action(epsilon, policy_net, obs, env)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay_buffer.push(obs, action, next_obs, reward, terminated)

        ep_reward += reward
        obs = next_obs

        if terminated or truncated:
            episode_rewards.append(ep_reward)  # store full episode reward
            ep_reward = 0
            obs, _ = env.reset()
        epsilon = max(0.01, 0.995 * epsilon)

        

# ------------------------
# Main DQN
# ------------------------
def MP_DQN(learning_rate=0.005, gamma=0.99, episodes=500, target_update=70,
           epsilon=1.0, capacity=10000, batch_size=32, n_agents=8):

    policy_net = QNetwork(4,2)
    target_net = QNetwork(4,2)
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.share_memory()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    manager = Manager()
    replay_buffer = ReplayMemoryMP(capacity, manager)
    stop_flag = manager.Value('b', False)
    episode_rewards = manager.list()

    # Start agent processes
    processes = []
    for _ in range(n_agents):
        p = Process(target=run_agent, args=(replay_buffer, policy_net, epsilon, stop_flag, episode_rewards))
        p.start()
        processes.append(p)

    time.sleep(1)  # let agents fill some data

    plot_avg_rewards = []
    step_count = 0

    for ep in range(1, episodes+1):
        if len(replay_buffer) < batch_size:
            time.sleep(0.01)
            continue

        obs, actions, rewards, next_obs, dones = sample_batch(replay_buffer, batch_size)

        next_q_values = target_net(next_obs)
        next_q_max = next_q_values.max(1, keepdim=True)[0].detach()
        td_target = rewards + gamma * (1 - dones) * next_q_max

        current_q = policy_net(obs).gather(1, actions)
        loss = compute_loss(current_q, td_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_count += 1
        if step_count % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Average reward over last 25 episodes
        if ep % 25 == 0:
            with manager.Lock():
                last_25 = list(episode_rewards)[-25:]
                avg_reward = np.mean(last_25) if last_25 else 0
            print(f"Episode {ep}, Avg Reward: {avg_reward:.2f}")
            plot_avg_rewards.append(avg_reward)

    stop_flag.value = True
    for p in processes:
        p.join()

    manager.shutdown()
    return plot_avg_rewards, policy_net

# ------------------------
# Plotting
# ------------------------
def plot_results(plot_avg_rewards):
    y_axis = [25*i for i in range(len(plot_avg_rewards))]
    plt.plot(y_axis, plot_avg_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Massive Parallel DQN")
    plt.show()

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    learning_rate=0.001
    gamma=0.99
    n_agents=12
    episodes=2000
    target_update=100
    epsilon=1.0
    capacity=10000
    batch_size=64

    avg_rewards, network = MP_DQN(
        learning_rate, gamma, episodes, target_update,
           epsilon, capacity, batch_size, n_agents
    )
    os.makedirs("weights", exist_ok=True)
    model_path = "weights/MP_DQN2.pth"
    torch.save(network.state_dict(), model_path)

    plot_results(avg_rewards)
