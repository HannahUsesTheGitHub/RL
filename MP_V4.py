import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
from gym.envs.classic_control.cartpole import CartPoleEnv
from collections import namedtuple, deque
from multiprocessing import Process, Queue, Value, Manager
import time
import os
import matplotlib.pyplot as plt
import multiprocessing as mp


# NN architecture (for target and policy) (taken from test_scipt.py)
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


# Replay Buffer
Transition = namedtuple('Transition', ('observation', 'action', 'next_observation', 'reward', 'done'))

#standard replay buffer (like in lecture)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    #changed sampling method (compared to lecture) to return the tensors, not just the whole batch
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


#cart pole environment but 
class RandomPoleCartPole(CartPoleEnv):
    def __init__(self, min_length=0.4, max_length=1.8, randomize_each_episode=False):
        super().__init__()
        #min and max pole lengths (interval that will be tested in the end)
        self.min_length = min_length
        self.max_length = max_length
        self.randomize_each_episode = randomize_each_episode
        self.length = random.uniform(self.min_length, self.max_length)
        self.polemass_length = self.masspole * self.length

    #if randomize_each_episode is set to true, the pole length is changed each episode (when the environment is reset)
    def reset(self, **kwargs):
        if self.randomize_each_episode:
            self.length = random.uniform(self.min_length, self.max_length)
            self.polemass_length = self.masspole * self.length
        self.steps = 0
        return super().reset(**kwargs)
    
    def step(self, action):
        self.steps += 1
        obs, reward, terminated, truncated, info = super().step(action)
        if self.steps >= 500:
            truncated = True
        return obs, reward, terminated, truncated, info


#loop for an intividual agent
def worker_process(worker_id, policy_weights_queue, experience_queue, stop_flag, global_epsilon, episode_rewards):
    env = RandomPoleCartPole(min_length=0.4, max_length=1.8, randomize_each_episode=True) #create environment (new pole length each episode)
    local_net = QNetwork(4, 2) #local version of the policy network
    obs, _ = env.reset()
    episode_reward = 0 #count the reward per episode
    step_count = 0 #count the steps per episode

    while not stop_flag.value:
        try:
            while True:
                #synchronize the weights with the other local policies (if possible)
                weights = policy_weights_queue.get_nowait() 
                local_net.load_state_dict(weights)
        except Exception:
            pass

        #choose action - epsilon greedy
        epsilon = global_epsilon.value #update epsilon value
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = local_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = int(torch.argmax(q_vals))

        #execute action, get new obsercation, reward etc.
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        try:
            experience_queue.put_nowait((obs, action, next_obs, reward, done)) #put experience in queue/buffer if possible
        except Exception:
            pass 

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

#plot results
def plot_results(avg_rewards):
    x = [i * 25 for i in range(len(avg_rewards))]
    plt.plot(x, avg_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward (last 25 episodes)")
    plt.title("Parallel DQN")
    plt.grid()
    #plt.show()


#Main parallel DQN loop
def MP_DQN(
    total_steps=20000,
    n_agents=4,
    batch_size=64,
    capacity=10000,
    gamma=0.99,
    learning_rate=1e-3,
    target_update=500
):
    #initialze policy and target network
    state_dim = 4
    action_dim = 2
    policy_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(capacity)

    #set up manager and values for parallel processing
    mp.set_start_method("spawn", force=True)
    manager = Manager()
    stop_flag = manager.Value('b', False)
    global_epsilon = manager.Value('d', 1.0)
    episode_rewards = manager.list()

    policy_weights_queue = mp.Queue()
    experience_queue = mp.Queue(maxsize=5000)

    #start all of the agents
    processes = []
    for agent in range(n_agents):
        p = Process(target=worker_process,
                    args=(agent, policy_weights_queue, experience_queue, stop_flag, global_epsilon, episode_rewards))
        p.start()
        processes.append(p)

    step = 0
    plot_avg_rewards = []
    last_sync = 0
    last_printed_count = 0

    while step < total_steps:
        #in the experience queue is not empty, add those experiences to the buffer
        if not experience_queue.empty():
            obs, act, next_obs, rew, done = experience_queue.get()
            buffer.push(obs, act, next_obs, rew, done)
            step += 1

        #if the buffer is full enough, sample from it and then train the policy network
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

        #update target network if it hasn't been updated for the number of steps specified in target_update
        if step - last_sync >= target_update:
            target_net.load_state_dict(policy_net.state_dict())
            last_sync = step
            weights = policy_net.state_dict()
            for _ in range(n_agents):
                policy_weights_queue.put(weights)

        #global epsilon decay
        if global_epsilon.value > 0.01:
            global_epsilon.value *= 0.9995

        #safe average rewards and print them
        if step % 2000 == 0 and step > 0:
            # last_25 = list(episode_rewards)[-25:] # something wrong here
            # avg_reward = np.mean(last_25) if last_25 else 0
            # plot_avg_rewards.append(avg_reward)
            # print(f"Step {step}/{total_steps}, Avg Reward: {avg_reward:.2f}, Epsilon: {global_epsilon.value:.3f}")
            current_rewards = list(episode_rewards)
            if len(current_rewards) > last_printed_count:
                new_rewards = current_rewards[last_printed_count:]
                avg_reward = np.mean(new_rewards)
                last_printed_count = len(current_rewards)
            else:
                avg_reward = 0
            plot_avg_rewards.append(avg_reward)
            print(f"Step {step}/{total_steps}, Avg Reward (since last print): {avg_reward:.2f}, Epsilon: {global_epsilon.value:.3f}")


        time.sleep(0.001)

    # Stop agents
    stop_flag.value = True
    while not experience_queue.empty():
        try:
            experience_queue.get_nowait()
        except Exception:
            break

    #added print statements becuase i had trouble figuring out if the processes actually ended otherwise
    print("Training finished. Shutting down workers...")
    for p in processes:
        p.join(timeout=5)
    print("All processes joined successfully.")
    return plot_avg_rewards, policy_net

# Run 
if __name__ == "__main__":
    # avg_rewards, network = MP_DQN(
    #     total_steps=20000,
    #     n_agents=4,
    #     batch_size=64,
    #     capacity=10000,
    #     learning_rate=1e-3
    # )

    # os.makedirs("weights", exist_ok=True)
    # model_path = "weights/MP_DQN_random_pole2.pth"
    # torch.save(network.state_dict(), model_path)

    # plot_results(avg_rewards)

    #hyperparameter tuning
    total_steps_list = [20000, 40000, 60000]
    n_agents_list = [4, 8, 12]
    batch_size_list = [32, 64, 128]
    capacity_list = [5000, 10000, 15000]
    learning_rate_list = [1e-3, 3e-4, 1e-4]

    best_rewards = []
    best_networks = []
    for i in range(10):
        best_network = None
        best_reward = 0
        for steps in total_steps_list:
            for n_agents in n_agents_list:
                for batch_size in batch_size_list:
                    for capacity in capacity_list:
                        for learning_rate in learning_rate_list:
                            print()
                            print(f'TRAINING: {steps}steps_{n_agents}agents_{batch_size}batch_{capacity}cap_{learning_rate}lr')
                            print()
                            avg_rewards, network = MP_DQN(steps, n_agents, batch_size, capacity, learning_rate)

                            if avg_rewards[-1] > best_reward:
                                best_reward = avg_rewards[-1]
                                best_network = f'weights/tuning/{i}/DQN_{steps}steps_{n_agents}agents_{batch_size}batch_{capacity}cap_{learning_rate}lr.pth'

                            os.makedirs(f"weights/tuning/{i}", exist_ok=True)
                            model_path = f"weights/tuning/{i}/DQN_{steps}steps_{n_agents}agents_{batch_size}batch_{capacity}cap_{learning_rate}lr.pth"
                            torch.save(network.state_dict(), model_path)
                            plot_results(avg_rewards)
                            fig = plt.gcf()
                            fig.savefig(f"weights/tuning/{i}/DQN_{steps}steps_{n_agents}agents_{batch_size}batch_{capacity}cap_{learning_rate}lr.png", dpi=300, bbox_inches='tight')
        
        print(f'The best network ended with a reward of {best_reward} and the path {best_network}')
        best_networks.append(best_network)
        best_rewards.append(best_reward)
    
    for reward, network in zip(best_networks, best_rewards):
        print(reward, network)
