import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# =========================
# 1️⃣ Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity):
        # Pamięć przechowuje doświadczenia w kolejce o stałej długości
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Dodajemy doświadczenie: (s, a, r, s', done)
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Losowy batch do treningu
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Zamiana na numpy array
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.memory)


# =========================
# 2️⃣ Sieć Q
# =========================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 3️⃣ Parametry
# =========================
env = gym.make("CartPole-v1", render_mode=None)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

memory = ReplayBuffer(10000)       # Replay Buffer
batch_size = 64
gamma = 0.99                       # współczynnik dyskontowania
epsilon = 1.0                      # eksploracja
epsilon_decay = 0.995
epsilon_min = 0.01
lr = 0.001
tau = 0.01                         # soft update
episode_rewards = []

# =========================
# 4️⃣ Sieci Q
# =========================
# Główna sieć, którą uczymy
policy_net = DQN(state_size, action_size).to(device)
# Target Network do stabilnych targetów
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)

# =========================
# 5️⃣ Funkcja soft update
# =========================
def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        # θ_target = τ*θ_source + (1-τ)*θ_target
        target_param.data.copy_(tau*source_param.data + (1.0 - tau)*target_param.data)

# =========================
# 6️⃣ Wybór akcji (ε-greedy)
# =========================
def select_action(state, epsilon):
    if random.random() < epsilon:
        # losowa akcja
        return env.action_space.sample()
    else:
        # najlepsza akcja według sieci
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.argmax().item()

# =========================
# 7️⃣ Trening
# =========================
def train():
    global epsilon
    num_episodes = 500
    step_count = 0
    solved_threshold = 475

    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

            # Trening tylko gdy mamy wystarczająco danych i co 4 kroki
            if len(memory) >= 1000 and step_count % 4 == 0:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)

                q_values = policy_net(states)
                q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # DOUBLE DQN
                next_actions = policy_net(next_states).argmax(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states)
                    next_q_value = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    expected_q_value = rewards + gamma * next_q_value * (1 - dones)

                loss = (q_value - expected_q_value).pow(2).mean()

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — ogranicza maksymalną normę gradientów do 1.0
                # Chroni przed "eksplozją gradientów" (duże TD-errory → za duży update wag)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

                soft_update(target_net, policy_net, tau)

        # zmniejszamy epsilon (mniej eksploracji z czasem)
        if epsilon > epsilon_min:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        avg_reward_100 = np.mean(episode_rewards[-100:])

        print(
            f"Episode {episode}, Reward: {total_reward:.1f}, "
            f"Avg100: {avg_reward_100:.1f}, Epsilon: {epsilon:.3f}"
        )

        if len(episode_rewards) >= 100 and avg_reward_100 > solved_threshold:
            torch.save(policy_net.state_dict(), "dqn_cartpole.pth")
            print(
                f"Early stopping: Avg100 = {avg_reward_100:.1f} > {solved_threshold}. "
                "Zapisano najlepszy model do dqn_cartpole.pth"
            )
            break


# =========================
# 8️⃣ Start treningu
# =========================
train()
env.close()
plt.figure(figsize=(10,5))
plt.plot(episode_rewards)

plt.title("DQN Training Progress")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.show()