import gymnasium as gym
import torch
import torch.nn as nn

# ===== ta sama sieć co w treningu =====
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

# ===== środowisko =====
env = gym.make("CartPole-v1", render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== tworzymy sieć =====
model = DQN(state_size, action_size).to(device)

# ===== wczytujemy wytrenowane wagi =====
model.load_state_dict(
    torch.load("dqn_cartpole.pth", map_location=device, weights_only=True)
)

model.eval()

# ===== symulacja =====
num_episodes = 20

for episode in range(num_episodes):

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = model(state_tensor)

        action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        state = next_state
        total_reward += reward

    print(f"Episode {episode+1}, Reward: {total_reward}")

env.close()