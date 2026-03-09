import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

state, info = env.reset()

print("Stan początkowy:", state)
print("Liczba możliwych akcji:", env.action_space.n)

for step in range(10):
    action = env.action_space.sample()  # losowa akcja
    next_state, reward, terminated, truncated, info = env.step(action)

    print("\nKrok:", step)
    print("Akcja:", action)
    print("Nowy stan:", next_state)
    print("Nagroda:", reward)

    if terminated or truncated:
        print("Epizod zakończony")
        state, info = env.reset()