# DQN CartPole

Deep Q-Network (Double DQN) agent solving the **CartPole-v1** environment from Gymnasium.

## Project structure

| File | Description |
|---|---|
| `dqn_cartpole.py` | Training script (replay buffer, ε-greedy, soft target update, Double DQN) |
| `dqn_model.py` | Standalone DQN model definition |
| `cartpole_test.py` | Quick environment test (random policy) |
| `dqn_cartpole.pth` | Pre-trained model weights |

## Requirements

- Python 3.11
- PyTorch 2.5 (CUDA 12.1)
- Gymnasium 1.2
- Matplotlib, NumPy

## Usage

**Train:**

```bash
python dqn_cartpole.py
```

**Test environment:**

```bash
python cartpole_test.py
```
