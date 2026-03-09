# DQN CartPole

PyTorch implementation of a Double DQN agent for solving `CartPole-v1` from Gymnasium.

## Project structure

| File | Description |
|---|---|
| `dqn_cartpole.py` | Main training script (replay buffer, epsilon-greedy policy, Double DQN target computation, soft target update) |
| `dqn_model.py` | Standalone DQN architecture example |
| `cartpole_test.py` | Quick random-policy environment check with rendering |
| `dqn_cartpole.pth` | Saved model weights (created/updated during training) |

## Requirements

- Python 3.11
- PyTorch 2.5.1 (`+cu121` in current environment)
- Gymnasium 1.2.3
- NumPy 2.3.5
- Matplotlib 3.10.8

## Environment setup (Windows PowerShell)

```powershell
./rlenv/Scripts/Activate.ps1
```

If you need to install dependencies manually:

```powershell
pip install torch torchvision torchaudio gymnasium matplotlib numpy
```

## Run

Train agent:

```bash
python dqn_cartpole.py
```

Test environment interaction (random actions + `render_mode="human"`):

```bash
python cartpole_test.py
```

## Training behavior

Current defaults in `dqn_cartpole.py`:

- episodes: `500`
- replay buffer size: `10000`
- batch size: `64`
- gamma: `0.99`
- learning rate: `0.001`
- epsilon schedule: `1.0 -> 0.01` with decay `0.995`
- target update: soft update with `tau = 0.01`
- optimization step: every 4 environment steps (after replay warmup)
- gradient clipping: max norm `1.0`

Early stopping is enabled when average reward over the last 100 episodes exceeds `475`. On early stop, model weights are saved to `dqn_cartpole.pth`.

## Outputs

- Console logs per episode (`Reward`, `Avg100`, `Epsilon`)
- Matplotlib training plot shown at the end of training
- Updated checkpoint file: `dqn_cartpole.pth`
