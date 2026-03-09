# DQN CartPole

Implementacja agenta Double DQN w PyTorch dla środowiska `CartPole-v1` (Gymnasium).

## Struktura projektu

| Plik | Opis |
|---|---|
| `dqn_cartpole.py` | Główny skrypt treningowy: replay buffer, polityka epsilon-greedy, Double DQN, soft update target network |
| `play_cartpole.py` | Odtwarzanie wytrenowanego modelu z wizualizacją (`render_mode="human"`) |
| `cartpole_test.py` | Krótki test środowiska na losowych akcjach |
| `dqn_model.py` | Przykładowa definicja architektury sieci DQN |
| `dqn_cartpole.pth` | Zapisane wagi modelu (checkpoint po treningu) |

## Wymagania

- Python 3.11
- PyTorch 2.5.1 (`+cu121` w aktualnym środowisku)
- Gymnasium 1.2.3
- NumPy 2.3.5
- Matplotlib 3.10.8

## Uruchomienie środowiska (Windows PowerShell)

```powershell
./rlenv/Scripts/Activate.ps1
```

Jeśli instalujesz zależności ręcznie:

```powershell
pip install torch torchvision torchaudio gymnasium matplotlib numpy
```

## Jak uruchomić

Trening agenta:

```bash
python dqn_cartpole.py
```

Odtwarzanie wytrenowanego modelu (20 epizodów, okno środowiska):

```bash
python play_cartpole.py
```

Szybki test środowiska (losowe akcje):

```bash
python cartpole_test.py
```

## Domyślne parametry treningu

Aktualne wartości w `dqn_cartpole.py`:

- liczba epizodów: `500`
- replay buffer: `10000`
- batch size: `64`
- gamma: `0.99`
- learning rate: `0.001`
- epsilon: `1.0 -> 0.01` (decay `0.995`)
- soft update: `tau = 0.01`
- uczenie co `4` kroki środowiska (po zapełnieniu bufora min. `1000` próbek)
- gradient clipping: max norm `1.0`
- early stopping: średnia nagroda z ostatnich 100 epizodów > `400`

Po spełnieniu warunku early stopping model jest zapisywany do `dqn_cartpole.pth`.

## Wyniki

- logi w konsoli per epizod: `Reward`, `Avg100`, `Epsilon`
- wykres postępu treningu (Matplotlib) po zakończeniu
- aktualizacja pliku checkpointu: `dqn_cartpole.pth`

## Uwagi

- `play_cartpole.py` wymaga istniejącego pliku `dqn_cartpole.pth`.
- Jeśli renderowanie nie działa, upewnij się, że środowisko ma dostęp do okna GUI (lokalna sesja desktopowa).
