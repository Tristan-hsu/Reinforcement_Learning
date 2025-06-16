# Reinforcement Learning Snake Game

This project implements a Snake game trained via Deep Q-Network (DQN) reinforcement learning.

## Included Files

- `snake_game.py`: Play Snake manually using a simple `tkinter` interface.
- `snake_game_env.py`: Environment wrapper that exposes the game state, rewards and step function for RL agents.
- `model.py`: Implementation of the DQN agent.
- `snake_game_ai_agent.py`: Visual interface for training and testing the agent. Models can be saved and loaded from here.

## Running the Project

1. **Play manually**

   ```bash
   python snake_game.py
   ```

2. **Launch the training interface**

   ```bash
   python snake_game_ai_agent.py
   ```

   The interface lets you tweak environment and model parameters, start or pause training and run tests.

## Installation

Install Python 3 (3.8 or above is recommended) and the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Depending on your OS you may also need the `tk` package for `tkinter` support.

## Project Structure

```
Reinforcement_Learning/
├── model.py               # DQN agent
├── snake_game.py          # Manual game
├── snake_game_ai_agent.py # Training GUI
├── snake_game_env.py      # RL environment
└── README.md
```

The code is intended for learning and experimentation—you are encouraged to modify and extend it as needed.
