# Reinforcement Learning Snake Game

本專案實作了一個使用深度強化學習(DQN)訓練的貪吃蛇遊戲，
包含以下幾個部分：

- `snake_game.py`：純粹的貪吃蛇遊戲，使用 `tkinter` 介面手動遊玩。
- `snake_game_env.py`：將遊戲包裝成強化學習環境，提供狀態、獎勵與步驟函式。
- `model.py`：深度 Q 網路 (DQN) 智能體的實作。
- `snake_game_ai_agent.py`：視覺化訓練介面，可調整參數、訓練及測試 AI，並能儲存/載入模型。

## 執行方式

1. 手動遊玩：

```bash
python snake_game.py
```

2. 開啟 AI 訓練介面：

```bash
python snake_game_ai_agent.py
```

介面中可設定環境與模型參數，開始訓練、暫停及測試 AI。

## 需求套件

請先安裝 Python 3 (建議 3.8 以上)，並依照 `requirements.txt` 安裝必要套件：

```bash
pip install -r requirements.txt
```

## 專案結構

```
Reinforcement_Learning/
├── model.py               # DQN 智能體
├── snake_game.py          # 手動玩貪吃蛇
├── snake_game_ai_agent.py # AI 訓練介面
├── snake_game_env.py      # 貪吃蛇強化學習環境
└── README.md
```

本專案僅供學習與研究用途，歡迎依需求修改與擴充。
