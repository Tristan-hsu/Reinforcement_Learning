import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np

class DQNAgent:
    """Deep Q-Network Agent"""
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.batch_size = 32
        self.gamma = gamma
        
        # 神經網路
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # 更新目標網路
        self.update_target_network()
    
    def _build_model(self):
        """建立神經網路模型"""
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model
    
    def update_target_network(self):
        """更新目標網路"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存儲經驗"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """選擇動作"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """經驗回放"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.array([e[0] for e in batch])).float()
        actions = torch.from_numpy(np.array([e[1] for e in batch])).long()
        rewards = torch.from_numpy(np.array([e[2] for e in batch])).float()
        next_states = torch.from_numpy(np.array([e[3] for e in batch])).float()
        dones = torch.from_numpy(np.array([e[4] for e in batch])).bool()
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
