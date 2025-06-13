import random
import numpy as np


class SnakeEnvironment:
    """貪吃蛇遊戲環境"""
    def __init__(self, board_size=10,reward=10,punishment=-100,deadline_punishment=10):
        self.board_size = board_size
        self.reward = reward
        self.punishment = punishment
        self.deadline_punishment= deadline_punishment
        self.reset()
    
    def reset(self):
        """重置遊戲環境"""
        self.snake = [(self.board_size//2, self.board_size//2)]
        self.direction = 0  # 0:上, 1:右, 2:下, 3:左
        self.food = self._generate_food()
        self.score = 0
        self.steps = 0
        self.max_steps = 200  # 防止無限循環
        return self._get_state()
    
    def _generate_food(self):
        """生成食物位置"""
        while True:
            food = (random.randint(0, self.board_size-1), 
                   random.randint(0, self.board_size-1))
            if food not in self.snake:
                return food
    
    def _get_state(self):
        """獲取當前狀態"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # 方向向量
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 上右下左
        dx, dy = directions[self.direction]
        
        # 檢查危險（前方、左方、右方）
        danger_straight = self._is_collision(head_x + dx, head_y + dy)
        danger_right = self._is_collision(head_x + directions[(self.direction + 1) % 4][0], 
                                         head_y + directions[(self.direction + 1) % 4][1])
        danger_left = self._is_collision(head_x + directions[(self.direction - 1) % 4][0], 
                                        head_y + directions[(self.direction - 1) % 4][1])
        
        # 方向狀態
        dir_up = self.direction == 0
        dir_right = self.direction == 1
        dir_down = self.direction == 2
        dir_left = self.direction == 3
        
        # 食物位置相對於蛇頭
        food_up = food_y < head_y
        food_down = food_y > head_y
        food_left = food_x < head_x
        food_right = food_x > head_x
        
        # 距離邊界的距離
        dist_up = head_y
        dist_down = self.board_size - 1 - head_y
        dist_left = head_x
        dist_right = self.board_size - 1 - head_x
        
        state = [
            # 危險檢測
            danger_straight, danger_right, danger_left,
            
            # 移動方向
            dir_up, dir_right, dir_down, dir_left,
            
            # 食物位置
            food_up, food_down, food_left, food_right,
            
            # 距離邊界
            dist_up / self.board_size,
            dist_down / self.board_size,
            dist_left / self.board_size,
            dist_right / self.board_size,
            
            # 蛇的長度
            len(self.snake) / (self.board_size * self.board_size)
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _is_collision(self, x, y):
        """檢查是否碰撞"""
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return True
        if (x, y) in self.snake:
            return True
        return False
    
    def step(self, action):
        """執行動作"""
        # 動作：0:直走, 1:右轉, 2:左轉
        if action == 1:  # 右轉
            self.direction = (self.direction + 1) % 4
        elif action == 2:  # 左轉
            self.direction = (self.direction - 1) % 4
        
        # 移動蛇
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = directions[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)
        
        # 檢查遊戲結束
        done = False
        reward = 0
        
        if self._is_collision(new_head[0], new_head[1]):
            done = True
            reward = self.punishment # 死亡懲罰
        else:
            self.snake.insert(0, new_head)
            
            # 檢查是否吃到食物
            if new_head == self.food:
                reward = self.reward  # 吃到食物獎勵
                self.score += 1
                self.food = self._generate_food()
            else:
                self.snake.pop()
                reward = -0.1  # 小的時間懲罰，鼓勵盡快找到食物
        
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
            reward -= self.deadline_punishment  # 超時懲罰
        
        return self._get_state(), reward, done
