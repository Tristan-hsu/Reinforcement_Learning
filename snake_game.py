import tkinter as tk
from tkinter import messagebox
import random
import time

class SnakeGame:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("貪吃蛇遊戲")
        self.window.resizable(False, False)
        
        # 遊戲設定
        self.BOARD_SIZE = 10
        self.CELL_SIZE = 40
        self.CANVAS_SIZE = self.BOARD_SIZE * self.CELL_SIZE
        
        # 遊戲狀態
        self.snake = [(5, 5)]  # 蛇的初始位置 (中間)
        self.direction = (0, -1)  # 初始方向向上 (row, col)
        self.food = self.generate_food()
        self.score = 0
        self.game_running = False
        self.speed = 500  # 初始速度 (毫秒)
        self.min_speed = 100  # 最快速度
        
        self.setup_ui()
        self.bind_keys()
        
    def setup_ui(self):
        """設置用戶界面"""
        # 主框架
        main_frame = tk.Frame(self.window)
        main_frame.pack(padx=10, pady=10)
        
        # 標題
        title = tk.Label(main_frame, text="貪吃蛇遊戲", font=("Arial", 20, "bold"))
        title.pack(pady=5)
        
        # 分數顯示
        self.score_label = tk.Label(main_frame, text=f"分數: {self.score}", font=("Arial", 14))
        self.score_label.pack(pady=5)
        
        # 遊戲畫布
        self.canvas = tk.Canvas(
            main_frame, 
            width=self.CANVAS_SIZE, 
            height=self.CANVAS_SIZE,
            bg="black",
            highlightthickness=2,
            highlightbackground="gray"
        )
        self.canvas.pack(pady=10)
        
        # 控制按鈕框架
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # 開始/暫停按鈕
        self.start_button = tk.Button(
            button_frame, 
            text="開始遊戲", 
            font=("Arial", 12),
            command=self.start_game,
            bg="green",
            fg="white",
            width=10
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # 重新開始按鈕
        self.restart_button = tk.Button(
            button_frame, 
            text="重新開始", 
            font=("Arial", 12),
            command=self.restart_game,
            bg="orange",
            fg="white",
            width=10
        )
        self.restart_button.pack(side=tk.LEFT, padx=5)
        
        # 結束按鈕
        self.quit_button = tk.Button(
            button_frame, 
            text="結束遊戲", 
            font=("Arial", 12),
            command=self.quit_game,
            bg="red",
            fg="white",
            width=10
        )
        self.quit_button.pack(side=tk.LEFT, padx=5)
        
        # 方向控制按鈕
        control_frame = tk.Frame(main_frame)
        control_frame.pack(pady=10)
        
        tk.Label(control_frame, text="方向控制 (或使用方向鍵):", font=("Arial", 10)).pack()
        
        # 上按鈕
        up_btn = tk.Button(control_frame, text="↑", font=("Arial", 12), width=3, command=lambda: self.change_direction(0, -1))
        up_btn.pack()
        
        # 左右按鈕
        lr_frame = tk.Frame(control_frame)
        lr_frame.pack()
        left_btn = tk.Button(lr_frame, text="←", font=("Arial", 12), width=3, command=lambda: self.change_direction(-1, 0))
        left_btn.pack(side=tk.LEFT)
        right_btn = tk.Button(lr_frame, text="→", font=("Arial", 12), width=3, command=lambda: self.change_direction(1, 0))
        right_btn.pack(side=tk.LEFT)
        
        # 下按鈕
        down_btn = tk.Button(control_frame, text="↓", font=("Arial", 12), width=3, command=lambda: self.change_direction(0, 1))
        down_btn.pack()
        
        # 遊戲說明
        instructions = tk.Label(
            main_frame, 
            text="使用方向鍵或按鈕控制蛇的移動\n吃到紅色食物會增長並加分\n碰到牆壁或自己身體會死亡",
            font=("Arial", 9),
            justify=tk.CENTER
        )
        instructions.pack(pady=5)
        
        # 初始化畫面
        self.draw_board()
    
    def bind_keys(self):
        """綁定鍵盤事件"""
        self.window.bind('<Key>', self.on_key_press)
        self.window.focus_set()
    
    def on_key_press(self, event):
        """處理鍵盤按鍵"""
        key = event.keysym
        if key == 'Up':
            self.change_direction(0, -1)
        elif key == 'Down':
            self.change_direction(0, 1)
        elif key == 'Left':
            self.change_direction(-1, 0)
        elif key == 'Right':
            self.change_direction(1, 0)
        elif key == 'space':
            if self.game_running:
                self.pause_game()
            else:
                self.start_game()
    
    def change_direction(self, dx, dy):
        """改變蛇的移動方向"""
        if not self.game_running:
            return
            
        # 防止蛇直接反向移動
        current_dx, current_dy = self.direction
        if (dx, dy) != (-current_dx, -current_dy):
            self.direction = (dx, dy)
    
    def generate_food(self):
        """生成食物位置"""
        while True:
            food_pos = (random.randint(0, self.BOARD_SIZE-1), random.randint(0, self.BOARD_SIZE-1))
            if food_pos not in self.snake:
                return food_pos
    
    def move_snake(self):
        """移動蛇"""
        if not self.game_running:
            return
            
        # 計算新的頭部位置
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        
        # 檢查碰撞
        if self.check_collision(new_head):
            self.game_over()
            return
        
        # 移動蛇
        self.snake.insert(0, new_head)
        
        # 檢查是否吃到食物
        if new_head == self.food:
            self.score += 10
            self.score_label.config(text=f"分數: {self.score}")
            self.food = self.generate_food()
            # 增加速度
            if self.speed > self.min_speed:
                self.speed = max(self.min_speed, self.speed - 20)
        else:
            # 如果沒吃到食物，移除尾巴
            self.snake.pop()
        
        # 重繪畫面
        self.draw_board()
        
        # 設置下一次移動
        self.window.after(self.speed, self.move_snake)
    
    def check_collision(self, pos):
        """檢查碰撞"""
        x, y = pos
        
        # 檢查是否撞牆
        if x < 0 or x >= self.BOARD_SIZE or y < 0 or y >= self.BOARD_SIZE:
            return True
        
        # 檢查是否撞到自己
        if pos in self.snake:
            return True
        
        return False
    
    def draw_board(self):
        """繪製遊戲畫面"""
        self.canvas.delete("all")
        
        # 繪製網格
        for i in range(self.BOARD_SIZE + 1):
            # 垂直線
            self.canvas.create_line(
                i * self.CELL_SIZE, 0,
                i * self.CELL_SIZE, self.CANVAS_SIZE,
                fill="gray", width=1
            )
            # 水平線
            self.canvas.create_line(
                0, i * self.CELL_SIZE,
                self.CANVAS_SIZE, i * self.CELL_SIZE,
                fill="gray", width=1
            )
        
        # 繪製蛇
        for i, (x, y) in enumerate(self.snake):
            x1 = x * self.CELL_SIZE + 2
            y1 = y * self.CELL_SIZE + 2
            x2 = x1 + self.CELL_SIZE - 4
            y2 = y1 + self.CELL_SIZE - 4
            
            if i == 0:  # 蛇頭
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="lime", outline="green", width=2)
                # 畫眼睛
                eye_size = 4
                self.canvas.create_oval(x1+8, y1+8, x1+8+eye_size, y1+8+eye_size, fill="black")
                self.canvas.create_oval(x2-8-eye_size, y1+8, x2-8, y1+8+eye_size, fill="black")
            else:  # 蛇身
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="lightgreen", outline="green")
        
        # 繪製食物
        if self.food:
            fx, fy = self.food
            fx1 = fx * self.CELL_SIZE + 4
            fy1 = fy * self.CELL_SIZE + 4
            fx2 = fx1 + self.CELL_SIZE - 8
            fy2 = fy1 + self.CELL_SIZE - 8
            self.canvas.create_oval(fx1, fy1, fx2, fy2, fill="red", outline="darkred", width=2)
    
    def start_game(self):
        """開始遊戲"""
        if not self.game_running:
            self.game_running = True
            self.start_button.config(text="暫停遊戲", bg="orange")
            self.move_snake()
        else:
            self.pause_game()
    
    def pause_game(self):
        """暫停遊戲"""
        self.game_running = False
        self.start_button.config(text="繼續遊戲", bg="green")
    
    def restart_game(self):
        """重新開始遊戲"""
        self.game_running = False
        self.snake = [(5, 5)]
        self.direction = (0, -1)
        self.food = self.generate_food()
        self.score = 0
        self.speed = 500
        self.score_label.config(text=f"分數: {self.score}")
        self.start_button.config(text="開始遊戲", bg="green")
        self.draw_board()
    
    def game_over(self):
        """遊戲結束"""
        self.game_running = False
        self.start_button.config(text="開始遊戲", bg="green")
        messagebox.showinfo("遊戲結束", f"遊戲結束！\n最終分數: {self.score}")
    
    def quit_game(self):
        """退出遊戲"""
        if messagebox.askquestion("確認", "確定要退出遊戲嗎？") == "yes":
            self.window.quit()
    
    def run(self):
        """運行遊戲"""
        self.window.mainloop()

# 運行遊戲
if __name__ == "__main__":
    game = SnakeGame()
    game.run()