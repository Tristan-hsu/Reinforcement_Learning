import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from snake_game_env import SnakeEnvironment
import torch
from model import DQNAgent
import os

class SnakeAIVisualizer:
    """AI訓練視覺化介面"""
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("貪吃蛇 AI 強化學習")
        self.window.geometry("1200x750")
        
        # 初始化環境和agent為None，稍後根據參數創建
        self.env = None
        self.agent = None
        
        # 訓練統計
        self.scores = []
        self.episodes = []
        self.training = False
        self.reward_history = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """設置用戶界面"""
        # 標題
        title = tk.Label(self.window, text="貪吃蛇 AI 強化學習", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # 參數設定區域
        self.setup_parameter_frame()
        
        # 模型管理區域
        self.setup_model_management_frame()
        
        # 控制面板
        control_frame = tk.Frame(self.window)
        control_frame.pack(pady=10)
        
        self.start_btn = tk.Button(control_frame, text="開始訓練", command=self.start_training, 
                                  bg="green", fg="white", font=("Arial", 12))
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(control_frame, text="停止訓練", command=self.stop_training, 
                                 bg="red", fg="white", font=("Arial", 12))
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.check_btn = tk.Button(control_frame, text="查看訓練歷史", command=self.plot_train_reward_history, 
            bg="orange", fg="white", font=("Arial", 12))
        self.check_btn.pack(side=tk.LEFT, padx=5)
        
        self.test_btn = tk.Button(control_frame, text="測試AI", command=self.test_ai, 
                                 bg="blue", fg="white", font=("Arial", 12))
        self.test_btn.pack(side=tk.LEFT, padx=5)
        
        # 統計顯示
        stats_frame = tk.Frame(self.window)
        stats_frame.pack(pady=10)
        
        self.episode_label = tk.Label(stats_frame, text="回合: 0", font=("Arial", 12))
        self.episode_label.pack(side=tk.LEFT, padx=20)
        
        self.score_label = tk.Label(stats_frame, text="最高分數: 0", font=("Arial", 12))
        self.score_label.pack(side=tk.LEFT, padx=20)
        
        self.epsilon_label = tk.Label(stats_frame, text="探索率: 1.00", font=("Arial", 12))
        self.epsilon_label.pack(side=tk.LEFT, padx=20)
        
        # 遊戲顯示區域
        self.game_frame = tk.Frame(self.window, bg="black", width=400, height=400)
        self.game_frame.pack(pady=20)
        self.game_frame.pack_propagate(False)
        
        self.canvas = tk.Canvas(self.game_frame, width=400, height=400, bg="black")
        self.canvas.pack()
        
        # 訓練日誌
        log_frame = tk.Frame(self.window)
        log_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        tk.Label(log_frame, text="訓練日誌:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.log_text = tk.Text(log_frame, height=6, width=80)
        scrollbar = tk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.config(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_parameter_frame(self):
        """設置參數設定區域"""
        param_main_frame = tk.Frame(self.window, relief=tk.RIDGE, bd=2)
        param_main_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # 參數標題
        param_title = tk.Label(param_main_frame, text="參數設定", font=("Arial", 12, "bold"))
        param_title.pack(pady=5)
        
        # 創建兩列佈局
        param_frame = tk.Frame(param_main_frame)
        param_frame.pack(pady=5, padx=10, fill=tk.X)
        
        # 左列 - 環境參數
        env_frame = tk.LabelFrame(param_frame, text="環境參數", font=("Arial", 10, "bold"))
        env_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # 棋盤大小
        tk.Label(env_frame, text="棋盤大小:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.board_size_var = tk.StringVar(value="10")
        tk.Entry(env_frame, textvariable=self.board_size_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        # 獎勵值
        tk.Label(env_frame, text="獎勵值:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.reward_var = tk.StringVar(value="10")
        tk.Entry(env_frame, textvariable=self.reward_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # 懲罰值
        tk.Label(env_frame, text="懲罰值:", font=("Arial", 10)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.punishment_var = tk.StringVar(value="-100")
        tk.Entry(env_frame, textvariable=self.punishment_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # 超時懲罰
        tk.Label(env_frame, text="超時懲罰:", font=("Arial", 10)).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.deadline_punishment_var = tk.StringVar(value="10")
        tk.Entry(env_frame, textvariable=self.deadline_punishment_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        # 右列 - 智能體參數
        agent_frame = tk.LabelFrame(param_frame, text="智能體參數", font=("Arial", 10, "bold"))
        agent_frame.pack(side=tk.RIGHT, padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # 狀態空間大小
        tk.Label(agent_frame, text="狀態空間大小:", font=("Arial", 10)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.state_size_var = tk.StringVar(value="16")
        tk.Entry(agent_frame, textvariable=self.state_size_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        # 動作空間大小
        tk.Label(agent_frame, text="動作空間大小:", font=("Arial", 10)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.action_size_var = tk.StringVar(value="3")
        tk.Entry(agent_frame, textvariable=self.action_size_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # 學習率
        tk.Label(agent_frame, text="學習率:", font=("Arial", 10)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.learning_rate_var = tk.StringVar(value="0.001")
        tk.Entry(agent_frame, textvariable=self.learning_rate_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # 折扣因子
        tk.Label(agent_frame, text="折扣因子:", font=("Arial", 10)).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.gamma_var = tk.StringVar(value="0.95")
        tk.Entry(agent_frame, textvariable=self.gamma_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        # 應用參數按鈕
        apply_frame = tk.Frame(param_main_frame)
        apply_frame.pack(pady=5)
        
        self.apply_btn = tk.Button(apply_frame, text="應用參數", command=self.apply_parameters, 
                                  bg="darkgreen", fg="white", font=("Arial", 10, "bold"))
        self.apply_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(apply_frame, text="重置參數", command=self.reset_parameters, 
                                  bg="gray", fg="white", font=("Arial", 10))
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # 狀態標籤
        self.param_status_label = tk.Label(param_main_frame, text="狀態: 請應用參數後開始訓練", 
                                          font=("Arial", 9), fg="red")
        self.param_status_label.pack(pady=2)
    
    def setup_model_management_frame(self):
        """設置模型管理區域"""
        model_main_frame = tk.Frame(self.window, relief=tk.RIDGE, bd=2)
        model_main_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # 模型管理標題
        model_title = tk.Label(model_main_frame, text="模型管理", font=("Arial", 12, "bold"))
        model_title.pack(pady=5)
        
        # 模型路徑輸入區域
        path_frame = tk.Frame(model_main_frame)
        path_frame.pack(pady=5, padx=10, fill=tk.X)
        
        # 儲存路徑
        save_frame = tk.Frame(path_frame)
        save_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(save_frame, text="儲存路徑:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.save_path_var = tk.StringVar(value="snake_ai_model.pth")
        self.save_path_entry = tk.Entry(save_frame, textvariable=self.save_path_var, width=40)
        self.save_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.browse_save_btn = tk.Button(save_frame, text="瀏覽", command=self.browse_save_path, 
                                        bg="lightblue", font=("Arial", 9))
        self.browse_save_btn.pack(side=tk.LEFT, padx=2)
        
        self.save_btn = tk.Button(save_frame, text="儲存模型", command=self.save_model, 
                                 bg="purple", fg="white", font=("Arial", 10))
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # 載入路徑
        load_frame = tk.Frame(path_frame)
        load_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(load_frame, text="載入路徑:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.load_path_var = tk.StringVar(value="snake_ai_model.pth")
        self.load_path_entry = tk.Entry(load_frame, textvariable=self.load_path_var, width=40)
        self.load_path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.browse_load_btn = tk.Button(load_frame, text="瀏覽", command=self.browse_load_path, 
                                        bg="lightblue", font=("Arial", 9))
        self.browse_load_btn.pack(side=tk.LEFT, padx=2)
        
        self.load_btn = tk.Button(load_frame, text="載入模型", command=self.load_model, 
                                 bg="darkblue", fg="white", font=("Arial", 10))
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # 模型狀態顯示
        self.model_status_label = tk.Label(model_main_frame, text="模型狀態: 未載入模型", 
                                          font=("Arial", 9), fg="gray")
        self.model_status_label.pack(pady=2)
    
    def browse_save_path(self):
        """瀏覽儲存路徑"""
        filename = filedialog.asksaveasfilename(
            title="選擇儲存位置",
            defaultextension=".pth",
            filetypes=[("PyTorch模型", "*.pth"), ("所有檔案", "*.*")]
        )
        if filename:
            self.save_path_var.set(filename)
    
    def browse_load_path(self):
        """瀏覽載入路徑"""
        filename = filedialog.askopenfilename(
            title="選擇模型檔案",
            filetypes=[("PyTorch模型", "*.pth"), ("所有檔案", "*.*")]
        )
        if filename:
            self.load_path_var.set(filename)
    
    def apply_parameters(self):
        """應用參數設定"""
        try:
            # 獲取環境參數
            board_size = int(self.board_size_var.get())
            reward = float(self.reward_var.get())
            punishment = float(self.punishment_var.get())
            deadline_punishment = float(self.deadline_punishment_var.get())
            
            # 獲取智能體參數
            state_size = int(self.state_size_var.get())
            action_size = int(self.action_size_var.get())
            learning_rate = float(self.learning_rate_var.get())
            gamma = float(self.gamma_var.get())
            
            # 驗證參數合理性
            if board_size < 5 or board_size > 50:
                raise ValueError("棋盤大小應在5-50之間")
            if state_size < 1 or action_size < 1:
                raise ValueError("狀態空間和動作空間大小必須大於0")
            if learning_rate <= 0 or learning_rate > 1:
                raise ValueError("學習率應在0-1之間")
            if gamma <= 0 or gamma > 1:
                raise ValueError("折扣因子應在0-1之間")
            
            # 創建環境和智能體
            self.env = SnakeEnvironment(
                board_size=board_size,
                reward=reward,
                punishment=punishment,
                deadline_punishment=deadline_punishment
            )
            
            self.agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=learning_rate,
                gamma=gamma
            )
            
            # 更新狀態
            self.param_status_label.config(text="狀態: 參數已應用，可以開始訓練", fg="green")
            self.start_btn.config(state=tk.NORMAL)
            self.model_status_label.config(text="模型狀態: 新模型已創建", fg="green")
            
            # 記錄參數
            self.log_message("參數設定成功:")
            self.log_message(f"  環境: 棋盤大小={board_size}, 獎勵={reward}, 懲罰={punishment}, 超時懲罰={deadline_punishment}")
            self.log_message(f"  智能體: 狀態空間={state_size}, 動作空間={action_size}, 學習率={learning_rate}, 折扣因子={gamma}")
            
        except ValueError as e:
            self.param_status_label.config(text=f"錯誤: {str(e)}", fg="red")
            self.log_message(f"參數設定錯誤: {str(e)}")
        except Exception as e:
            self.param_status_label.config(text=f"未知錯誤: {str(e)}", fg="red")
            self.log_message(f"參數應用失敗: {str(e)}")
    
    def reset_parameters(self):
        """重置參數為預設值"""
        self.board_size_var.set("10")
        self.reward_var.set("10")
        self.punishment_var.set("-100")
        self.deadline_punishment_var.set("10")
        self.state_size_var.set("16")
        self.action_size_var.set("3")
        self.learning_rate_var.set("0.001")
        self.gamma_var.set("0.95")
        
        self.param_status_label.config(text="狀態: 參數已重置，請重新應用", fg="orange")
        self.log_message("參數已重置為預設值")
    
    def log_message(self, message):
        """記錄訓練日誌"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.window.update()
    
    def draw_game(self, snake, food, score):
        """繪製遊戲狀態"""
        if not self.env:
            return
            
        self.canvas.delete("all")
        cell_size = 400 // self.env.board_size
        
        # 繪製網格
        for i in range(self.env.board_size + 1):
            self.canvas.create_line(i * cell_size, 0, i * cell_size, 400, fill="gray", width=1)
            self.canvas.create_line(0, i * cell_size, 400, i * cell_size, fill="gray", width=1)
        
        # 繪製蛇
        for i, (x, y) in enumerate(snake):
            x1, y1 = x * cell_size + 2, y * cell_size + 2
            x2, y2 = x1 + cell_size - 4, y1 + cell_size - 4
            color = "lime" if i == 0 else "lightgreen"
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="green")
        
        # 繪製食物
        fx, fy = food
        fx1, fy1 = fx * cell_size + 4, fy * cell_size + 4
        fx2, fy2 = fx1 + cell_size - 8, fy1 + cell_size - 8
        self.canvas.create_oval(fx1, fy1, fx2, fy2, fill="red", outline="darkred", width=2)
        
        # 顯示分數
        self.canvas.create_text(200, 20, text=f"分數: {score}", fill="white", font=("Arial", 14, "bold"))
    
    def start_training(self):
        """開始訓練"""
        if not self.env or not self.agent:
            self.log_message("錯誤: 請先應用參數設定!")
            return
            
        if not self.training:
            self.training = True
            self.start_btn.config(state=tk.DISABLED)
            self.apply_btn.config(state=tk.DISABLED)  # 訓練時禁用參數修改
            self.log_message("開始AI訓練...")
            
            # 在新線程中運行訓練
            training_thread = threading.Thread(target=self.train_agent)
            training_thread.daemon = True
            training_thread.start()
    
    def stop_training(self):
        """停止訓練"""
        self.training = False
        self.start_btn.config(state=tk.NORMAL)
        self.apply_btn.config(state=tk.NORMAL)  # 恢復參數修改
        self.log_message("訓練已停止")
    
    def train_agent(self):
        """訓練AI agent"""
        episode = 0
        max_score = 0
        self.reward_history = []
        
        while self.training:
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                if not self.training:
                    break
                
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                # 視覺化當前遊戲狀態
                if episode % 10 == 0:  # 每10回合顯示一次
                    self.draw_game(self.env.snake, self.env.food, self.env.score)
                    time.sleep(0.1)
                
                if done:
                    break
            
            # 經驗回放
            if len(self.agent.memory) > self.agent.batch_size:
                self.agent.replay()
            
            # 更新目標網路
            if episode % 100 == 0:
                self.agent.update_target_network()
            
            # 更新統計
            episode += 1
            max_score = max(max_score, self.env.score)
            
            # 更新UI
            self.episode_label.config(text=f"回合: {episode}")
            self.score_label.config(text=f"最高分數: {max_score}")
            self.epsilon_label.config(text=f"探索率: {self.agent.epsilon:.3f}")
            self.reward_history.append(total_reward)
            
            if episode % 50 == 0:
                avg_score = np.mean([self.env.score for _ in range(min(50, len(self.scores)))])
                self.log_message(f"回合 {episode}: 平均分數 {avg_score:.2f}, 最高分數 {max_score}, 探索率 {self.agent.epsilon:.3f}")
        
        self.log_message("訓練完成!")
        
    def plot_train_reward_history(self):
        """顯示訓練歷史"""
        if not self.reward_history:
            self.log_message("尚無訓練歷史數據")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.reward_history)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Reward History')
        plt.grid(True)
        plt.show()
     
    def save_model(self):
        """儲存模型"""
        if not self.agent:
            self.log_message("錯誤: 沒有可儲存的模型")
            self.model_status_label.config(text="模型狀態: 儲存失敗 - 無模型", fg="red")
            return
            
        try:
            model_path = self.save_path_var.get().strip()
            if not model_path:
                model_path = "snake_ai_model.pth"
                self.save_path_var.set(model_path)
            
            # 確保目錄存在
            save_dir = os.path.dirname(model_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 儲存完整的模型狀態
            model_state = {
                'q_network_state_dict': self.agent.q_network.state_dict(),
                'target_network_state_dict': self.agent.target_network.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
                'epsilon': self.agent.epsilon,
                'model_params': {
                    'state_size': self.agent.state_size,
                    'action_size': self.agent.action_size,
                    'learning_rate': self.agent.learning_rate,
                    'gamma': self.agent.gamma
                },
                'training_episodes': len(self.reward_history) if hasattr(self, 'reward_history') else 0
            }
            
            torch.save(model_state, model_path)
            
            # 檢查檔案大小
            file_size = os.path.getsize(model_path)
            self.log_message(f"模型已儲存為 {model_path} ({file_size/1024:.2f} KB)")
            self.model_status_label.config(text=f"模型狀態: 已儲存到 {os.path.basename(model_path)}", fg="green")
            
        except Exception as e:
            self.log_message(f"模型儲存失敗: {str(e)}")
            self.model_status_label.config(text=f"模型狀態: 儲存失敗 - {str(e)}", fg="red")
    
    def load_model(self):
        """載入模型"""
        if not self.agent:
            self.log_message("錯誤: 請先應用參數設定創建智能體!")
            self.model_status_label.config(text="模型狀態: 載入失敗 - 無智能體", fg="red")
            return
            
        model_path = self.load_path_var.get().strip()
        if not model_path:
            self.log_message("錯誤: 請輸入模型路徑!")
            self.model_status_label.config(text="模型狀態: 載入失敗 - 無路徑", fg="red")
            return
            
        try:
            if not os.path.exists(model_path):
                self.log_message(f"錯誤: 找不到模型檔案 {model_path}")
                self.model_status_label.config(text="模型狀態: 載入失敗 - 檔案不存在", fg="red")
                return
            
            # 載入模型狀態
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                # 檢查模型參數是否匹配
                if 'model_params' in checkpoint:
                    saved_params = checkpoint['model_params']
                    current_params = {
                        'state_size': self.agent.state_size,
                        'action_size': self.agent.action_size,
                        'learning_rate': self.agent.learning_rate,
                        'gamma': self.agent.gamma
                    }
                    
                    # 檢查關鍵參數
                    if (saved_params['state_size'] != current_params['state_size'] or 
                        saved_params['action_size'] != current_params['action_size']):
                        
                        # 詢問用戶是否要更新參數
                        response = messagebox.askyesno(
                            "參數不匹配", 
                            f"模型參數與當前設定不匹配:\n"
                            f"模型: 狀態={saved_params['state_size']}, 動作={saved_params['action_size']}\n"
                            f"當前: 狀態={current_params['state_size']}, 動作={current_params['action_size']}\n\n"
                            f"是否要根據模型參數重新創建智能體?"
                        )
                        
                        if response:
                            # 更新參數並重新創建智能體
                            self.state_size_var.set(str(saved_params['state_size']))
                            self.action_size_var.set(str(saved_params['action_size']))
                            self.learning_rate_var.set(str(saved_params['learning_rate']))
                            self.gamma_var.set(str(saved_params['gamma']))
                            
                            # 重新應用參數
                            self.apply_parameters()
                        else:
                            self.log_message("載入取消: 參數不匹配")
                            self.model_status_label.config(text="模型狀態: 載入取消", fg="orange")
                            return
                
                # 載入網路權重
                if 'q_network_state_dict' in checkpoint:
                    self.agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                    self.log_message("✓ Q網路權重已載入")
                    
                if 'target_network_state_dict' in checkpoint and hasattr(self.agent, 'target_network'):
                    self.agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                    self.log_message("✓ 目標網路權重已載入")
                    
                if 'optimizer_state_dict' in checkpoint:
                    self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.log_message("✓ 優化器狀態已載入")
                    
                if 'epsilon' in checkpoint:
                    self.agent.epsilon = checkpoint['epsilon']
                    self.epsilon_label.config(text=f"探索率: {self.agent.epsilon:.3f}")
                    self.log_message(f"✓ 探索率已設定為: {self.agent.epsilon:.3f}")
                
                # 載入訓練統計
                if 'training_episodes' in checkpoint:
                    self.episode_label.config(text=f"回合: {checkpoint['training_episodes']}")
                    self.log_message(f"✓ 訓練回合數: {checkpoint['training_episodes']}")
                
                self.log_message(f"模型成功載入: {model_path}")
                self.model_status_label.config(text=f"模型狀態: 已載入 {os.path.basename(model_path)}", fg="blue")
                
            else:
                # 如果只是state_dict格式
                self.agent.q_network.load_state_dict(checkpoint)
                self.log_message(f"模型權重已載入: {model_path}")
                self.model_status_label.config(text=f"模型狀態: 權重已載入 {os.path.basename(model_path)}", fg="blue")
            
            # 檢查檔案大小
            file_size = os.path.getsize(model_path)
            self.log_message(f"檔案大小: {file_size/1024:.2f} KB")
            
        except Exception as e:
            self.log_message(f"模型載入失敗: {str(e)}")
            self.model_status_label.config(text=f"模型狀態: 載入失敗 - {str(e)}", fg="red")
            import traceback
            self.log_message(f"詳細錯誤: {traceback.format_exc()}")
    
    def test_ai(self):
        """測試訓練好的AI"""
        if not self.env or not self.agent:
            self.log_message("錯誤: 請先應用參數設定!")
            return
            
        self.log_message("測試AI性能...")   
        # 設置為測試模式（不探索）
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0
        
        try:
            state = self.env.reset()
            step_count = 0
            max_steps = 1000
            
            while step_count < max_steps:
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)
                
                self.draw_game(self.env.snake, self.env.food, self.env.score)
                self.window.update()
                time.sleep(0.2)
                
                state = next_state
                step_count += 1
                
                if done:
                    break
            
            self.log_message(f"測試完成! AI獲得分數: {self.env.score}, 步數: {step_count}")
            
        except Exception as e:
            self.log_message(f"測試錯誤: {str(e)}")
        finally:
            # 恢復探索率
            self.agent.epsilon = original_epsilon
            self.epsilon_label.config(text=f"探索率: {self.agent.epsilon:.3f}")

    def run(self):
        """運行應用程式"""
        # 初始狀態下禁用開始訓練按鈕
        self.start_btn.config(state=tk.DISABLED)
        self.window.mainloop()

# 主程式
if __name__ == "__main__":
    # 使用說明
    print("貪吃蛇 AI 強化學習系統")
    print("=" * 60)
    print("功能說明:")
    print("1. 在參數設定區域調整環境和智能體參數")
    print("2. 點擊'應用參數'確認設定")
    print("3. 使用模型管理區域儲存和載入訓練好的模型")
    print("4. 點擊'開始訓練'開始AI學習過程")
    print("5. 點擊'停止訓練'停止當前訓練")
    print("6. 點擊'測試AI'查看訓練結果")
    print("7. 點擊'查看訓練歷史'查看訓練圖表")
    print("8. 可以在訓練過程中隨時儲存模型")
    print("9. 載入預訓練模型繼續訓練或測試")
    print("=" * 60)
    
    app = SnakeAIVisualizer()
    app.run()