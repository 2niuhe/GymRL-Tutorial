#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版自定义Gymnasium环境示例
本示例创建了一个简单的"寻宝"环境，智能体需要在网格世界中找到宝藏
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import pygame
import time

class SimpleTreasureEnv(gym.Env):
    """
    简化版寻宝环境：
    - 智能体在网格世界中移动
    - 目标是找到宝藏
    - 每走一步得到-1的奖励（表示时间成本）
    - 找到宝藏得到+10的奖励
    """
    metadata = {"render_modes": ["human", "rgb_array", "text"], "render_fps": 4}
    
    def __init__(self, grid_size=4, render_mode=None, fixed_treasure=True, fixed_treasure_pos=None):
        super(SimpleTreasureEnv, self).__init__()
        
        # 环境参数
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.cell_size = 100  # 每个网格单元的像素大小
        self.window_size = self.grid_size * self.cell_size  # 窗口大小
        
        # 固定宝藏位置选项
        self.fixed_treasure = fixed_treasure
        self.fixed_treasure_pos = fixed_treasure_pos
        if fixed_treasure and fixed_treasure_pos is None:
            # 默认固定宝藏位置在右上角
            self.fixed_treasure_pos = (self.grid_size - 1, self.grid_size - 1)
        
        # 动作空间: 0=上, 1=右, 2=下, 3=左
        self.action_space = spaces.Discrete(4)
        
        # 观察空间: 智能体位置 (x, y)，整数坐标
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, 
            shape=(2,), dtype=np.int32
        )
        
        # Pygame相关设置
        self.window = None
        self.clock = None
        
        # 重置环境
        self.reset()
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 随机放置智能体
        self.agent_position = np.array([
            self.np_random.integers(0, self.grid_size),
            self.np_random.integers(0, self.grid_size)
        ])
        
        # 放置宝藏
        if self.fixed_treasure:
            self.treasure_position = np.array(self.fixed_treasure_pos)
        else:
            # 随机放置宝藏（确保不与智能体重叠）
            while True:
                self.treasure_position = np.array([
                    self.np_random.integers(0, self.grid_size),
                    self.np_random.integers(0, self.grid_size)
                ])
                if not np.array_equal(self.agent_position, self.treasure_position):
                    break
        
        # 计算初始距离
        self.prev_distance = np.linalg.norm(self.agent_position - self.treasure_position)
        
        # 返回初始观察和信息
        observation = self.agent_position.copy()
        info = {"distance": self.prev_distance}
        
        return observation, info
    
    def step(self, action):
        """执行动作"""
        # 根据动作移动智能体
        if action == 0:  # 上
            self.agent_position[1] = min(self.agent_position[1] + 1, self.grid_size - 1)
        elif action == 1:  # 右
            self.agent_position[0] = min(self.agent_position[0] + 1, self.grid_size - 1)
        elif action == 2:  # 下
            self.agent_position[1] = max(self.agent_position[1] - 1, 0)
        elif action == 3:  # 左
            self.agent_position[0] = max(self.agent_position[0] - 1, 0)
        
        # 计算新距离
        current_distance = np.linalg.norm(self.agent_position - self.treasure_position)
        
        # 判断是否找到宝藏
        done = np.array_equal(self.agent_position, self.treasure_position)
        
        # 计算奖励
        if done:
            reward = 10.0  # 找到宝藏的奖励
        else:
            # 基于距离变化的奖励
            distance_delta = self.prev_distance - current_distance
            reward = distance_delta  # 距离减少得正奖励，增加得负奖励
            
            # 额外的小惩罚，鼓励智能体尽快找到宝藏
            reward -= 0.1
        
        # 更新前一次距离
        self.prev_distance = current_distance
        
        # 返回结果
        observation = self.agent_position.copy()
        info = {"distance": current_distance}
        
        # 渲染环境（如果需要）
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, done, False, info
    
    def _calculate_distance(self):
        """计算智能体与宝藏之间的距离"""
        return np.linalg.norm(self.agent_position - self.treasure_position)
    
    def render(self):
        """渲染环境"""
        if self.render_mode is None:
            return
            
        if self.render_mode == "text":
            # 文本渲染模式
            grid = [['·' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            grid[self.treasure_position[1]][self.treasure_position[0]] = 'T'
            grid[self.agent_position[1]][self.agent_position[0]] = 'A'
            
            # 如果智能体和宝藏在同一位置
            if np.array_equal(self.agent_position, self.treasure_position):
                grid[self.agent_position[1]][self.agent_position[0]] = 'X'
            
            # 打印网格
            print("-" * (self.grid_size * 2 + 1))
            for row in reversed(grid):  # 反转行以使y=0在底部
                print("|", end="")
                print(" ".join(row), end="")
                print("|")
            print("-" * (self.grid_size * 2 + 1))
            print(f"智能体位置: {self.agent_position}, 宝藏位置: {self.treasure_position}")
            print(f"距离: {self.prev_distance:.2f}")
            print()
            
        elif self.render_mode in ["human", "rgb_array"]:
            # 初始化Pygame
            if self.window is None and self.render_mode == "human":
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("寻宝环境")
                
            if self.clock is None and self.render_mode == "human":
                self.clock = pygame.time.Clock()
                
            # 创建画布
            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((255, 255, 255))  # 白色背景
            
            # 绘制网格线
            for i in range(self.grid_size + 1):
                pygame.draw.line(
                    canvas, 
                    (0, 0, 0), 
                    (0, i * self.cell_size), 
                    (self.window_size, i * self.cell_size), 
                    2
                )
                pygame.draw.line(
                    canvas, 
                    (0, 0, 0), 
                    (i * self.cell_size, 0), 
                    (i * self.cell_size, self.window_size), 
                    2
                )
            
            # 绘制宝藏
            treasure_rect = pygame.Rect(
                self.treasure_position[0] * self.cell_size + 10,
                (self.grid_size - 1 - self.treasure_position[1]) * self.cell_size + 10,  # 反转y坐标以使y=0在底部
                self.cell_size - 20,
                self.cell_size - 20
            )
            pygame.draw.rect(canvas, (255, 215, 0), treasure_rect)  # 金色宝藏
            
            # 绘制智能体
            agent_rect = pygame.Rect(
                self.agent_position[0] * self.cell_size + 15,
                (self.grid_size - 1 - self.agent_position[1]) * self.cell_size + 15,  # 反转y坐标以使y=0在底部
                self.cell_size - 30,
                self.cell_size - 30
            )
            pygame.draw.rect(canvas, (0, 0, 255), agent_rect)  # 蓝色智能体
            
            # 如果找到宝藏，绘制特效
            if np.array_equal(self.agent_position, self.treasure_position):
                # 绘制一个星形或其他标记表示成功
                center_x = self.agent_position[0] * self.cell_size + self.cell_size // 2
                center_y = (self.grid_size - 1 - self.agent_position[1]) * self.cell_size + self.cell_size // 2
                radius = self.cell_size // 3
                pygame.draw.circle(canvas, (255, 0, 0), (center_x, center_y), radius)  # 红色圆圈
            
            # 显示距离信息
            if self.render_mode == "human":
                font = pygame.font.SysFont(None, 24)
                distance_text = font.render(f"距离: {self.prev_distance:.2f}", True, (0, 0, 0))
                canvas.blit(distance_text, (10, 10))
            
            if self.render_mode == "human":
                # 复制画布到窗口并更新显示
                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()
                
                # 控制帧率
                self.clock.tick(self.metadata["render_fps"])
                
                # 添加短暂延迟，确保窗口显示
                time.sleep(0.1)
                
            elif self.render_mode == "rgb_array":
                # 返回RGB数组
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(canvas)), 
                    axes=(1, 0, 2)
                )
    
    def close(self):
        """关闭环境"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

class SimpleQLearningAgent:
    """简化版Q-learning智能体"""
    
    def __init__(self, action_size, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05):
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}
        
    def get_q_value(self, state, action):
        """获取状态-动作对的Q值"""
        state_key = tuple(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return self.q_table[state_key][action]
    
    def choose_action(self, state, eval_mode=False):
        """选择动作"""
        state_key = tuple(state)
        
        # 确保状态在Q表中
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # 探索-利用权衡
        if eval_mode:
            # 评估模式下使用较低的探索率
            explore_prob = 0.05
        else:
            explore_prob = self.epsilon
            
        if np.random.random() < explore_prob:
            # 探索: 随机选择动作
            return np.random.randint(self.action_size)
        else:
            # 利用: 选择Q值最高的动作
            q_values = self.q_table[state_key]
            
            # 如果多个动作有相同的最大Q值，随机选择其中一个
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            
            if len(max_actions) > 1:
                return np.random.choice(max_actions)
            else:
                return np.argmax(q_values)
    
    def learn(self, state, action, reward, next_state, done):
        """更新Q值"""
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # 确保状态在Q表中
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # 当前Q值
        current_q = self.q_table[state_key][action]
        
        # 下一状态的最大Q值
        max_next_q = np.max(self.q_table[next_state_key])
        
        # 计算新的Q值
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * max_next_q
        
        # 更新Q值
        self.q_table[state_key][action] += self.lr * (target_q - current_q)
        
        # 如果回合结束，减小探索率
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def render_q_values(env, agent):
    """渲染Q值和最优动作"""
    # 初始化Pygame
    pygame.init()
    pygame.display.init()
    
    # 创建窗口
    cell_size = 100
    window_size = env.grid_size * cell_size
    window = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Q值和最优动作")
    
    # 创建画布
    canvas = pygame.Surface((window_size, window_size))
    canvas.fill((255, 255, 255))  # 白色背景
    
    # 绘制网格线
    for i in range(env.grid_size + 1):
        pygame.draw.line(
            canvas, 
            (0, 0, 0), 
            (0, i * cell_size), 
            (window_size, i * cell_size), 
            2
        )
        pygame.draw.line(
            canvas, 
            (0, 0, 0), 
            (i * cell_size, 0), 
            (i * cell_size, window_size), 
            2
        )
    
    # 绘制宝藏
    treasure_rect = pygame.Rect(
        env.treasure_position[0] * cell_size + 10,
        (env.grid_size - 1 - env.treasure_position[1]) * cell_size + 10,
        cell_size - 20,
        cell_size - 20
    )
    pygame.draw.rect(canvas, (255, 215, 0), treasure_rect)  # 金色宝藏
    
    # 绘制每个状态的最优动作
    font = pygame.font.SysFont(None, 24)
    
    # 动作到箭头的映射
    # 0=上, 1=右, 2=下, 3=左
    arrows = ["↑", "→", "↓", "←"]
    arrow_colors = [(0, 200, 0), (200, 0, 0), (0, 0, 200), (200, 200, 0)]  # 不同颜色表示不同动作
    
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            state = (x, y)
            state_key = tuple(state)
            
            # 计算单元格中心
            center_x = x * cell_size + cell_size // 2
            center_y = (env.grid_size - 1 - y) * cell_size + cell_size // 2
            
            # 如果状态在Q表中
            if state_key in agent.q_table:
                q_values = agent.q_table[state_key]
                best_action = np.argmax(q_values)
                max_q = np.max(q_values)
                
                # 绘制箭头
                arrow_text = font.render(arrows[best_action], True, arrow_colors[best_action])
                arrow_rect = arrow_text.get_rect(center=(center_x, center_y))
                canvas.blit(arrow_text, arrow_rect)
                
                # 绘制Q值
                q_text = font.render(f"{max_q:.1f}", True, (0, 0, 0))
                q_rect = q_text.get_rect(center=(center_x, center_y + 20))
                canvas.blit(q_text, q_rect)
                
                # 绘制箭头线条
                arrow_length = 20
                if best_action == 0:  # 上
                    pygame.draw.line(canvas, arrow_colors[best_action], 
                                    (center_x, center_y), 
                                    (center_x, center_y - arrow_length), 3)
                    pygame.draw.polygon(canvas, arrow_colors[best_action], 
                                       [(center_x - 5, center_y - arrow_length + 5),
                                        (center_x + 5, center_y - arrow_length + 5),
                                        (center_x, center_y - arrow_length)])
                elif best_action == 1:  # 右
                    pygame.draw.line(canvas, arrow_colors[best_action], 
                                    (center_x, center_y), 
                                    (center_x + arrow_length, center_y), 3)
                    pygame.draw.polygon(canvas, arrow_colors[best_action], 
                                       [(center_x + arrow_length - 5, center_y - 5),
                                        (center_x + arrow_length - 5, center_y + 5),
                                        (center_x + arrow_length, center_y)])
                elif best_action == 2:  # 下
                    pygame.draw.line(canvas, arrow_colors[best_action], 
                                    (center_x, center_y), 
                                    (center_x, center_y + arrow_length), 3)
                    pygame.draw.polygon(canvas, arrow_colors[best_action], 
                                       [(center_x - 5, center_y + arrow_length - 5),
                                        (center_x + 5, center_y + arrow_length - 5),
                                        (center_x, center_y + arrow_length)])
                elif best_action == 3:  # 左
                    pygame.draw.line(canvas, arrow_colors[best_action], 
                                    (center_x, center_y), 
                                    (center_x - arrow_length, center_y), 3)
                    pygame.draw.polygon(canvas, arrow_colors[best_action], 
                                       [(center_x - arrow_length + 5, center_y - 5),
                                        (center_x - arrow_length + 5, center_y + 5),
                                        (center_x - arrow_length, center_y)])
    
    # 显示画布
    window.blit(canvas, canvas.get_rect())
    pygame.display.update()
    
    # 等待用户关闭窗口
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    # 清理
    pygame.quit()

def train_agent(env, agent, episodes=300, max_steps=100):
    """训练智能体"""
    rewards = []
    success_count = 0
    
    for episode in range(1, episodes+1):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done, _, _ = env.step(action)
            
            # 学习
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # 检查是否找到宝藏
            if done and reward > 0:
                success_count += 1
        
        rewards.append(episode_reward)
        
        # 每20回合打印一次进度
        if episode % 20 == 0 or episode == 1:
            avg_reward = np.mean(rewards[-20:])
            success_rate = success_count / episode * 100
            print(f"回合: {episode}/{episodes}, 平均奖励: {avg_reward:.2f}, 成功率: {success_rate:.1f}%, 探索率: {agent.epsilon:.4f}")
    
    print(f"训练完成! 总成功率: {success_count / episodes * 100:.1f}%")
    return rewards

def evaluate_agent(env, agent, episodes=5, max_steps=30):
    """评估智能体性能"""
    total_reward = 0
    total_steps = 0
    success_count = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        visited_states = set()  # 记录已访问状态，检测循环
        
        print(f"\n===== 评估回合 {episode+1} =====")
        
        while not done and steps < max_steps:
            # 记录当前状态
            state_key = tuple(state)
            visited_states.add(state_key)
            
            # 选择动作（评估时使用小概率随机动作，打破循环）
            action = agent.choose_action(state, eval_mode=True)
            
            # 执行动作
            next_state, reward, done, _, _ = env.step(action)
            
            # 检测循环 - 如果下一个状态已经访问过多次，增加随机性
            next_state_key = tuple(next_state)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # 渲染环境
            env.render()
            
            # 检查是否找到宝藏
            if done and reward > 0:
                success_count += 1
                print(f"成功! 找到宝藏用了 {steps} 步")
        
        if not done:
            print(f"失败! 达到最大步数 {max_steps}")
            
        total_reward += episode_reward
        total_steps += steps
    
    avg_reward = total_reward / episodes
    avg_steps = total_steps / episodes
    success_rate = success_count / episodes * 100
    print(f"\n评估 {episodes} 回合的平均奖励: {avg_reward:.2f}, 平均步数: {avg_steps:.2f}")
    print(f"成功率: {success_rate:.1f}%")
    
    # 显示Q值和最优动作
    print("\n显示Q值和最优动作...")
    render_q_values(env, agent)
    
    return avg_reward

def plot_training_results(rewards, filename="simple_treasure_results.png"):
    """绘制训练结果"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-learning Training Results')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    """主函数"""
    # 创建环境 - 训练时不使用渲染，使用固定宝藏位置
    # 固定宝藏位置在右上角 (3, 3)
    fixed_treasure_pos = (3, 3)
    env = SimpleTreasureEnv(grid_size=4, render_mode=None, fixed_treasure=True, fixed_treasure_pos=fixed_treasure_pos)
    
    # 创建智能体 - 使用更优化的参数
    action_size = env.action_space.n
    agent = SimpleQLearningAgent(
        action_size,
        learning_rate=0.2,       # 提高学习率
        discount_factor=0.99,    # 提高折扣因子，更重视未来奖励
        epsilon=1.0,             # 初始探索率
        epsilon_decay=0.99,      # 减缓探索率衰减
        min_epsilon=0.05         # 最小探索率
    )
    
    print(f"状态空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"宝藏位置固定在: {fixed_treasure_pos}")
    
    # 训练智能体
    episodes = 500  # 增加训练回合数
    print(f"开始训练 {episodes} 回合...")
    rewards = train_agent(env, agent, episodes=episodes, max_steps=100)
    
    # 绘制训练结果
    plot_training_results(rewards)
    
    # 评估智能体 - 使用Pygame渲染模式，保持相同的固定宝藏位置
    print("\n评估智能体性能...")
    eval_env = SimpleTreasureEnv(grid_size=4, render_mode="human", fixed_treasure=True, fixed_treasure_pos=fixed_treasure_pos)
    evaluate_agent(eval_env, agent, episodes=5, max_steps=30)
    
    # 关闭环境
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
