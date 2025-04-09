#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gymnasium示例：Q-Learning算法实现（以MountainCar为例）
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        # 环境相关
        self.env = env
        # 将状态空间离散化为20x20的网格
        self.state_bins = [20, 20]
        self.action_size = env.action_space.n
        
        # Q-learning参数
        self.learning_rate = learning_rate  # 学习率
        self.discount_factor = discount_factor  # 折扣因子
        self.exploration_rate = exploration_rate  # 探索率
        self.exploration_min = 0.01  # 最小探索率
        self.exploration_decay = exploration_decay  # 探索率衰减
        
        # 初始化Q表
        self.q_table = np.zeros((*self.state_bins, self.action_size))
    
    def discretize_state(self, state):
        """将连续状态离散化"""
        discrete_state = []
        for i, (s, low, high, bins) in enumerate(zip(
                state, 
                self.env.observation_space.low, 
                self.env.observation_space.high,
                self.state_bins
            )):
            # 处理无限边界
            if np.isinf(low):
                low = -1e10
            if np.isinf(high):
                high = 1e10
                
            # 离散化
            scaled = int(np.floor((s - low) / (high - low) * bins))
            scaled = min(bins - 1, max(0, scaled))
            discrete_state.append(scaled)
            
        return tuple(discrete_state)
    
    def discretize_range(self, lower, upper, values, bins=None):
        """将值范围离散化为指定数量的箱子"""
        # 这个方法在当前实现中不再需要，但保留它以避免修改太多代码
        return values
    
    def choose_action(self, state):
        """根据当前状态选择动作（ε-贪婪策略）"""
        # 探索：随机选择动作
        if np.random.random() < self.exploration_rate:
            return self.env.action_space.sample()
        
        # 利用：选择Q值最大的动作
        discrete_state = self.discretize_state(state)
        return np.argmax(self.q_table[discrete_state])
    
    def learn(self, state, action, reward, next_state, done):
        """更新Q表"""
        # 离散化状态
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # 当前状态-动作对的Q值
        current_q = self.q_table[discrete_state + (action,)]
        
        # 下一个状态的最大Q值
        max_next_q = np.max(self.q_table[discrete_next_state])
        
        # Q-learning更新公式
        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * max_next_q
        
        # 更新Q值
        self.q_table[discrete_state + (action,)] += self.learning_rate * (target_q - current_q)
        
        # 衰减探索率
        if done:
            self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)
    
    def train(self, episodes, render_interval=100):
        """训练智能体"""
        rewards = []
        
        for episode in range(1, episodes + 1):
            # 重置环境
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            # 渲染模式
            render = (episode % render_interval == 0)
            
            while not done:
                # 选择动作
                action = self.choose_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 学习
                self.learn(state, action, reward, next_state, done)
                
                # 更新状态和奖励
                state = next_state
                total_reward += reward
                
                # 渲染
                if render and self.env.render_mode == "human":
                    self.env.render()
                    time.sleep(0.01)
            
            # 记录奖励
            rewards.append(total_reward)
            
            # 打印进度
            if episode % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                print(f"回合: {episode}/{episodes}, 平均奖励: {avg_reward:.2f}, 探索率: {self.exploration_rate:.4f}")
        
        return rewards

def main():
    # 创建环境
    env = gym.make("MountainCar-v0", render_mode="human")
    
    # 创建Q-learning智能体
    agent = QLearningAgent(env)
    
    # 训练智能体
    print("开始训练Q-learning智能体...")
    episodes = 100
    rewards = agent.train(episodes)
    
    # 绘制训练过程中的奖励变化
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, episodes + 1), rewards)
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.title('Q-learning训练过程')
    plt.grid(True)
    plt.savefig('q_learning_training.png')
    
    # 测试训练好的智能体
    print("\n测试训练好的智能体...")
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # 使用学到的策略（不再探索）
        discrete_state = agent.discretize_state(state)
        action = np.argmax(agent.q_table[discrete_state])
        
        # 执行动作
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        
        # 渲染
        env.render()
        time.sleep(0.05)
    
    print(f"测试回合奖励: {total_reward}")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
