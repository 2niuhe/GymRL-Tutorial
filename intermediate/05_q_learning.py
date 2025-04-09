#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gymnasium示例：Q-Learning算法实现（以CartPole为例）
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import deque

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        # 环境相关
        self.env = env
        
        # 为CartPole定义状态空间离散化
        # 将连续的观察空间离散化为有限数量的状态
        # CartPole的观察空间包含4个值：[车位置, 车速度, 杆角度, 杆角速度]
        self.state_bins = [20, 20, 20, 20]  # 增加粒度，从10x10x10x10到20x20x20x20
        self.action_size = env.action_space.n
        
        # Q-learning参数
        self.learning_rate = learning_rate  # 学习率
        self.discount_factor = discount_factor  # 折扣因子
        self.exploration_rate = exploration_rate  # 探索率
        self.exploration_min = 0.01  # 最小探索率
        self.exploration_decay = exploration_decay  # 探索率衰减
        
        # 初始化Q表
        self.q_table = np.zeros((*self.state_bins, self.action_size))
        
        # 获取观察空间的边界
        self.obs_low = env.observation_space.low
        self.obs_high = env.observation_space.high
        
        # 处理无限边界
        self.obs_low[1] = -5.0  # 车速度下限
        self.obs_low[3] = -5.0  # 杆角速度下限
        self.obs_high[1] = 5.0  # 车速度上限
        self.obs_high[3] = 5.0  # 杆角速度上限
        
        # 添加经验回放
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
    
    def discretize_state(self, state):
        """将连续状态离散化"""
        discrete_state = []
        for i, (s, low, high, bins) in enumerate(zip(
                state, 
                self.obs_low, 
                self.obs_high,
                self.state_bins
            )):
            # 离散化
            scaled = int(np.floor((s - low) / (high - low) * bins))
            scaled = min(bins - 1, max(0, scaled))
            discrete_state.append(scaled)
            
        return tuple(discrete_state)
    
    def choose_action(self, state):
        """根据当前状态选择动作（ε-贪婪策略）"""
        # 探索：随机选择动作
        if np.random.random() < self.exploration_rate:
            return self.env.action_space.sample()
        
        # 利用：选择Q值最大的动作
        discrete_state = self.discretize_state(state)
        return np.argmax(self.q_table[discrete_state])
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放记忆"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size):
        """从经验回放中学习"""
        if len(self.memory) < batch_size:
            return
        
        # 随机采样一批经验
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            self.learn(state, action, reward, next_state, done)
    
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
    
    def train(self, episodes, render_interval=100):
        """训练智能体"""
        rewards = []
        episode_lengths = []
        best_reward = 0
        
        for episode in range(1, episodes + 1):
            # 重置环境
            state, _ = self.env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            # 渲染模式
            render = (episode % render_interval == 0)
            
            while not done:
                # 选择动作
                action = self.choose_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                steps += 1
                
                # 修改奖励以加速学习
                # 1. 如果杆子倒了，给予负奖励
                # 2. 根据杆子的角度给予奖励（越直立越好）
                # 3. 根据车的位置给予奖励（越居中越好）
                modified_reward = reward
                
                if terminated and steps < 500:  # 如果杆子倒了（游戏失败）
                    modified_reward = -10
                else:
                    # 杆子角度奖励（越直立越好）
                    angle = abs(next_state[2])  # 杆子角度的绝对值
                    angle_reward = 1.0 - angle * 2  # 角度越小，奖励越大
                    
                    # 车位置奖励（越居中越好）
                    position = abs(next_state[0])  # 车位置的绝对值
                    position_reward = 1.0 - position  # 位置越居中，奖励越大
                    
                    # 组合奖励
                    modified_reward = reward + 0.1 * angle_reward + 0.1 * position_reward
                
                # 存储经验
                self.remember(state, action, modified_reward, next_state, done)
                
                # 学习
                self.learn(state, action, modified_reward, next_state, done)
                
                # 从经验回放中学习
                if len(self.memory) > self.batch_size:
                    self.replay(self.batch_size)
                
                # 更新状态和奖励
                state = next_state
                total_reward += reward
                
                # 渲染
                if render and self.env.render_mode == "human":
                    self.env.render()
                    time.sleep(0.01)
            
            # 记录奖励和回合长度
            rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # 更新探索率（根据性能动态调整）
            if total_reward > best_reward:
                best_reward = total_reward
                # 如果有进步，减缓探索率的衰减
                self.exploration_rate = max(self.exploration_min, self.exploration_rate * 0.99)
            else:
                # 如果没有进步，加速探索率的衰减
                self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)
            
            # 打印进度
            if episode % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"回合: {episode}/{episodes}, 平均奖励: {avg_reward:.2f}, 平均步数: {avg_length:.2f}, 探索率: {self.exploration_rate:.4f}")
        
        return rewards, episode_lengths

def main():
    # 创建环境
    env = gym.make("CartPole-v1", render_mode="human")
    
    # 创建Q-learning智能体
    agent = QLearningAgent(
        env, 
        learning_rate=0.05,  # 降低学习率以提高稳定性
        discount_factor=0.99, 
        exploration_rate=1.0, 
        exploration_decay=0.99  # 减缓探索率衰减
    )
    
    # 训练智能体
    print("开始训练Q-learning智能体...")
    episodes = 300
    rewards, episode_lengths = agent.train(episodes)
    
    # 绘制训练过程中的奖励和回合长度变化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, episodes + 1), rewards)
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.title('Q-learning训练奖励')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, episodes + 1), episode_lengths)
    plt.xlabel('回合')
    plt.ylabel('回合长度（步数）')
    plt.title('Q-learning训练回合长度')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('q_learning_cartpole_training.png')
    
    # 测试训练好的智能体
    print("\n测试训练好的智能体...")
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done:
        # 使用学到的策略（不再探索）
        discrete_state = agent.discretize_state(state)
        action = np.argmax(agent.q_table[discrete_state])
        
        # 执行动作
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        steps += 1
        
        # 渲染
        env.render()
        time.sleep(0.01)
    
    print(f"测试回合奖励: {total_reward}, 步数: {steps}")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
