#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gymnasium示例：深度Q网络 (DQN) 实现（以CartPole为例）
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# 设置随机种子以便结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 检查是否有可用的GPU
device_name = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
device = torch.device(device_name)
print(f"使用设备: {device_name}")

class DQN(nn.Module):
    """深度Q网络模型"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN智能体"""
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        
        # DQN参数
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min  # 最小探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰减
        self.batch_size = batch_size  # 批量大小
        
        # 创建Q网络和目标网络
        self.q_network = DQN(state_size, action_size).to(device)
        self.target_network = DQN(state_size, action_size).to(device)
        self.update_target_network()  # 初始化目标网络
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 损失函数
        self.criterion = nn.MSELoss()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def choose_action(self, state):
        """选择动作（ε-贪婪策略）"""
        if np.random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def learn(self):
        """从经验回放缓冲区中学习"""
        # 如果缓冲区中的样本不足，则不学习
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从缓冲区中采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # 计算当前Q值
        q_values = self.q_network(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = self.criterion(q_values, target_q_values)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train(self, env, episodes, target_update=10, render_interval=100):
        """训练智能体"""
        rewards = []
        losses = []
        
        for episode in range(1, episodes + 1):
            state, _ = env.reset()
            total_reward = 0
            done = False
            episode_losses = []
            
            # 渲染模式
            render = (episode % render_interval == 0)
            
            while not done:
                # 选择动作
                action = self.choose_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 存储经验
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # 学习
                loss = self.learn()
                if loss is not None:
                    episode_losses.append(loss)
                
                # 更新状态和奖励
                state = next_state
                total_reward += reward
                
                # 渲染
                if render and env.render_mode == "human":
                    env.render()
            
            # 更新目标网络
            if episode % target_update == 0:
                self.update_target_network()
            
            # 记录奖励和损失
            rewards.append(total_reward)
            if episode_losses:
                losses.append(np.mean(episode_losses))
            
            # 打印进度
            if episode % 10 == 0:
                avg_reward = np.mean(rewards[-10:])
                print(f"回合: {episode}/{episodes}, 平均奖励: {avg_reward:.2f}, 探索率: {self.epsilon:.4f}")
        
        return rewards, losses
    
    def save(self, filename):
        """保存模型"""
        torch.save(self.q_network.state_dict(), filename)
    
    def load(self, filename):
        """加载模型"""
        self.q_network.load_state_dict(torch.load(filename))
        self.update_target_network()

def main():
    # 创建环境
    env = gym.make("CartPole-v1", render_mode="human")
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 创建DQN智能体
    agent = DQNAgent(state_size, action_size)
    
    # 训练智能体
    print("开始训练DQN智能体...")
    episodes = 200
    rewards, losses = agent.train(env, episodes)
    
    # 保存模型
    agent.save("dqn_cartpole.pth")
    
    # 绘制训练过程
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, episodes + 1), rewards)
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.title('DQN训练奖励')
    plt.grid(True)
    
    if losses:
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel('回合')
        plt.ylabel('平均损失')
        plt.title('DQN训练损失')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('dqn_training.png')
    
    # 测试训练好的智能体
    print("\n测试训练好的智能体...")
    agent.epsilon = 0  # 测试时不探索
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # 使用学到的策略
        action = agent.choose_action(state)
        
        # 执行动作
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        
        # 渲染
        env.render()
    
    print(f"测试回合奖励: {total_reward}")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
