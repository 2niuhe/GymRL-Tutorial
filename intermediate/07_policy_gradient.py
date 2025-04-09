#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
策略梯度方法（REINFORCE算法）训练CartPole
与DQN不同，策略梯度直接学习策略函数而非价值函数
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import random
import time

# 设置随机种子，以便结果可重现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}")

class PolicyNetwork(nn.Module):
    """策略网络：直接输出动作的概率分布"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # 添加批归一化
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)  # 添加批归一化
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # 使用Kaiming初始化改善训练
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc3.weight)  # 输出层使用Xavier初始化
    
    def forward(self, x):
        # 检查是否为单个样本（评估时）
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = self.fc1(x)
        # 只有在批量大于1时才使用BatchNorm
        if x.size(0) > 1:
            x = self.bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = F.relu(x)
        
        # 使用softmax确保输出是一个概率分布
        return F.softmax(self.fc3(x), dim=1)

class REINFORCEAgent:
    """REINFORCE算法的智能体实现"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        """初始化智能体参数和网络"""
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # 折扣因子
        self.initial_lr = learning_rate  # 保存初始学习率用于调度
        
        # 策略网络
        self.policy_network = PolicyNetwork(state_size, action_size).to(device)
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # 学习率调度器 - 随着训练进行逐渐降低学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=20, verbose=True
        )
        
        # 存储一个回合的轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []  # 存储动作的对数概率，避免重复计算
        
        # 添加奖励归一化的滑动统计
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_alpha = 0.05  # 滑动平均的更新率
        
        # 添加熵正则化系数
        self.entropy_beta = 0.01
    
    def choose_action(self, state):
        """根据当前策略选择动作"""
        state = torch.FloatTensor(state).to(device)
        # 获取动作概率分布
        probs = self.policy_network(state)
        # 根据概率分布采样动作
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        # 返回动作和其对数概率
        return action.item(), action_dist.log_prob(action)
    
    def remember(self, state, action, reward, log_prob):
        """记录状态、动作、奖励和对数概率"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)  # 存储对数概率
    
    def discount_rewards(self):
        """计算折扣累积奖励并应用基线"""
        discounted_rewards = []
        cumulative_reward = 0
        
        # 从后往前计算折扣累积奖励
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        # 转换为张量
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # 使用滑动平均更新奖励统计
        batch_mean = discounted_rewards.mean().item()
        batch_std = discounted_rewards.std().item() + 1e-9
        
        self.reward_mean = (1 - self.reward_alpha) * self.reward_mean + self.reward_alpha * batch_mean
        self.reward_std = (1 - self.reward_alpha) * self.reward_std + self.reward_alpha * batch_std
        
        # 使用滑动统计进行归一化，更稳定
        normalized_rewards = (discounted_rewards - self.reward_mean) / (self.reward_std + 1e-9)
        
        return normalized_rewards
    
    def learn(self):
        """策略梯度更新"""
        # 计算折扣累积奖励
        discounted_rewards = self.discount_rewards()
        
        # 转换为张量
        log_probs = torch.cat(self.log_probs).to(device)
        
        # 计算策略损失：负的期望奖励（目标是最大化期望奖励）
        policy_loss = -(log_probs * discounted_rewards.to(device)).mean()
        
        # 添加熵正则化以鼓励探索
        if len(self.states) > 1:
            states = torch.FloatTensor(np.array(self.states)).to(device)
            probs = self.policy_network(states)
            dist = torch.distributions.Categorical(probs)
            entropy = dist.entropy().mean()
            # 减去熵（因为我们想最大化熵）
            policy_loss -= self.entropy_beta * entropy
        
        # 优化模型
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # 清空回合记录
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        
        return policy_loss.item()
    
    def train(self, env, episodes, render_interval=100):
        """训练智能体"""
        scores = []
        avg_scores = []
        best_avg_score = -float('inf')
        
        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=RANDOM_SEED + episode)
            score = 0
            done = False
            
            # 收集一个回合的轨迹
            while not done:
                # 选择动作
                action, log_prob = self.choose_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 记录轨迹
                self.remember(state, action, reward, log_prob)
                
                state = next_state
                score += reward
            
            # 回合结束后学习
            loss = self.learn()
            
            # 记录得分
            scores.append(score)
            avg_score = np.mean(scores[-100:])  # 最近100回合的平均分
            avg_scores.append(avg_score)
            
            # 更新学习率调度器
            self.scheduler.step(avg_score)
            
            # 动态调整熵正则化系数
            if avg_score > 300:  # 当性能良好时，减少探索
                self.entropy_beta = max(0.001, self.entropy_beta * 0.995)
            
            # 打印训练进度
            if episode % 10 == 0:
                print(f"回合: {episode}/{episodes}, 得分: {score:.2f}, 平均得分: {avg_score:.2f}, 损失: {loss:.4f}")
            
            # 渲染一些回合以便观察
            if episode % render_interval == 0:
                eval_score = self.evaluate(env, render=True)
                # 如果性能提升，保存模型
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    torch.save(self.policy_network.state_dict(), "best_policy.pth")
        
        return scores, avg_scores
    
    def evaluate(self, env, episodes=1, render=False):
        """评估智能体性能"""
        total_reward = 0
        
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                if render:
                    env.render()  # 这会产生警告，但不影响功能
                    time.sleep(0.01)
                
                # 选择动作（评估时使用确定性策略）
                state_tensor = torch.FloatTensor(state).to(device)
                with torch.no_grad():  # 评估时不需要梯度
                    probs = self.policy_network(state_tensor).cpu().numpy()[0]
                action = np.argmax(probs)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
            
            total_reward += episode_reward
        
        avg_reward = total_reward / episodes
        if render:
            print(f"评估 {episodes} 回合的平均奖励: {avg_reward:.2f}")
        
        return avg_reward

def plot_training_results(scores, avg_scores, filename="policy_gradient_results.png"):
    """绘制训练结果"""
    plt.figure(figsize=(12, 6))
    plt.plot(scores, alpha=0.6, label='Score')
    plt.plot(avg_scores, label='Average Score', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Policy Gradient Training Results')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def main():
    """主函数"""
    # 创建环境，指定render_mode避免警告
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]  # 4
    action_size = env.action_space.n  # 2
    
    print(f"状态空间大小: {state_size}")
    print(f"动作空间大小: {action_size}")
    
    # 创建智能体
    agent = REINFORCEAgent(state_size, action_size)
    
    # 训练智能体
    episodes = 500
    print(f"开始训练 {episodes} 回合...")
    scores, avg_scores = agent.train(env, episodes)
    
    # 绘制训练结果
    plot_training_results(scores, avg_scores)
    
    # 评估智能体
    print("\n评估智能体性能...")
    # 创建一个专门用于渲染的环境
    eval_env = gym.make("CartPole-v1", render_mode="human")
    agent.evaluate(eval_env, episodes=5, render=True)
    
    # 关闭环境
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
