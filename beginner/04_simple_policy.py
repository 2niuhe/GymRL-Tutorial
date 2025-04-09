#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gymnasium示例：实现简单的策略（以CartPole为例）
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def random_policy(observation):
    """随机策略：随机选择动作"""
    return np.random.randint(0, 2)

def simple_policy(observation):
    """简单规则策略：如果杆子倾向右边，向右移动；如果倾向左边，向左移动"""
    # 观察中的第3个值是杆子的角度
    pole_angle = observation[2]
    # 如果杆子倾向右边（正角度），选择动作1（向右移动）
    # 如果杆子倾向左边（负角度），选择动作0（向左移动）
    return 1 if pole_angle > 0 else 0

def run_episode(env, policy, render=False):
    """运行一个回合，使用给定的策略"""
    observation, info = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # 根据策略选择动作
        action = policy(observation)
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 检查回合是否结束
        done = terminated or truncated
        
        # 如果需要渲染
        if render:
            env.render()
    
    return total_reward

def main():
    # 创建环境
    env = gym.make("CartPole-v1", render_mode="human")
    
    # 比较不同策略
    n_episodes = 10
    
    # 运行随机策略
    print("运行随机策略...")
    random_rewards = []
    for i in range(n_episodes):
        reward = run_episode(env, random_policy, render=(i == n_episodes-1))
        random_rewards.append(reward)
        print(f"随机策略回合 {i+1}: 奖励 = {reward}")
    
    # 运行简单规则策略
    print("\n运行简单规则策略...")
    simple_rewards = []
    for i in range(n_episodes):
        reward = run_episode(env, simple_policy, render=(i == n_episodes-1))
        simple_rewards.append(reward)
        print(f"简单规则策略回合 {i+1}: 奖励 = {reward}")
    
    # 关闭环境
    env.close()
    
    # 绘制结果比较
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_episodes+1), random_rewards, 'o-', label='随机策略')
    plt.plot(range(1, n_episodes+1), simple_rewards, 'o-', label='简单规则策略')
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.title('不同策略的性能比较')
    plt.legend()
    plt.grid(True)
    plt.savefig('policy_comparison.png')
    plt.show()
    
    # 打印平均奖励
    print(f"\n随机策略平均奖励: {np.mean(random_rewards):.2f}")
    print(f"简单规则策略平均奖励: {np.mean(simple_rewards):.2f}")

if __name__ == "__main__":
    main()
