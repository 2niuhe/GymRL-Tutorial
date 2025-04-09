#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gymnasium基础示例：环境交互
"""
import gymnasium as gym
import time

def main():
    # 创建CartPole环境
    env = gym.make("CartPole-v1", render_mode="human")
    
    # 重置环境，获取初始观察
    observation, info = env.reset()
    
    # 打印环境信息
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"初始观察: {observation}")
    print(f"观察空间包含: [车位置, 车速度, 杆角度, 杆角速度]")
    
    # 执行随机动作的回合
    total_reward = 0
    for step in range(500):  # 最多执行500步
        # 随机选择动作
        action = env.action_space.sample()
        
        # 执行动作
        observation, reward, terminated, truncated, info = env.step(action)
        
        # 累计奖励
        total_reward += reward
        
        # 打印每一步的信息
        print(f"步骤 {step}: 动作={action}, 奖励={reward}, 累计奖励={total_reward}")
        print(f"观察: {observation}")
        
        # 如果回合结束，则退出循环
        if terminated or truncated:
            print(f"回合在 {step+1} 步后结束")
            break
        
        # 短暂暂停以便观察
        time.sleep(0.05)
    
    # 关闭环境
    env.close()
    print(f"回合总奖励: {total_reward}")

if __name__ == "__main__":
    main()
