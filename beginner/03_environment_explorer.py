#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gymnasium示例：多种环境的比较与探索
"""
import gymnasium as gym
import numpy as np
import time
from gymnasium.envs.registration import registry

def explore_environment(env_id, steps=100, render=True):
    """探索指定的环境"""
    try:
        # 创建环境
        render_mode = "human" if render else None
        env = gym.make(env_id, render_mode=render_mode)
        
        # 打印环境信息
        print(f"\n环境: {env_id}")
        print(f"观察空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")
        
        # 获取环境描述（如果有）
        env_spec = registry.get(env_id)
        if env_spec and hasattr(env_spec, 'description'):
            print(f"描述: {env_spec.description}")
        
        # 重置环境
        observation, info = env.reset()
        print(f"初始观察: {observation}")
        if hasattr(info, 'items') and len(info) > 0:
            print(f"初始信息: {info}")
        
        # 执行随机动作
        total_reward = 0
        for step in range(steps):
            # 随机选择动作
            action = env.action_space.sample()
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # 打印信息（仅打印前5步和最后一步）
            if step < 5 or (terminated or truncated):
                print(f"步骤 {step}: 动作={action}, 奖励={reward}, 累计奖励={total_reward}")
                if step < 5:
                    print(f"观察: {observation}")
            
            # 如果回合结束，则退出循环
            if terminated or truncated:
                print(f"回合在 {step+1} 步后结束")
                break
            
            # 如果渲染，短暂暂停以便观察
            if render:
                time.sleep(0.01)
        
        print(f"回合总奖励: {total_reward}")
        
        # 关闭环境
        env.close()
        return True
    
    except Exception as e:
        print(f"探索环境 {env_id} 时出错: {e}")
        return False

def list_environment_categories():
    """列出Gymnasium环境的类别"""
    categories = {}
    
    for env_id in registry.keys():
        # 跳过别名
        if env_id.startswith('ALE/'):
            continue
        
        # 提取类别
        category = env_id.split('/')[0] if '/' in env_id else env_id.split('-')[0]
        
        if category not in categories:
            categories[category] = []
        
        categories[category].append(env_id)
    
    return categories

def main():
    # 列出环境类别
    categories = list_environment_categories()
    print("Gymnasium环境类别:")
    for category, envs in categories.items():
        print(f"{category}: {len(envs)}个环境")
    
    # 选择一些典型环境进行探索
    environments_to_explore = [
        # 经典控制问题
        "CartPole-v1",
        "Pendulum-v1",
        "MountainCar-v0",
        "Acrobot-v1",
        
        # Box2D物理环境
        "LunarLander-v2",
        "BipedalWalker-v3",
        
        # Atari游戏（如果安装了gym[atari]）
        "ALE/Breakout-v5",
        
        # MuJoCo机器人环境（如果安装了gym[mujoco]）
        "HalfCheetah-v4",
        
        # 其他有趣的环境
        "FrozenLake-v1",
        "Taxi-v3"
    ]
    
    # 探索环境
    successful_envs = []
    for env_id in environments_to_explore:
        print(f"\n{'='*50}")
        print(f"探索环境: {env_id}")
        if explore_environment(env_id, steps=200):
            successful_envs.append(env_id)
        print(f"{'='*50}")
        
        # 暂停一下，以便观察
        time.sleep(1)
    
    # 打印成功探索的环境
    print(f"\n成功探索的环境: {len(successful_envs)}/{len(environments_to_explore)}")
    for env_id in successful_envs:
        print(f"- {env_id}")

if __name__ == "__main__":
    main()
