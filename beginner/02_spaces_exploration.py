#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gymnasium示例：探索不同环境的观察和动作空间
"""
import gymnasium as gym
import numpy as np

def explore_space(space):
    """探索并打印空间的详细信息"""
    space_type = type(space).__name__
    print(f"空间类型: {space_type}")
    
    if hasattr(space, 'shape'):
        print(f"形状: {space.shape}")
    
    if hasattr(space, 'n'):
        print(f"离散动作数量: {space.n}")
    
    if hasattr(space, 'high') and hasattr(space, 'low'):
        print(f"最小值: {space.low}")
        print(f"最大值: {space.high}")
    
    # 采样示例
    sample = space.sample()
    print(f"采样示例: {sample}")
    print("-" * 50)

def main():
    # 探索不同环境的空间
    environments = [
        "CartPole-v1",      # 离散动作空间，连续观察空间
        "Pendulum-v1",      # 连续动作空间，连续观察空间
        "MountainCar-v0",   # 离散动作空间，连续观察空间
        "Acrobot-v1",       # 离散动作空间，连续观察空间
    ]
    
    for env_name in environments:
        print(f"\n探索环境: {env_name}")
        env = gym.make(env_name)
        
        print("\n观察空间:")
        explore_space(env.observation_space)
        
        print("\n动作空间:")
        explore_space(env.action_space)
        
        # 获取初始观察示例
        observation, info = env.reset()
        print(f"初始观察示例: {observation}")
        print("=" * 70)
        
        env.close()

if __name__ == "__main__":
    main()
