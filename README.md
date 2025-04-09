# GymRL-Tutorial: 强化学习入门指南 🤖

<div align="center">
  
![GitHub stars](https://img.shields.io/github/stars/2niuhe/GymRL-Tutorial?style=social)
![GitHub forks](https://img.shields.io/github/forks/2niuhe/GymRL-Tutorial?style=social)
![GitHub license](https://img.shields.io/github/license/2niuhe/GymRL-Tutorial)

</div>

<p align="center">
  <strong>通过实践学习强化学习 - 从基础概念到高级算法的系统教程</strong>
</p>

## 📖 项目简介

这个项目包含了一系列由浅入深的 Gymnasium（前身为 OpenAI Gym）示例代码，帮助你系统地学习强化学习环境交互和算法实现。每个示例都有详细注释，适合从零开始学习强化学习的初学者和希望巩固知识的进阶学习者。

## 🔍 什么是 Gymnasium？

Gymnasium 是一个用于开发和比较强化学习算法的标准 API。它提供了各种环境，让你能够训练智能体解决从简单到复杂的任务。Gymnasium 是 OpenAI Gym 的继承者，提供了更现代化的实现和更好的维护。

### 主要特点

- **标准化接口**：所有环境遵循相同的 API，便于算法开发和比较
- **多样化环境**：从简单的经典控制问题到复杂的 Atari 游戏和机器人控制
- **可扩展性**：可以创建自定义环境
- **与深度学习框架兼容**：易于与 PyTorch、TensorFlow 等框架集成

## 🗂️ 项目结构

```
GymRL-Tutorial/
├── beginner/     # 入门级示例
├── intermediate/ # 进阶级示例
├── advanced/     # 专家级示例
└── README.md     # 项目说明
```

## 🎓 学习路径

### 🔰 入门级 (Beginner)

适合强化学习初学者，介绍基本概念和简单实现。

<details>
<summary><b>✅ 已完成</b></summary>

1. **beginner/01_basic_interaction.py** - 基础环境交互
   - 学习如何创建环境、获取观察、执行动作和接收反馈
   - 使用 CartPole 环境演示基本概念

2. **beginner/02_spaces_exploration.py** - 理解观察和动作空间
   - 探索不同环境的观察空间和动作空间
   - 了解离散空间和连续空间的区别

3. **beginner/03_environment_explorer.py** - 多种环境的比较与探索
   - 探索 Gymnasium 提供的各种环境
   - 了解不同类型环境的特点和挑战

4. **beginner/04_simple_policy.py** - 实现简单的策略
   - 比较随机策略和基于规则的策略
   - 使用 CartPole 环境评估不同策略的性能
</details>

<details>
<summary><b>📝 待完成</b></summary>

5. **beginner/05_basic_visualization.py** - 强化学习基础可视化工具
   - 创建简单的可视化工具展示环境状态和智能体行为
   - 学习如何解释智能体的决策过程
   - 适合初学者的直观理解工具
</details>

### 🚀 进阶级 (Intermediate)

适合已了解基本概念的学习者，介绍经典算法和实现方法。

<details>
<summary><b>✅ 已完成</b></summary>

1. **intermediate/05_q_learning.py** - Q-Learning 算法实现
   - 使用表格型 Q-learning 解决经典控制问题
   - 学习状态离散化、ε-贪婪策略和 Q 值更新

2. **intermediate/06_deep_q_network.py** - 深度 Q 网络 (DQN) 实现
   - 使用神经网络代替 Q 表处理连续状态空间
   - 学习经验回放、目标网络等 DQN 核心概念
   - 在 CartPole 环境中训练深度强化学习模型

3. **intermediate/07_policy_gradient.py** - 策略梯度方法
   - 实现 REINFORCE 算法（蒙特卡洛策略梯度）
   - 学习直接优化策略而非值函数的方法
   - 在 CartPole 环境中训练策略网络
</details>

<details>
<summary><b>📝 待完成</b></summary>

4. **intermediate/08_actor_critic.py** - Actor-Critic 方法
   - 结合策略梯度和值函数近似的优势
   - 实现 A2C (Advantage Actor-Critic) 算法
   - 减少策略梯度的方差，提高训练稳定性

5. **intermediate/09_exploration_strategies.py** - 探索与利用的平衡
   - 深入探讨强化学习中的核心挑战
   - 实现 UCB (Upper Confidence Bound)、Thompson 采样等探索策略
   - 比较不同探索策略在各种环境中的效果

6. **intermediate/10_hyperparameter_tuning.py** - 超参数调优
   - 系统地调整强化学习算法的超参数
   - 实现网格搜索或随机搜索
   - 学习如何评估不同参数设置的性能
</details>

### 🔥 专家级 (Advanced)

适合有一定强化学习经验的学习者，介绍高级算法和复杂应用。

<details>
<summary><b>✅ 已完成</b></summary>

1. **advanced/08_custom_environment.py** - 自定义环境创建与训练
   - 学习如何从头创建符合 Gymnasium 接口的自定义环境
   - 实现简单的寻宝游戏环境
   - 使用 Q-learning 在自定义环境中训练智能体
   - 使用 Pygame 进行可视化渲染
</details>

<details>
<summary><b>📝 待完成</b></summary>

2. **advanced/09_model_based_rl.py** - 基于模型的强化学习
   - 学习环境模型并用于规划
   - 实现 Dyna-Q 等算法
   - 比较基于模型和无模型方法的优缺点

3. **advanced/10_multi_agent_rl.py** - 多智能体强化学习
   - 创建简单的多智能体环境
   - 实现独立 Q 学习或集中式训练分散式执行
   - 探索合作与竞争场景

4. **advanced/11_hierarchical_rl.py** - 分层强化学习
   - 学习将复杂任务分解为子任务
   - 实现 Options 框架或分层策略
   - 解决长期依赖问题

5. **advanced/12_advanced_visualization.py** - 高级强化学习可视化与分析
   - 创建高级训练过程、奖励、Q 值等可视化工具
   - 使用 Tensorboard 或自定义可视化
   - 深入分析智能体行为和学习过程
</details>

## 🚀 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv rl-env
source rl-env/bin/activate  # Linux/Mac
# 或
.\rl-env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行示例

每个示例都是独立的 Python 脚本，可以直接运行：

```bash
# 入门级示例
python beginner/01_basic_interaction.py
python beginner/02_spaces_exploration.py

# 进阶级示例
python intermediate/05_q_learning.py
python intermediate/06_deep_q_network.py

# 专家级示例
python advanced/08_custom_environment.py
```

## 📚 学习建议

1. 按照难度级别顺序学习，从入门级到专家级
2. 每个难度级别内的示例也建议按顺序学习
3. 每个示例都包含详细注释，帮助理解代码和概念
4. 尝试修改超参数和网络结构，观察对性能的影响
5. 将学到的算法应用到不同的环境中，加深理解
6. 完成一个难度级别后，可以尝试实现该级别的待完成内容

## 📖 资源链接

| 资源 | 链接 |
|------|------|
| Gymnasium 官方文档 | [gymnasium.farama.org](https://gymnasium.farama.org/) |
| 强化学习简介 - Sutton & Barto | [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html) |
| 深度强化学习课程 - David Silver | [YouTube 播放列表](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) |
| Spinning Up in Deep RL - OpenAI | [spinningup.openai.com](https://spinningup.openai.com/) |
| 强化学习：导论 - 周志华等 | [rl.qiwihui.com](https://rl.qiwihui.com/zh_CN/latest/) |

## 👥 贡献

欢迎通过 Issue 和 Pull Request 贡献代码、修复错误或提出改进建议。

## 📄 许可

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">
  <p>祝你学习愉快！🎉</p>
  <p>如果这个项目对你有帮助，请考虑给它一个 ⭐️</p>
</div>
