# 网络流量识别系统 (IDS) - 基于分层深度强化学习

## 项目概述

本项目实现了一个基于分层深度强化学习（Hierarchical DQN）的网络流量识别系统（Intrusion Detection System, IDS）。该系统通过智能选择不同的分类器和参数组合，实现对网络流量的高效分类和异常检测。

## 核心功能

- **分层强化学习**：使用两层DQN网络，分别用于选择分类器和对应参数
- **优先经验回放**：提高训练效率和稳定性
- **多分类器集成**：支持多种分类器和参数组合
- **实时评估**：训练过程中定期评估模型性能
- **详细日志**：生成训练过程的详细日志和可视化结果

## 技术栈

- **Python 3.8+**
- **PyTorch**：深度学习框架
- **NumPy**：数值计算
- **Pandas**：数据处理
- **Matplotlib**：结果可视化
- **tqdm**：训练进度显示

## 项目结构

```
├── agent/              # 智能体相关代码
│   ├── Experience_replay.py  # 经验回放缓冲区
│   └── NetworkTrafficEnv.py  # 网络流量环境
├── network/            # 网络模型
│   └── fc3.py          # 分类器和参数选择网络
├── dataset/            # 数据集
│   ├── IDS_train.csv   # 训练数据
│   └── IDS_sample_test.csv  # 测试数据
├── checkpoints/        # 模型检查点
├── logs/               # 训练日志和结果
├── DQN_agent.py        # 分层DQN智能体实现
├── train_main.py       # 训练主脚本
├── model_test_main.py  # 模型测试脚本
└── README.md           # 项目说明
```

## 安装说明

1. **克隆项目**

2. **安装依赖**
   ```bash
   pip install torch numpy pandas matplotlib tqdm
   ```

3. **准备数据**
   - 将训练数据放置在 `dataset/IDS_train.csv`
   - 将测试数据放置在 `dataset/IDS_sample_test.csv`

## 快速开始

### 训练模型

```bash
python train_main.py
```

训练过程中会：
- 加载训练数据
- 初始化环境和智能体
- 执行训练循环
- 定期评估模型性能
- 保存最佳模型和训练日志

### 测试模型

```bash
python model_test_main.py
```

测试过程中会：
- 加载训练好的模型
- 评估模型在测试数据上的性能
- 生成混淆矩阵和测试结果

## 核心算法

### 分层DQN

该系统使用分层深度Q网络（Hierarchical DQN），包含两个层次：

1. **分类器选择层**：选择最合适的分类器
2. **参数选择层**：为选定的分类器选择最佳参数组合

这种分层结构允许智能体在复杂的决策空间中高效地找到最优策略。

### 优先经验回放

系统实现了优先经验回放机制，根据TD误差的大小为经验分配优先级，提高了训练效率和稳定性。

## 超参数设置

在 `train_main.py` 中可以调整以下超参数：

- `lr`：学习率
- `gamma`：折扣因子
- `init_epsilon`：初始探索率
- `fin_epsilon`：最终探索率
- `total_iterations`：总训练步数
- `batch_size`：批处理大小
- `prioritized`：是否使用优先经验回放

## 结果评估

训练完成后，系统会生成以下结果：

- **训练日志**：保存在 `logs/` 目录下
- **训练曲线**：包含奖励、准确率和损失的变化趋势
- **评估结果**：包含模型在测试数据上的性能
- **混淆矩阵**：展示分类结果的详细情况

## 模型部署

训练完成后，最佳模型会保存在 `checkpoints/` 目录下，可以用于实际的网络流量识别任务。

## 扩展和改进

- **添加更多分类器**：在 `NetworkTrafficEnv.py` 中添加新的分类器
- **调整网络结构**：修改 `network/fc3.py` 中的网络结构
- **优化奖励函数**：在 `NetworkTrafficEnv.py` 中调整奖励函数
- **尝试其他强化学习算法**：如DQN的变体（Double DQN、Dueling DQN等）

## 注意事项

- 确保数据集格式正确，标签列名为 `label_encoded`
- 训练过程可能需要较长时间，建议使用GPU加速
- 调整超参数以获得最佳性能

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请联系项目维护者。
