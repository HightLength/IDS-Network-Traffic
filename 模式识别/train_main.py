#!/usr/bin/env python3
import os
import warnings
import random
import numpy as np
import pandas as pd
import torch as T
from tqdm import tqdm
from agent.NetworkTrafficEnv import NetworkTrafficEnv
from DQN_agent import HierarchicalDQN
from datetime import datetime
import shutil
import matplotlib.pyplot as plt

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

now = datetime.now()
dt_string = now.strftime("_%d_%m_%H_%M")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_agent(agent, env, num_samples=1000):
    """定期评估智能体性能"""
    accuracy_list = []
    reward_list = []

    # 随机选择评估起点
    start_idx = np.random.randint(0, len(env.df) - num_samples - 100)
    state = env.reset_with_index(start_idx)

    agent.epsilon = 0.0  # 评估时使用贪婪策略

    for i in range(num_samples):
        action = agent.choose_action(state)
        next_state, reward, done, _, count_t, count_f = env.step(action)
        state = next_state

        reward_list.append(reward)

        if len(count_t) + len(count_f) > 0:
            accuracy = len(count_t) / (len(count_t) + len(count_f))
            accuracy_list.append(accuracy)

        if done or i == num_samples - 1:
            break

    agent.epsilon = agent.fin_epsilon  # 恢复探索率

    if accuracy_list:
        avg_accuracy = np.mean(accuracy_list)
        avg_reward = np.mean(reward_list)
    else:
        avg_accuracy = 0
        avg_reward = 0

    return avg_accuracy, avg_reward


def save_log(reward, accuracy, loss, parameter, pwd):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(reward)
    plt.xlabel("时间步")
    plt.ylabel("奖励")
    plt.title("训练奖励")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(accuracy)
    plt.xlabel("时间步")
    plt.ylabel("准确率")
    plt.title("训练准确率")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(loss)
    plt.xlabel("时间步")
    plt.ylabel("损失")
    plt.title("训练损失")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(pwd + "/training_summary.jpg", dpi=300)
    plt.close()

    # 保存数据到文件
    np.savetxt(pwd + "/reward.txt", reward)
    np.savetxt(pwd + "/accuracy.txt", accuracy)
    np.savetxt(pwd + "/loss.txt", loss)

    with open(pwd + "/parameter.txt", "w") as f:
        for key, value in parameter.items():
            f.write(f"{key}= {value}\n")


if __name__ == '__main__':
    # 超参数设置
    lr = 0.005
    gamma = 0.90
    init_epsilon = 0.8
    fin_epsilon = 0.01
    total_iterations = 25000  # 总训练步数
    warmup_steps = 2000  # 预热步数
    buffer_size = 100000
    minimal_size = 2000  # 最小经验数量
    batch_size = 128
    target_update = 100  # 目标网络更新频率
    eval_frequency = 2000  # 评估频率
    save_frequency = 5000  # 保存模型频率

    # 是否使用优先经验回放
    prioritized = True
    alpha = 0.6
    beta = 0.4

    # 设备设置
    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
    print(f"使用设备: {device}")
    if T.cuda.is_available():
        print(f"GPU名称: {T.cuda.get_device_name(0)}")
        print(f"GPU内存: {T.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 加载训练数据
    filename = 'dataset/IDS_train.csv'
    df = pd.read_csv(filename)

    # 确保标签列是整数类型
    if 'label_encoded' in df.columns:
        df['label_encoded'] = df['label_encoded'].astype(int)

    print(f"数据集形状: {df.shape}")
    print(f"特征数: {df.shape[1] - 1}")
    print(f"类别分布:\n{df.iloc[:, -1].value_counts()}")

    # 创建环境
    env = NetworkTrafficEnv(df, episode=32)
    random.seed(36)
    np.random.seed(36)
    T.manual_seed(36)

    # 获取状态维度和分类器数量
    state_dim = env.observation_space.shape[0]
    num_classifiers = len(env.classifiers)
    num_params_per_classifier = 3

    print(f"\n=== 环境设置 ===")
    print(f"状态维度: {state_dim}")
    print(f"分类器数量: {num_classifiers}")
    print(f"每个分类器的参数选项: {num_params_per_classifier}")

    # 创建智能体
    agent = HierarchicalDQN(
        gamma=gamma,
        lr=lr,
        state_dim=state_dim,
        num_classifiers=num_classifiers,
        num_params_per_classifier=num_params_per_classifier,
        buffer_size=buffer_size,
        batch_size=batch_size,
        init_epsilon=init_epsilon,
        fin_epsilon=fin_epsilon,
        max_episode=total_iterations,
        replace=target_update,
        prioritized=prioritized,
        alpha=alpha,
        beta=beta
    )

    # 创建日志目录
    pwd = os.path.join(os.getcwd(), 'logs', dt_string)
    os.makedirs(pwd, exist_ok=True)

    # 训练循环
    return_list = []
    loss_list = []
    accuracy_list = []
    eval_results = []

    total_steps = 0
    best_accuracy = 0

    print(f"\n=== 开始训练 ===")
    print(f"总训练步数: {total_iterations}")
    print(f"优先经验回放: {prioritized}")

    with tqdm(total=total_iterations, desc='训练进度') as pbar:
        while total_steps < total_iterations:
            # 随机选择起始位置
            start_idx = np.random.randint(0, len(df) - 2000)
            state = env.reset_with_index(start_idx)

            episode_done = False
            episode_steps = 0
            episode_reward = 0

            while not episode_done and total_steps < total_iterations:
                total_steps += 1
                episode_steps += 1

                # 选择动作
                action = agent.choose_action(state)

                # 执行动作
                next_state, reward, done, _, count_t, count_f = env.step(action)

                # 存储经验
                agent.store_transition(state, action, reward, next_state)
                state = next_state
                episode_reward += reward

                # 定期学习（预热阶段后）
                if agent.memory.count > minimal_size and total_steps % 4 == 0:
                    loss, td_error = agent.learn()
                    if loss is not None:
                        loss_list.append(loss)

                # 定期记录进度
                if total_steps % 100 == 0:
                    if len(count_t) + len(count_f) > 0:
                        accuracy = len(count_t) / (len(count_t) + len(count_f))
                        accuracy_list.append(accuracy)

                    # 更新进度条
                    avg_reward = np.mean(return_list[-10:]) if return_list else 0
                    avg_accuracy = np.mean(accuracy_list[-10:]) if accuracy_list else 0

                    pbar.set_postfix({
                        '准确率': f'{avg_accuracy:.3f}',
                        '奖励': f'{avg_reward:.3f}',
                        'ε': f'{agent.epsilon:.3f}',
                        '经验': f'{agent.memory.count}'
                    })

                # 定期评估
                if total_steps % eval_frequency == 0 and agent.memory.count > minimal_size:
                    eval_accuracy, eval_reward = evaluate_agent(agent, env, 500)
                    eval_results.append((total_steps, eval_accuracy, eval_reward))

                    print(f"\n评估结果 (步数: {total_steps}):")
                    print(f"  评估准确率: {eval_accuracy:.4f}")
                    print(f"  评估奖励: {eval_reward:.4f}")

                    # 保存最佳模型
                    if eval_accuracy > best_accuracy:
                        best_accuracy = eval_accuracy
                        agent.save_models('./checkpoints/HDQN_best')
                        print(f"  保存最佳模型，准确率: {best_accuracy:.4f}")

                # 定期保存模型
                if total_steps % save_frequency == 0 and total_steps > 0:
                    agent.save_models(f'./checkpoints/HDQN_step_{total_steps}')
                    print(f"  保存检查点模型，步数: {total_steps}")

                pbar.update(1)

                # 检查是否完成一个episode
                if done or episode_steps >= 1000:
                    episode_done = True
                    return_list.append(episode_reward)

                    # 衰减探索率
                    if agent.epsilon > fin_epsilon:
                        epsilon_decay = (init_epsilon - fin_epsilon) / total_iterations * 1000
                        agent.epsilon = max(fin_epsilon, agent.epsilon - epsilon_decay)

    # 训练完成后保存最终模型
    agent.save_models('./checkpoints/HDQN_final')

    # 复制模型到日志目录
    try:
        shutil.copy('./checkpoints/HDQN_best_classifier_eval', pwd)
        print(f"最佳模型已保存到: {pwd}")
    except Exception as e:
        print(f"复制模型失败: {e}")

    # 保存训练参数
    parameter = {
        'lr': lr,
        'gamma': gamma,
        'init_epsilon': init_epsilon,
        'fin_epsilon': fin_epsilon,
        'total_iterations': total_iterations,
        'warmup_steps': warmup_steps,
        'buffer_size': buffer_size,
        'minimal_size': minimal_size,
        'batch_size': batch_size,
        'target_update': target_update,
        'eval_frequency': eval_frequency,
        'save_frequency': save_frequency,
        'prioritized': prioritized,
        'alpha': alpha,
        'beta': beta,
        'device': str(device),
        'classifiers': ', '.join(env.classifiers),
        'file_name': filename,
        'state_dim': state_dim,
        'best_accuracy': best_accuracy,
        'final_epsilon': agent.epsilon
    }

    # 平滑曲线（可选）
    if len(return_list) > 100:
        window_size = 10
        return_smoothed = []
        for i in range(len(return_list)):
            start = max(0, i - window_size)
            end = min(len(return_list), i + window_size + 1)
            return_smoothed.append(np.mean(return_list[start:end]))
        return_list = return_smoothed

    # 保存日志
    save_log(reward=return_list, accuracy=accuracy_list, loss=loss_list,
             parameter=parameter, pwd=pwd)

    # 绘制评估结果
    if eval_results:
        eval_steps, eval_accuracies, eval_rewards = zip(*eval_results)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(eval_steps, eval_accuracies, 'b-o', linewidth=2, markersize=4)
        plt.xlabel("训练步数")
        plt.ylabel("评估准确率")
        plt.title("评估准确率 vs 训练步数")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(eval_steps, eval_rewards, 'r-o', linewidth=2, markersize=4)
        plt.xlabel("训练步数")
        plt.ylabel("评估奖励")
        plt.title("评估奖励 vs 训练步数")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(pwd + "/evaluation_results.jpg", dpi=300)
        plt.show()

    print(f"\n训练完成!")
    print(f"总训练步数: {total_steps}")
    print(f"最终探索率: {agent.epsilon:.4f}")
    print(f"最佳评估准确率: {best_accuracy:.4f}")
    print(f"经验回放缓冲区大小: {agent.memory.count}")
    print(f"日志保存在: {pwd}")