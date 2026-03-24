#!/usr/bin/env python3
import os
import random
import numpy as np
import pandas as pd
import torch as T
from tqdm import tqdm
from agent.NetworkTrafficEnv import NetworkTrafficEnv
from DQN_agent import HierarchicalDQN
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

now = datetime.now()
dt_string = now.strftime("_%d_%m_%H_%M")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_cm(true_list=[], action_list=[]):
    cm = confusion_matrix(true_list, action_list)
    return cm


def draw_cm(cm, classes=['BENIGN', 'Botnet', 'Brute_Force', 'Dos/DDos', 'Infiltration', 'PortScan', 'Web_Attack']):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=classes, yticklabels=classes)
    for label in ax.get_xticklabels():
        label.set_style("italic")

    for label in ax.get_yticklabels():
        label.set_style("italic")

    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix{dt_string}.png', dpi=300)
    plt.show()


def draw_cm_normalized(cm, classes=['BENIGN', 'Botnet', 'Brute_Force', 'Dos/DDos', 'Infiltration', 'PortScan',
                                    'Web_Attack']):
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap='Blues',
        vmin=0, vmax=1,
        xticklabels=classes,
        yticklabels=classes
    )

    for label in ax.get_xticklabels():
        label.set_style("italic")

    for label in ax.get_yticklabels():
        label.set_style("italic")

    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('归一化混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'normalized_confusion_matrix{dt_string}.png', dpi=300)
    plt.show()


def generate_static(true_list=[], action_list=[]):
    TN, FP, FN, TP = 0, 0, 0, 0
    for i in range(0, len(true_list)):
        if action_list[i] == true_list[i]:
            if true_list[i] == 0:
                TN += 1
            else:
                TP += 1
        else:
            if true_list[i] == 0:
                FP += 1
            else:
                FN += 1

    print(f"TN: {TN}, TP: {TP}, FN: {FN}, FP: {FP}")
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    far = FP / (TP + FP) if (TP + FP) > 0 else 0
    f1_s = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

    return accuracy, precision, recall, f1_s, far


def evaluate_model_on_subset(agent, env, subset_size=2000):
    """在数据子集上评估模型"""
    # 随机选择测试子集
    start_idx = np.random.randint(0, len(env.df) - subset_size)
    state = env.reset_with_index(start_idx)

    action_list = []
    true_list = []
    classifier_selections = []

    agent.epsilon = 0.0  # 测试时关闭探索

    for i in range(subset_size):
        action = agent.choose_action(state)
        classifier_idx, param_idx = action
        classifier_selections.append(classifier_idx)

        next_state, reward, done, predictions, count_t, count_f = env.step(action)
        state = next_state

        # 收集真实标签
        if env.current_row_index - 1 < len(env.df):
            true_label = env.df.iloc[env.current_row_index - 1]['label_encoded']
            true_list.append(true_label)

        if done:
            break

    agent.epsilon = agent.fin_epsilon  # 恢复探索率

    return action_list, true_list, classifier_selections


if __name__ == '__main__':
    # 超参数
    lr = 0.0005
    gamma = 0.99
    init_epsilon = 0.01
    fin_epsilon = 0.01
    iteration = 1
    buffer_size = 10000
    minimal_size = 100
    batch_size = 32
    target_update = 50
    episode = 32

    # 加载测试数据
    test_file = 'dataset/IDS_sample_test.csv'
    df = pd.read_csv(test_file)
    true_list = df.iloc[:, -1].tolist()

    print(f"测试集形状: {df.shape}")
    print(f"测试集类别分布:\n{df.iloc[:, -1].value_counts()}")

    # 创建环境和智能体
    env = NetworkTrafficEnv(df, episode)
    random.seed(36)
    np.random.seed(36)
    T.manual_seed(36)

    state_dim = env.observation_space.shape[0]
    num_classifiers = len(env.classifiers)
    num_params_per_classifier = 3

    agent = HierarchicalDQN(gamma, lr, state_dim, num_classifiers, num_params_per_classifier,
                            buffer_size, batch_size, init_epsilon, fin_epsilon,
                            iteration * len(df), replace=target_update)

    # 加载预训练模型
    try:
        agent.load_models('./checkpoints/HDQN_best')  # 加载最佳模型
        print("成功加载预训练模型")
    except Exception as e:
        print(f"加载最佳模型失败: {e}")
        try:
            agent.load_models('./checkpoints/HDQN_final')  # 尝试加载最终模型
            print("成功加载最终模型")
        except Exception as e:
            print(f"加载最终模型失败: {e}")
            print("将使用随机初始化的模型")

    # 测试循环
    return_list = []
    loss_list = []
    action_list = []
    classifier_selections = []
    accuracy_list = []

    # 在完整测试集上测试
    with tqdm(total=int(len(df)), desc='完整测试') as pbar:
        state = env.reset()
        done = False
        step_count = 0

        while not done and step_count < len(df):
            action = agent.choose_action(state)
            classifier_idx, param_idx = action
            classifier_selections.append(classifier_idx)

            next_state, reward, done, predictions, count_t, count_f = env.step(action)
            state = next_state
            step_count += 1

            if step_count % 100 == 0:
                total_predictions = len(count_t) + len(count_f)
                accuracy = len(count_t) / total_predictions if total_predictions > 0 else 0
                pbar.set_postfix({
                    'accuracy': '%.3f' % accuracy,
                    'step': step_count
                })

            pbar.update(1)

    # 收集最终的动作列表
    action_list = env.action_list

    # 生成性能报告
    print(f"\n预测数量: {len(action_list)}")
    print(f"真实标签数量: {len(true_list)}")

    if len(action_list) != len(true_list):
        min_len = min(len(action_list), len(true_list))
        action_list = action_list[:min_len]
        true_list = true_list[:min_len]
        print(f"调整后的长度: {min_len}")

    # 计算混淆矩阵和指标
    cm = generate_cm(true_list, action_list)
    draw_cm(cm)
    draw_cm_normalized(cm)

    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(true_list, action_list,
                                target_names=['BENIGN', 'Botnet', 'Brute_Force', 'Dos/DDos',
                                              'Infiltration', 'PortScan', 'Web_Attack']))

    accuracy, precision, recall, f1_s, far = generate_static(true_list, action_list)
    print("\n性能指标:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1_s:.4f}")
    print(f"误报率 (FAR): {far:.4f}")

    # 分析分类器选择分布
    if classifier_selections:
        print(f"\n分类器选择分布:")
        for i, cls_name in enumerate(env.classifiers):
            count = classifier_selections.count(i)
            percentage = count / len(classifier_selections) * 100
            print(f"  {cls_name}: {count}次 ({percentage:.2f}%)")

    # 可视化参数选择分布
    if hasattr(env, 'classifier_params'):
        print(f"\n参数选择分布:")
        for i, cls_name in enumerate(env.classifiers):
            param_counts = {}
            for j in range(len(env.classifier_params[i])):
                param_counts[j] = 0

            # 统计该分类器的参数选择
            for k in range(len(action_list)):
                if k < len(classifier_selections) and classifier_selections[k] == i:
                    # 这里需要从action_list中获取参数选择
                    # 注意：action_list存储的是预测标签，不是动作
                    # 需要修改环境来存储动作历史
                    pass

    # 可视化结果
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    if return_list:
        plt.plot(return_list)
    plt.title('测试奖励')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if env.brute_accuracy:
        plt.plot(env.brute_accuracy)
    plt.title('Brute Force攻击检测准确率')
    plt.xlabel('时步')
    plt.ylabel('准确率')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'test_results{dt_string}.png', dpi=300)
    plt.show()