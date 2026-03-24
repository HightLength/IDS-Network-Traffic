# NetworkTrafficEnv.py
import gym
import pandas as pd
import numpy as np
import torch
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class NetworkTrafficEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    count_true = []
    count_false = []
    action_list = []
    brute_accuracy = []
    time_steps = 0
    episodes = 0
    TN = 0
    FN = 0
    TP = 0
    FP = 0
    w1 = 1
    w2 = 1
    w3 = 2

    # 更新分类器池
    classifiers = ['RandomForest', 'DecisionTree', 'SVM', 'MLP', 'KNN', 'GradientBoosting']
    classifier_params = [
        # RandomForest参数
        [{'n_estimators': 50, 'max_depth': 10},
         {'n_estimators': 100, 'max_depth': 20},
         {'n_estimators': 150, 'max_depth': None}],

        # DecisionTree参数
        [{'max_depth': 5, 'criterion': 'gini'},
         {'max_depth': 10, 'criterion': 'entropy'},
         {'max_depth': 15, 'criterion': 'gini'}],

        # SVM参数
        [{'C': 0.1, 'kernel': 'linear'},
         {'C': 1.0, 'kernel': 'rbf'},
         {'C': 10.0, 'kernel': 'rbf'}],

        # MLP参数
        [{'hidden_layer_sizes': (64,), 'activation': 'relu'},
         {'hidden_layer_sizes': (128,), 'activation': 'relu'},
         {'hidden_layer_sizes': (64, 64), 'activation': 'relu'}],

        # KNN参数
        [{'n_neighbors': 3, 'weights': 'uniform'},
         {'n_neighbors': 5, 'weights': 'distance'},
         {'n_neighbors': 7, 'weights': 'uniform'}],

        # GradientBoosting参数
        [{'n_estimators': 50, 'learning_rate': 0.1},
         {'n_estimators': 100, 'learning_rate': 0.1},
         {'n_estimators': 150, 'learning_rate': 0.05}]
    ]

    def __init__(self, df: pd.DataFrame, episode):
        super(NetworkTrafficEnv, self).__init__()

        self.action_space = gym.spaces.Tuple([
            gym.spaces.Discrete(len(self.classifiers)),  # 分类器选择
            gym.spaces.Discrete(3)  # 每个分类器有3个参数选项
        ])

        # 计算状态维度
        num_features = df.shape[1] - 1
        state_dim = num_features + 2

        print(f"\n[NetworkTrafficEnv] 初始化:")
        print(f"  数据集形状: {df.shape}")
        print(f"  特征数: {num_features}")
        print(f"  状态维度: {state_dim}")

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(state_dim,), dtype=np.float32)

        self.df = df
        self.episodes = episode
        self.current_row_index = 0
        self.state_dim = state_dim

        # 初始化分类器模型字典
        self.models = {}
        self.initialize_models()

    def initialize_models(self):
        """初始化所有分类器模型"""
        for cls_idx, cls_name in enumerate(self.classifiers):
            self.models[cls_name] = {}
            for param_idx in range(len(self.classifier_params[cls_idx])):
                self.models[cls_name][param_idx] = None

    def train_classifier(self, classifier_idx, param_idx):
        """训练指定的分类器"""
        cls_name = self.classifiers[classifier_idx]
        params = self.classifier_params[classifier_idx][param_idx]

        # 创建分类器模型
        if cls_name == 'RandomForest':
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif cls_name == 'DecisionTree':
            model = DecisionTreeClassifier(**params, random_state=42)
        elif cls_name == 'SVM':
            model = SVC(**params, probability=True, random_state=42)
        elif cls_name == 'MLP':
            model = MLPClassifier(**params, random_state=42, max_iter=100)
        elif cls_name == 'KNN':
            model = KNeighborsClassifier(**params, n_jobs=-1)
        elif cls_name == 'GradientBoosting':
            model = GradientBoostingClassifier(**params, random_state=42)
        else:
            raise ValueError(f"未知的分类器: {cls_name}")

        # 使用部分数据进行训练
        train_size = min(2000, len(self.df) // 2)
        train_data = self.df.iloc[:train_size]

        # 检查标签是否包含所有类别
        unique_labels = train_data['label_encoded'].unique()
        num_classes = len(self.df['label_encoded'].unique())

        if len(unique_labels) < num_classes:
            # 如果训练数据中缺少某些类别，使用完整数据集
            train_data = self.df.sample(n=min(2000, len(self.df)), random_state=42)

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # 训练分类器
        model.fit(X_train, y_train)
        self.models[cls_name][param_idx] = model
        return model

    def predict_with_classifier(self, classifier_idx, param_idx, features):
        """使用指定的分类器进行预测"""
        cls_name = self.classifiers[classifier_idx]

        if self.models[cls_name][param_idx] is None:
            try:
                self.train_classifier(classifier_idx, param_idx)
            except Exception as e:
                print(f"训练分类器 {cls_name} 失败: {e}")
                # 返回默认预测（多数类）
                return 0

        model = self.models[cls_name][param_idx]
        try:
            prediction = model.predict([features])[0]
            return prediction
        except Exception as e:
            print(f"预测失败: {e}")
            return 0

    def step(self, action):
        if self.current_row_index >= len(self.df):
            observation = self.df.iloc[0][:-1].values.astype(np.float32)
            new = [0.0, 0.0]  # 准确率和召回率
            observation = np.concatenate([observation, new])
            return observation, 0.0, True, self.action_list, self.count_true, self.count_false

        correct_action = self.df.iloc[self.current_row_index]['label_encoded']
        classifier_idx, param_idx = action

        features = self.df.iloc[self.current_row_index][:-1].values.astype(np.float32)
        predicted_action = self.predict_with_classifier(classifier_idx, param_idx, features)

        # 更新统计
        if predicted_action == correct_action:
            if correct_action == 0:
                self.TN += 1
            else:
                self.TP += 1
        else:
            if correct_action == 0:
                self.FP += 1
            else:
                self.FN += 1

        # 计算性能指标
        total = self.TP + self.TN + self.FP + self.FN
        if total > 0:
            accuracy = float(self.TP + self.TN) / total
            precision = float(self.TP) / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0.0
            recall = float(self.TP) / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0.0
            far = float(self.FP) / (self.TN + self.FP) if (self.TN + self.FP) > 0 else 0.0
        else:
            accuracy = precision = recall = far = 0.0

        # 记录预测结果
        if predicted_action == correct_action:
            self.count_true.append(predicted_action)
        else:
            self.count_false.append(predicted_action)

        self.action_list.append(predicted_action)

        # 准备下一个状态
        next_row_index = self.current_row_index + 1
        if next_row_index < len(self.df):
            observation = self.df.iloc[next_row_index][:-1].values.astype(np.float32)
        else:
            observation = self.df.iloc[0][:-1].values.astype(np.float32)

        new = [accuracy, recall]
        observation = np.concatenate([observation, new])

        # 确保状态维度正确
        if len(observation) != self.state_dim:
            if len(observation) < self.state_dim:
                padding = np.zeros(self.state_dim - len(observation), dtype=np.float32)
                observation = np.concatenate([observation, padding])
            else:
                observation = observation[:self.state_dim]

        self.current_row_index += 1
        self.time_steps += 1
        done = self.current_row_index >= len(self.df)

        # 计算奖励
        reward = float(self.w1 * accuracy + self.w2 * recall + self.w3*precision - far)

        # 重置统计（如果到达episode边界）
        if self.time_steps == self.episodes or done:
            if correct_action == 2:  # 如果是Brute_Force攻击
                self.brute_accuracy.append(accuracy)
            self.time_steps = 0
            self.TP = 0
            self.FP = 0
            self.FN = 0
            self.TN = 0

        return observation, reward, done, self.action_list, self.count_true, self.count_false

    def reset(self):
        """重置环境到数据集的起始位置"""
        self.current_row_index = 0
        self.count_true.clear()
        self.count_false.clear()
        self.action_list.clear()
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

        init = [0, 0]  # 初始准确率和召回率
        return np.concatenate([self.df.iloc[0][:-1].values, init])

    def reset_with_index(self, start_idx=None):
        """从指定索引重置环境"""
        if start_idx is None:
            start_idx = np.random.randint(0, len(self.df) - 1000)  # 留出足够的空间

        self.current_row_index = start_idx
        self.count_true.clear()
        self.count_false.clear()
        self.action_list.clear()
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        self.time_steps = 0

        init = [0, 0]  # 初始准确率和召回率

        # 确保索引在范围内
        if self.current_row_index >= len(self.df):
            self.current_row_index = len(self.df) - 1

        return np.concatenate([self.df.iloc[self.current_row_index][:-1].values, init])

    def get_random_batch_indices(self, batch_size):
        """获取随机的批次索引"""
        indices = np.random.choice(len(self.df), size=min(batch_size, len(self.df)), replace=False)
        return indices

    def get_batch_states(self, indices):
        """获取指定索引的状态"""
        states = []
        for idx in indices:
            if idx < len(self.df):
                state = np.concatenate([self.df.iloc[idx][:-1].values, [0, 0]])  # 初始指标为0
                states.append(state)
        return np.array(states)

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(f"当前行索引: {self.current_row_index}")
            print(f"正确预测: {len(self.count_true)}, 错误预测: {len(self.count_false)}")