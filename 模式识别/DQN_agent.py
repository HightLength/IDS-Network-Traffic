# DQN_agent.py
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from agent.Experience_replay import ReplayBuffer, PrioritizedReplayBuffer
from network.fc3 import ClassifierSelector, ParameterSelector

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class HierarchicalDQN:
    def __init__(self, gamma, lr, state_dim, num_classifiers, num_params_per_classifier,
                 buffer_size, batch_size, init_epsilon, fin_epsilon, max_episode,
                 replace=1000, prioritized=False, alpha=0.6, beta=0.4):
        self.gamma = gamma
        self.lr = lr
        self.state_dim = state_dim
        self.num_classifiers = num_classifiers
        self.num_params_per_classifier = num_params_per_classifier
        self.prioritized = prioritized

        self.replace_target_cnt = replace
        self.learn_step_counter = 0

        # 选择经验回放缓冲区类型
        if prioritized:
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha, beta)
        else:
            self.memory = ReplayBuffer(buffer_size)

        self.batch_size = batch_size

        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.fin_epsilon = fin_epsilon
        self.max_episode = max_episode

        # 分类器选择网络
        self.classifier_q_eval = ClassifierSelector(lr, state_dim, num_classifiers)
        self.classifier_q_next = ClassifierSelector(lr, state_dim, num_classifiers)

        # 参数选择网络（每个分类器一个）
        self.param_qs_eval = []
        self.param_qs_next = []
        param_input_dim = state_dim + num_classifiers

        for _ in range(num_classifiers):
            param_q_eval = ParameterSelector(lr, param_input_dim, num_params_per_classifier)
            param_q_next = ParameterSelector(lr, param_input_dim, num_params_per_classifier)
            self.param_qs_eval.append(param_q_eval)
            self.param_qs_next.append(param_q_next)

        self.update_network_parameters()
        print(f"使用设备: {self.classifier_q_eval.device}")

    def update_network_parameters(self):
        for target_param, param in zip(self.classifier_q_next.parameters(), self.classifier_q_eval.parameters()):
            target_param.data.copy_(param.data)

        for i in range(self.num_classifiers):
            for target_param, param in zip(self.param_qs_next[i].parameters(), self.param_qs_eval[i].parameters()):
                target_param.data.copy_(param.data)

    def choose_action(self, observation):
        # 确保observation是float32
        observation = np.array(observation, dtype=np.float32).flatten()

        # 调整维度
        if len(observation) != self.state_dim:
            if len(observation) < self.state_dim:
                observation = np.concatenate(
                    [observation, np.zeros(self.state_dim - len(observation), dtype=np.float32)])
            else:
                observation = observation[:self.state_dim]

        state = T.tensor([observation], dtype=T.float32).to(self.classifier_q_eval.device)

        # 第一层：选择分类器
        if np.random.random() > self.epsilon:
            classifier_q_vals = self.classifier_q_eval.forward(state)
            classifier_action = T.argmax(classifier_q_vals).item()
        else:
            classifier_action = np.random.choice(self.num_classifiers)

        # 第二层：选择参数
        classifier_onehot = np.zeros(self.num_classifiers, dtype=np.float32)
        classifier_onehot[classifier_action] = 1

        combined_input = np.concatenate([observation, classifier_onehot])

        # 检查维度
        expected_dim = self.state_dim + self.num_classifiers
        if len(combined_input) != expected_dim:
            if len(combined_input) < expected_dim:
                combined_input = np.concatenate(
                    [combined_input, np.zeros(expected_dim - len(combined_input), dtype=np.float32)])
            else:
                combined_input = combined_input[:expected_dim]

        combined_tensor = T.tensor([combined_input], dtype=T.float32).to(self.classifier_q_eval.device)

        if np.random.random() > self.epsilon:
            param_q_vals = self.param_qs_eval[classifier_action].forward(combined_tensor)
            param_action = T.argmax(param_q_vals).item()
            if param_action >= self.num_params_per_classifier:
                param_action = self.num_params_per_classifier - 1
        else:
            param_action = np.random.choice(self.num_params_per_classifier)

        return (classifier_action, param_action)

    def store_transition(self, state, action, reward, state_, td_error=None):
        if self.prioritized and td_error is not None:
            self.memory.add(state, action, reward, state_, td_error)
        else:
            self.memory.add(state, action, reward, state_)

    def sample_memory(self):
        if self.memory.count < self.batch_size:
            return None, None, None, None, None, None, None

        if self.prioritized:
            state, action, reward, new_state, indices, weights = self.memory.sample_batch(self.batch_size)
            weights = T.tensor(weights, dtype=T.float32).to(self.classifier_q_eval.device)
        else:
            state, action, reward, new_state = self.memory.sample_batch(self.batch_size)
            indices = None
            weights = None

        # 将numpy数组转换为float32
        state = [np.array(s, dtype=np.float32) for s in state]
        new_state = [np.array(s, dtype=np.float32) for s in new_state]

        states = T.tensor(state, dtype=T.float32).to(self.classifier_q_eval.device)
        rewards = T.tensor(reward, dtype=T.float32).to(self.classifier_q_eval.device)
        states_ = T.tensor(new_state, dtype=T.float32).to(self.classifier_q_eval.device)

        # 处理动作
        classifier_actions = []
        param_actions = []
        for a in action:
            if isinstance(a, tuple) and len(a) == 2:
                classifier_actions.append(a[0])
                param_actions.append(a[1])
            else:
                classifier_actions.append(0)
                param_actions.append(0)

        classifier_actions = T.tensor(classifier_actions, dtype=T.long).to(self.classifier_q_eval.device)
        param_actions = T.tensor(param_actions, dtype=T.long).to(self.classifier_q_eval.device)

        return states, classifier_actions, param_actions, rewards, states_, indices, weights

    def learn(self):
        if self.memory.count < self.batch_size:
            return None, None

        result = self.sample_memory()
        if result[0] is None:
            return None, None

        states, classifier_actions, param_actions, rewards, states_, indices, weights = result
        batch_size = len(states)

        # 准备分类器one-hot编码
        classifier_onehots = T.zeros(batch_size, self.num_classifiers, dtype=T.float32).to(
            self.classifier_q_eval.device)
        classifier_onehots.scatter_(1, classifier_actions.unsqueeze(1), 1)

        # 计算目标Q值
        with T.no_grad():
            # 下一状态的分类器Q值
            next_classifier_q = self.classifier_q_next.forward(states_)
            next_classifier_vals, next_classifier_actions = next_classifier_q.max(1)

            # 下一状态的参数Q值
            next_classifier_onehots = T.zeros(batch_size, self.num_classifiers, dtype=T.float32).to(
                self.classifier_q_eval.device)
            next_classifier_onehots.scatter_(1, next_classifier_actions.unsqueeze(1), 1)
            combined_next = T.cat([states_, next_classifier_onehots], dim=1)

            next_param_vals = T.zeros(batch_size, dtype=T.float32).to(self.classifier_q_eval.device)
            for i in range(batch_size):
                classifier_idx = next_classifier_actions[i].item()
                if classifier_idx < len(self.param_qs_next):
                    param_q = self.param_qs_next[classifier_idx].forward(combined_next[i:i + 1])
                    next_param_vals[i] = param_q.max()

            q_target = rewards + self.gamma * (next_classifier_vals + next_param_vals)

        # 计算当前Q值
        classifier_q_pred = self.classifier_q_eval.forward(states)
        classifier_q_eval = classifier_q_pred[T.arange(batch_size), classifier_actions]

        combined_current = T.cat([states, classifier_onehots], dim=1)
        param_q_eval = T.zeros(batch_size, dtype=T.float32).to(self.classifier_q_eval.device)
        for i in range(batch_size):
            classifier_idx = classifier_actions[i].item()
            if classifier_idx < len(self.param_qs_eval):
                param_q = self.param_qs_eval[classifier_idx].forward(combined_current[i:i + 1])
                param_idx = param_actions[i].item()
                if param_idx < param_q.shape[1]:
                    param_q_eval[i] = param_q[0, param_idx]

        q_pred = classifier_q_eval + param_q_eval

        # 计算损失
        if self.prioritized and weights is not None:
            td_errors = (q_pred - q_target).detach().cpu().numpy()
            loss = (weights * (q_pred - q_target) ** 2).mean()
        else:
            td_errors = (q_pred - q_target).detach().cpu().numpy()
            loss = F.mse_loss(q_pred, q_target)

        # 更新网络
        self.classifier_q_eval.optimizer.zero_grad()
        for i in range(self.num_classifiers):
            self.param_qs_eval[i].optimizer.zero_grad()

        loss.backward()

        # 梯度裁剪 - 使用 T 而不是 torch
        T.nn.utils.clip_grad_norm_(self.classifier_q_eval.parameters(), 1.0)
        for i in range(self.num_classifiers):
            T.nn.utils.clip_grad_norm_(self.param_qs_eval[i].parameters(), 1.0)

        self.classifier_q_eval.optimizer.step()
        for i in range(self.num_classifiers):
            self.param_qs_eval[i].optimizer.step()

        # 更新优先经验回放的优先级
        if self.prioritized and indices is not None:
            self.memory.update_priorities(indices, td_errors)

        self.learn_step_counter += 1

        # 衰减探索率
        if self.max_episode > 0:
            epsilon_decay = (self.init_epsilon - self.fin_epsilon) / self.max_episode
            self.epsilon = max(self.fin_epsilon, self.epsilon - epsilon_decay)

        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.update_network_parameters()

        return loss.item(), td_errors.mean()

    def save_models(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.classifier_q_eval.save_checkpoint(f'{path}_classifier_eval')
        self.classifier_q_next.save_checkpoint(f'{path}_classifier_next')

        for i in range(min(self.num_classifiers, len(self.param_qs_eval))):
            self.param_qs_eval[i].save_checkpoint(f'{path}_param_{i}_eval')
            self.param_qs_next[i].save_checkpoint(f'{path}_param_{i}_next')

    def load_models(self, path):
        self.classifier_q_eval.load_checkpoint(f'{path}_classifier_eval')
        self.classifier_q_next.load_checkpoint(f'{path}_classifier_next')

        for i in range(min(self.num_classifiers, len(self.param_qs_eval))):
            self.param_qs_eval[i].load_checkpoint(f'{path}_param_{i}_eval')
            self.param_qs_next[i].load_checkpoint(f'{path}_param_{i}_next')