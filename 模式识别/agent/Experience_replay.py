# Experience_replay.py
from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, s_next):
        # 确保状态是float32类型
        s = np.array(s, dtype=np.float32)
        s_next = np.array(s_next, dtype=np.float32)
        experience = (s, a, r, s_next)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample_batch(self, batch_size):
        if self.count == 0:
            return [], [], [], []

        if self.count < batch_size:
            minibatch = random.sample(self.buffer, self.count)
        else:
            minibatch = random.sample(self.buffer, batch_size)

        state = [d[0] for d in minibatch]
        action = [d[1] for d in minibatch]
        reward = [d[2] for d in minibatch]
        next_state = [d[3] for d in minibatch]

        return state, action, reward, next_state

    def clear(self):
        self.buffer.clear()
        self.count = 0


class PrioritizedReplayBuffer:
    """优先经验回放"""

    def __init__(self, buffer_size, alpha=0.6, beta=0.4):
        self.buffer_size = buffer_size
        self.alpha = alpha  # 优先级程度
        self.beta = beta  # 重要性采样权重
        self.buffer = []
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        self.position = 0
        self.count = 0

    def add(self, s, a, r, s_next, td_error=1.0):
        priority = (abs(td_error) + 1e-6) ** self.alpha

        if self.count < self.buffer_size:
            self.buffer.append((s, a, r, s_next))
            self.priorities[self.position] = priority
            self.count += 1
        else:
            self.buffer[self.position] = (s, a, r, s_next)
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.buffer_size

    def sample_batch(self, batch_size):
        if self.count == 0:
            return [], [], [], [], [], []

        # 计算采样概率
        priorities = self.priorities[:self.count]
        probs = priorities / priorities.sum()

        # 采样索引
        indices = np.random.choice(self.count, size=min(batch_size, self.count), p=probs)

        # 计算重要性采样权重
        total = self.count
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        batch = [self.buffer[idx] for idx in indices]
        state = [d[0] for d in batch]
        action = [d[1] for d in batch]
        reward = [d[2] for d in batch]
        next_state = [d[3] for d in batch]

        return state, action, reward, next_state, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            if idx < self.count:
                self.priorities[idx] = (abs(error) + 1e-6) ** self.alpha