#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import pandas as pd
import torch as T
from tqdm import tqdm
from agent.NetworkTrafficEnv import NetworkTrafficEnv
from DQN_agent import DQN
from datetime import datetime
import matplotlib.pyplot as plt
now = datetime.now()
dt_string = now.strftime("_%d_%m_%H_%M")


if __name__ == '__main__':
    lr = 1e-3
    gamma = 0.98
    #init_epsilon = 0.99
    init_epsilon = 0.01
    fin_epsilon = 0.01
    iteration = 1
    buffer_size = 10000
    minimal_size = 320
    batch_size = 64
    target_update=250
    episode = 200
    device = T.device("cuda") if T.cuda.is_available() else T.device(
        "cpu")
    df = pd.read_csv(f'../dataset/IDS_test_real.csv')
    num_episodes = df.shape[0]
    max_episode=num_episodes * iteration*0.8
    env = NetworkTrafficEnv(df)
    random.seed(0)
    np.random.seed(0)
    T.manual_seed(1)
    state_dim = env.observation_space.shape[0]

    action_dim = 5
    agent = DQN(gamma,lr,action_dim,state_dim,buffer_size,batch_size,init_epsilon,fin_epsilon,int(max_episode),replace=target_update)
    agent.load_models(f'./checkpoints/DQN_resample')
    return_list = []
    loss_list = []
    for i in range(iteration):
        with tqdm(total=int(num_episodes), desc='Iteration %d' % i) as pbar:
            episode_return = 0
            state = env.reset()
            done = False
            step_count=0
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done ,info ,count_t,count_f = env.step(action)
                #next_state, reward, done, _, _ = env.step(action)
                agent.memory.add(state, action, reward, next_state)
                state = next_state
                # if agent.memory.count % 50 == 0:
                #     print(state.shape)

                episode_return += reward
                step_count +=1
                if step_count % episode == 0:
                    return_list.append(episode_return)
                    episode_return=0
                #(episode_return)
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if agent.memory.count > minimal_size:
                    agent.learn()
                    if step_count % episode ==0:
                        loss_list.append(agent.train_loss)

                    #replay_buffer.clear()


                    if (step_count ) % episode == 0:
                        pbar.set_postfix({
                            'acuracy':
                                '%.3f' % (len(count_t)/(len(count_f)+len(count_t))),
                            'return':
                                '%.3f' % np.mean(return_list[-10:])
                        })
                    if agent.memory.count == minimal_size + 1:
                        pbar.update(minimal_size)
                    else:
                        pbar.update(1)
    # 绘制训练曲线
    #平滑处理
    for i in range(len(return_list)):
        return_list[i]=np.mean(return_list[i:i+9])
    # agent.save_models(f'./checkpoints/dqn_model')
    plt.figure(figsize=(12, 5))
    # 奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(return_list)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.tight_layout()
    plt.show()

    plt.subplot(1 , 2 , 1)
    plt.plot(loss_list)
    plt.title('Training Loss')
    plt.xlabel('Update Step')
    plt.ylabel('MSE Loss')
    plt.tight_layout()
    plt.show()