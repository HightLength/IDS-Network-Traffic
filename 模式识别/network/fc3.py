import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class ClassifierSelector(nn.Module):
    def __init__(self, lr, input_dim, num_classifiers, chkpt_dir=None):
        super(ClassifierSelector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classifiers)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.save_file = chkpt_dir

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_checkpoint(self, save_file):
        print('... saving checkpoint ...')
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        T.save(self.state_dict(), save_file)

    def load_checkpoint(self, load_file):
        if os.path.exists(load_file):
            print('... loading checkpoint ...')
            self.load_state_dict(T.load(load_file, map_location=self.device))
        else:
            print(f'... checkpoint not found: {load_file} ...')


class ParameterSelector(nn.Module):
    def __init__(self, lr, input_dim, num_params, chkpt_dir=None):
        super(ParameterSelector, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_params)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.save_file = chkpt_dir

    def forward(self, combined_input):
        x = F.relu(self.fc1(combined_input))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_checkpoint(self, save_file):
        print('... saving parameter checkpoint ...')
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        T.save(self.state_dict(), save_file)

    def load_checkpoint(self, load_file):
        if os.path.exists(load_file):
            print('... loading parameter checkpoint ...')
            self.load_state_dict(T.load(load_file, map_location=self.device))
        else:
            print(f'... parameter checkpoint not found: {load_file} ...')