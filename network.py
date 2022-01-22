import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, simple=False):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.simple = simple

        if self.simple:
            self.fc1 = nn.Linear(in_features=input_dims[0], out_features=64)
            self.fc2 = nn.Linear(in_features=64, out_features=128)
            self.fc3 = nn.Linear(in_features=128, out_features=256)
            self.fc4 = nn.Linear(in_features=256, out_features=512)
            self.fc5 = nn.Linear(in_features=512, out_features=n_actions)
        else:
            self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

            fc_input_dims = self.calculate_conv_output_dims(
                input_dims)  # Function to calculate the output dims of the conv layers (needed for the input to FC)

            self.fc1 = nn.Linear(fc_input_dims, 512)
            self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Returns the shape after the first 3 layers
    # Only performed at the beginning
    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)  # Add 1 for the batch
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(np.prod(dims.size()))  # Product of the dimensions (h,w) is the input number FC layer should accept

    def forward(self, state):
        if self.simple:
            # If the network is MLP
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            actions = self.fc5(x)
        else:
            # If the network is convolutional
            conv1 = F.relu(self.conv1(state))
            conv2 = F.relu(self.conv2(conv1))
            conv3 = F.relu(self.conv3(conv2))  # conv3 shape is BS x n_filters x H x W

            conv_state = conv3.view(conv3.size()[0], -1)  # Flatten conv output (similar to numpy's flatten)

            flat1 = F.relu(self.fc1(conv_state))
            actions = self.fc2(flat1)

        return actions

    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, simple=False):
        super().__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.simple = simple

        if self.simple:
            self.fc1 = nn.Linear(in_features=input_dims[0], out_features=64)
            self.fc2 = nn.Linear(in_features=64, out_features=128)
            self.fc3 = nn.Linear(in_features=128, out_features=256)
            self.fc4 = nn.Linear(in_features=256, out_features=512)

            self.V = nn.Linear(in_features=512, out_features=1)
            self.A = nn.Linear(in_features=512, out_features=n_actions)
        else:
            self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

            fc_input_dims = self.calculate_conv_output_dims(
                input_dims)  # Function to calculate the output dims of the conv layers (needed for the input to FC)

            self.fc1 = nn.Linear(fc_input_dims, 512)

            self.V = nn.Linear(512, 1)
            self.A = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # Returns the shape after the first 3 layers
    # Only performed at the beginning
    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)  # Add 1 for the batch
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        return int(np.prod(dims.size()))  # Product of the dimensions (h,w) is the input number FC layer should accept

    def forward(self, state):
        if self.simple:
            # If the network is MLP
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))

            V = self.V(x)
            A = self.A(x)
        else:
            # If the network is convolutional
            conv1 = F.relu(self.conv1(state))
            conv2 = F.relu(self.conv2(conv1))
            conv3 = F.relu(self.conv3(conv2))  # conv3 shape is BS x n_filters x H x W

            conv_state = conv3.view(conv3.size()[0], -1)  # Flatten conv output (similar to numpy's flatten)

            flat1 = F.relu(self.fc1(conv_state))
            V = self.V(flat1)
            A = self.A(flat1)
        return V, A

    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))