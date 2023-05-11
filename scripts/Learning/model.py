import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionNetwork(nn.Module):
    def __init__(self, obs_size, num_actions, num_joints):
        super(ActionNetwork, self).__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.num_joints = num_joints
        
        # Layers for processing observation space
        self.obs_fc1 = nn.Linear(obs_size, 64)
        self.obs_fc2 = nn.Linear(64, 64)
        
        # Layers for processing joint space
        self.joint_fc1 = nn.Linear(num_joints, 64)
        self.joint_fc2 = nn.Linear(64, 64)
        
        # Layers for discrete actions
        self.discrete_fc1 = nn.Linear(64, 32)
        self.discrete_fc2 = nn.Linear(32, num_actions)
        
        # Layers for continuous actions
        self.continuous_fc1 = nn.Linear(64, 32)
        self.continuous_fc2 = nn.Linear(32, num_joints)
        
    def forward(self, obs, joints):
        # Process observation space
        x = F.relu(self.obs_fc1(obs))
        x = F.relu(self.obs_fc2(x))
        
        # Process joint space
        y = F.relu(self.joint_fc1(joints))
        y = F.relu(self.joint_fc2(y))
        
        # Combine processed observation and joint space
        z = torch.cat((x, y), dim=1)
        
        # Calculate discrete actions
        d = F.relu(self.discrete_fc1(z))
        d = self.discrete_fc2(d)
        
        # Calculate continuous actions
        c = F.relu(self.continuous_fc1(z))
        c = self.continuous_fc2(c)
        
        return d, c
