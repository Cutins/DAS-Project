import numpy
import torch
import torch.nn as nn
import networkx as nx
from types import *

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, layers):
        super().__init__()
        self.layers = []
        for _ in range(layers):
            self.layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class VGG(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.block_1 = ConvBlock(in_ch, 32, layers=2)
        self.block_2 = ConvBlock(32, 64, layers=2)

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = nn.AdaptiveAvgPool2d(output_size=(1,1))(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.fc3(x)
        return logits 
        
class Agent:
    def __init__(self, net, data, batch_size):
        self.net = net
        self.data = data
        self.batch_size = batch_size
        #self.local_estimate = torch.zeros_like(net.parameters())

    def update_state(self, weights, states):
        pass


        
class DistributedAgents:
    def __init__(self, n_agents, data: dict[list], batch_size=1):
        assert n_agents == len(data.keys()), "Number of agents and data splits must be the same!"

        # Communication graph
        #nx...

        # Create initial network
        self.net = VGG(in_ch=3, num_classes=2)

        # Create an initial copy of the network for each agent
        self.agents = {f'Agent_{n}': Agent(self.net, data[n], batch_size) for n in range(n_agents)}
