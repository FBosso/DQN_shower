#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:22:58 2023

@author: francesco
"""

#import section
from collections import deque #data structure to store the memory
import random
import torch
import numpy as np
import math
from environment import Environment

from setting_params import eps_decay

#device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


EPS_START = eps_decay["EPS_START"]
EPS_END = eps_decay["EPS_END"]
EPS_DECAY = eps_decay["EPS_DECAY"]

env = Environment()


#definition of the agent class
class Agent():
    def __init__(self, gamma, max_memory, policy_network, target_network, trainer, batch_size, tot_episodes):
        self.n_episodes = 0
        self.gamma = gamma
        self.memory = deque(maxlen = max_memory)
        self.policy_network = policy_network.to(device)
        self.target_network = target_network.to(device)
        self.trainer = trainer
        self.batch_size = batch_size
        self.tot_episodes = tot_episodes
        self.n_steps_tot = 0
        
        #assign the same weights to the to different networks
        self.target_network.load_state_dict(self.policy_network.state_dict())
        
    def remember(self, state, action, reward, state_, done):
        self.memory.append((state, action, reward, state_, done))
        
    def get_n_episodes(self):
        return self.n_episodes
    
    def get_action(self, state):
        #increase number of steps by 1
        self.n_steps_tot += 1
        
        action_array = [0,0,0]
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.n_steps_tot / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                #return self.policy_network(state).max(1).indices.view(1, 1)
                idx = torch.argmax(self.policy_network(torch.tensor(state, dtype=torch.float32).to(device)))
                action_array[idx.item()] = 1
                
                return action_array, eps_threshold
        else:
            idx = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
            action_array[idx.item()] = 1
            
            return action_array, eps_threshold
        
    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.policy_network, self.target_network = self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.policy_network, self.target_network = self.trainer.train_step(state, action, reward, next_state, done)
        
    