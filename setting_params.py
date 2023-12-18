#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:00:53 2023

@author: francesco
"""
from model import FFNN_model, ModelTrainer

# define size of the layer of the FFNN
input_shape = 1
hidden_shape = 56
output_shape = 3

# define the gamma value
gamma = 0.90
# define the model
policy_net = FFNN_model(input_shape, hidden_shape, output_shape)
target_net = FFNN_model(input_shape, hidden_shape, output_shape)

# define agent parameters
agent_params = {
    "gamma": gamma,
    "max_memory": 100000,
    "policy_net": policy_net,
    "target_net": target_net,
    "trainer": ModelTrainer(policy_net, target_net, lr=0.001, gamma=gamma),
    "batch_size": 100,
    "tot_episodes": 2000

}

# define epsilon decay params
eps_decay = {
    "EPS_START": 0.9,
    "EPS_END": 0.05,
    "EPS_DECAY": 3000
    }
