#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:23:19 2023

@author: francesco
"""

#import section
import torch
from torch import nn
import os

#device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#define model class
class FFNN_model(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super().__init__()
        #specify layers
        self.l1 = nn.Linear(input_shape, hidden_shape)
        self.l2 = nn.Linear(hidden_shape, hidden_shape)
        self.l3 = nn.Linear(hidden_shape, output_shape)
        #specify activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        
        return x
    
    def save(self, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        

class ModelTrainer():
    def __init__(self, policy_network, target_network, lr, gamma):
        self.policy_network = policy_network
        self.target_network = target_network
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(),lr=self.lr)
        self.loss = nn.SmoothL1Loss()
        
    def train_step(self, state, action, reward, next_state, done):
        
        #convert all to tensors
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        
        if len(state.shape) == 1:
            #if shape is (x,) put it in (1,x)
            
            state = torch.unsqueeze(state, dim=0)
            action = torch.unsqueeze(action, dim=0)
            reward = torch.unsqueeze(reward, dim=0)
            next_state = torch.unsqueeze(next_state, dim=0)
            done = (done,)
            '''
            state = state.reshape(-1,1)
            action = action.reshape(-1,1)
            reward = torch.unsqueeze(reward, dim=0)
            next_state = next_state.reshape(-1,1)
            done = (done,)
            '''
        ### DO THE FORWARD PASS ###
        
        # 1) get the predicted Q value with the current state (Base)
        pred = self.policy_network(state)
        
        # 2) R(s,a) + gamma * max(Q(s',a'))
        target = pred.clone() #to avoid gradient errors
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.target_network(next_state[idx]))
        
            #select, from the cloned variable of the pure NN prediction,
            #the action maximizing the value
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        ### CALCULATE THE LOSS ###
        loss = self.loss(pred, target)
        
        ### OPTIMIZER ZERO GRAD ###
        self.optimizer.zero_grad()
        
        ### LOSS BACKWARD ###
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        
        ### OPTIMIZER STEP ###
        self.optimizer.step()
        
        return self.policy_network, self.target_network