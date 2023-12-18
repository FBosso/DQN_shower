#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:23:30 2023

@author: francesco
"""

#import section
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from setting_params import output_shape

#import variable to define the action space based on the output of the NN
action_space = output_shape

#define environment class
class Environment(Env):
    def __init__(self):
        #action to take: decrease_t, stay, increase_t
        self.action_space = Discrete(action_space)
        self.action_values = [-1,0,1]
        #temperature array
        self.observation_space = Box(low=np.array([10]), high=np.array([50]))
        #set start temp
        self.state = [38 + random.randint(-3,3)]
        #set shower length
        self.shower_length = 60
    
    def step(self,action):
        #   -1--> decrease;   0 --> stay   1-->increase
        action = np.argmax(action)
        #select action
        action = np.array(self.action_values[action])
        self.state = self.state + action
        #reduce the shower length by 1 sec
        self.shower_length -= 1
        
        #calculate reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1
            
        #check if the shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False
            
        #apply temperature noise (simulating flush of toilet, etc ...)
        self.state = self.state + random.randint(-1,1)
        
        #set information variable
        info = {}
        
        return self.state, reward, done, info
        
    def render(self):
        #for example with pygame
        pass
    def reset(self):
        #reset shower temperature
        self.state = [38 + random.randint(-3,3)]
        #restart the shower timer
        self.shower_length = 60
        
        return self.state