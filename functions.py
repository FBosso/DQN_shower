#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:51:29 2023

@author: francesco
"""

#import section
from agent import Agent
from environment import Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import torch


from setting_params import agent_params

gamma = agent_params["gamma"]
max_memory = agent_params["max_memory"]
policy_network = agent_params["policy_net"]
target_network = agent_params["target_net"]
trainer = agent_params["trainer"]
batch_size = agent_params["batch_size"]
tot_episodes = agent_params["tot_episodes"]

device = "cuda" if torch.cuda.is_available() else "cpu"

#define training function
def train():
    plot_rewards = []
    tot_reward = 0
    record = 0
    global_reward = 0
    agent = Agent(gamma, max_memory, policy_network, target_network, trainer, batch_size, tot_episodes)
    environment = Environment()
    
    while True:
        #get the old state
        #state = torch.tensor(environment.state, dtype=torch.float32).to(device)
        state = environment.state
        #get action
        action, eps_threshold = agent.get_action(state)        
        #perform move and get new state
        state_, reward, done, info = environment.step(action)
        
        #train short memory
        #agent.train_short_memory(state, action, reward, state_, done)
        #get number of episodes
        n_episodes = agent.get_n_episodes()
        #update the network every coupe of episodes
        if n_episodes % 2 == 0:
            #train network by extracting asmples from the replay buffer
            agent.train_long_memory()
            #smooth weighted update of the target network parameters
            agent.target_network = smooth_swap(agent.policy_network, agent.target_network, TAU=0.05)
        #remember
        agent.remember(state, action, reward, state_, done)
        #tot reward counter
        tot_reward += reward
        if done:
            #train the long memory (this is called experience relpay)
            environment.reset()
            #increase the number of episodes by one
            agent.n_episodes += 1
            agent.train_long_memory()
            agent.target_network = smooth_swap(agent.policy_network, agent.target_network, TAU=0.05)
            if tot_reward > record:
                record = tot_reward
                agent.policy_network.save()
            plot_rewards.append(tot_reward)
            global_reward += tot_reward
            
            avg_rew = plot(plot_rewards)
            print(f"Episode {agent.n_episodes}, Reward {tot_reward}, Record {record}, Average {avg_rew}, Eps {eps_threshold}")
            tot_reward = 0

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
            
def plot(plot_rewards):
    x_rew = [i for i in range(len(plot_rewards))]
    plt.plot(x_rew,plot_rewards, label="Episodic reward")
    avg_rews = moving_average(plot_rewards, 10)
    x_avg_rew = [i for i in range(len(avg_rews))]
    plt.plot(x_avg_rew,avg_rews, label="Average reward (10 episodes)")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()
    
    return avg_rews[-1]

    
def get_epsilon(tot_episodes,episode_cardinality,expoit_vs_explor=0.5,slope_change=0.1,stepness=0.1):
    '''
    Parameters
    ----------
    episode_cardinality : TYPE int
        DESCRIPTION. total number of episodes
    episode_cardinality : TYPE int
        DESCRIPTION. integer related to the cardinality of the current episode
    expoit_vs_explor : TYPE float [0,1]
        DESCRIPTION. number that indicates if to focus more on the exploration
        or on the exploitation part. If lower the inflection point will occur 
        earlier. If higher it will occur leater. It can be interpreted as the 
        percentage of the total training time that we want to allocate to 
        explotarion (e.g. : 0.3 --> 30% exploration; 0.7: 70% exploration)
    slope_change : TYPE float [0,1]
        DESCRIPTION. it determines the slope of transition region between 
        Exploration and Exploitation zone. The higher the smoother is the
        tranzision over time
    stepness : TYPE float  [0,1]
        DESCRIPTION. steepness of left and right tail of the graph. Higher the 
        value of C, more steep are the left and right tail of the graph

    Returns
    -------
    epsilon : TYPE float
        DESCRIPTION. the epsilon value for exploration / exploitation

    '''
    standardized_time=(episode_cardinality-expoit_vs_explor*tot_episodes)/(slope_change*tot_episodes)
    cosh=np.cosh(math.exp(-standardized_time))
    epsilon=1.1-(1/cosh+(episode_cardinality*stepness/tot_episodes))
    
    return epsilon


def smooth_swap(policy_network, target_network, TAU):
    #extract the weights
    weights_policy = policy_network.state_dict()
    weights_target = target_network.state_dict()
    #compute the new target weights
    for key in weights_policy:
        weights_target[key] = weights_policy[key]*TAU + weights_target[key]*(1-TAU)
    target_network.load_state_dict(weights_target)
    
    return target_network
    
    
    
    
    

