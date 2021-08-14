# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:47:57 2020

@author: William Woodall
"""


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import datetime
import os
import pickle
import gym

env = gym.make('CartPole-v1')

timestamp = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.isdir('models'):
    os.makedirs('models')
if not os.path.isdir('stats'):
    os.makedirs('stats')

class PPOMemory:
    def __init__(self):
        self.index = 0
        self.state = []
        self.action = []
        self.log_prob = []
        self.value = []
        self.reward = []
        self.done = []
        
    def update(self, state, action, log_prob, value, reward, done):
        self.state.append(state)
        self.action.append(action)
        self.log_prob.append(log_prob)
        self.value.append(value)
        self.reward.append(reward)
        self.done.append(1-done)
        
        self.index += 1
    
    
    def sample(self):
        return (self.state, self.action, self.log_prob, self.value, 
                self.reward, self.done)
    
    def generate_batches(self, batch_size, shuffle=True):
        mem_len = len(self.state)
        indicies = np.arange(mem_len)
        if shuffle:
            np.random.shuffle(indicies)
        batch_start = np.arange(0, mem_len, batch_size)
        batches = [indicies[i:i+batch_size] for i in batch_start]
        return batches
        
    def clear(self):
        self.index = 0
        self.state = []
        self.action = []
        self.log_prob = []
        self.value = []
        self.reward = []
        self.done = []
 
        
class PPONetwork(nn.Module):
    def __init__(self, input_dims, num_actions):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = 1024
        self.fc2_dims = 1024
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.dist = nn.Linear(self.fc2_dims, num_actions)
        self.q_val = nn.Linear(self.fc2_dims, 1)
        
        self.input_embedder = nn.Sequential(
            nn.Linear(self.input_dims, self.fc1_dims), nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),   nn.ReLU(),
            )
        
    def actor(self, x):
        x =self.input_embedder(x)
        dist = self.dist(x)
        dist = F.softmax(dist, dim=1)
        dist = torch.distributions.Categorical(dist)  
        return dist

    def critic(self, x):
        x = self.input_embedder(x)
        q_val = self.q_val(x)
        return q_val
    
  
class PPOAgent:
    def __init__(self, input_dims, num_actions):
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.epochs = 8
        self.mem_size = 1024
        self.batch_size = 256
        self.shuffle = False
        self.lr = 1e-4
        self.lr_decay = False
        self.lr_patience = 5_000
        self.gamma = .99
        self.tau = .95
        self.kl_clip = .2
        self.adv_norm = True

        self.learn_step = 0
   
        self.mem = PPOMemory()
       
        self.model = PPONetwork(self.input_dims, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        dist = self.model.actor(state)
        value = self.model.critic(state)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        action = action.detach().item()    
        log_prob = log_prob.detach().item()
        value = value.detach().item()
        return action, log_prob, value
        
    def evaluate(self, state): 
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        value_ = self.model.critic(state)
        return value_.item()
    
    def calculate_advantages(self, value, reward, value_, done):     
        gae = 0
        advantages = []
        value = value + [value_]
        for step in reversed(range(len(done))):
            delta = reward[step] + self.gamma * value[step+1] * done[step] - value[step]
            gae = delta + self.gamma * self.tau * done[step] * gae
            advantages.append(gae)
        return advantages[::-1]
    
    def learn(self):  
        if self.mem.index < self.mem_size:
            return [], [], []
        else:   
            #print('''learning...''')             
            (state, action, log_prob, value, reward, done) = self.mem.sample()
            value_ = self.evaluate(state[-1])
            advantage = self.calculate_advantages(value, reward, value_, done)
            
            state = torch.tensor(state).float().to(device)
            action = torch.tensor(action).to(device)
            value = torch.tensor(value).to(device)
            log_prob = torch.tensor(log_prob).to(device)
            advantage = torch.tensor(advantage).to(device)
            
            returns = advantage + value
            
            if self.shuffle:         
                batches = self.mem.generate_batches(self.batch_size, shuffle=True)
            else:
                batches = self.mem.generate_batches(self.batch_size, shuffle=False)
                
            for epoch in range(self.epochs):
                for batch in batches:
                    
                    _state = state[batch]
                    _action = action[batch]
                    _advantage = advantage[batch]
                    _log_prob = log_prob[batch]
                    _returns = returns[batch]
             
                    dist = self.model.actor(_state)
                    value_ = self.model.critic(_state)
                    
                    new_log_prob = dist.log_prob(_action)
                    entropy = dist.entropy().mean()
                    
                    ratio = torch.exp(new_log_prob - _log_prob)
                    
                    if self.adv_norm:
                        advantage = (advantage-advantage.mean()) / (advantage.std() + 1e-8)
                    
                    surr1 = ratio * _advantage
                    surr2 = torch.clamp(ratio, 1 - self.kl_clip, 1 + self.kl_clip) * _advantage 
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    value_ = value_.squeeze()
                    returns = returns.squeeze()
                    critic_loss = F.mse_loss(_returns, value_)

                    loss = .5 * critic_loss + actor_loss - .001 * entropy
                    
                    self.optimizer.zero_grad()  
                    loss.backward()
                    self.optimizer.step()
                    self.learn_step += 1
                    
                    if self.learn_step > 0 and not self.learn_step % self.lr_patience:
                        self.lr *= .1
            self.mem.clear()
            return actor_loss.item(), critic_loss.item(), entropy.item()
          
                

stats_look_back = 10
stats_dict = {'episode': [],
              'min': [],
              'max': [],
              'average': [],
              'moving average': [],
              'epsilon': [],
              'actor_lr': [],
              'critic_lr': [],
              'actor_loss': [],
              'critic_loss': [],
              'entropy': []
              }


agent = PPOAgent(4,2)

high_score = 0
score_history = []
loss_history = []

episode = 0
high_score = 0

_actor_loss = 0
_critic_loss = 0
_entropy = 0

stats_update = 1

steps = 0
while True:
    
    loss_ = 0
    score = 0
    done = False
    state = env.reset() 
    while not done:
        env.render() 
        action, log_prob, value = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        agent.mem.update(state, action, log_prob, value, reward, done)
        
        actor_loss, critic_loss, entropy = agent.learn()

        state = state_
        score += reward
        steps += 1
        
        if actor_loss:
            _actor_loss = actor_loss
        if critic_loss:
            _critic_loss = critic_loss
        if entropy:
            _entropy = entropy
            
    score_history.append(score)  
    avg_score = np.mean(score_history[-100:])
    high_score = max(high_score, score)
    
    
      
    if not episode % stats_update:
        stats_dict['episode'].append(episode)
        stats_dict['moving average'].append(avg_score)
        stats_dict['min'].append(np.min(score_history[-stats_look_back:]))
        stats_dict['max'].append(np.max(score_history[-stats_look_back:]))
        stats_dict['average'].append(np.mean(score_history[-stats_look_back:]))
        stats_dict['actor_lr'].append(agent.lr)
        stats_dict['critic_lr'].append(agent.lr)
        stats_dict['actor_loss'].append(_actor_loss) 
        stats_dict['critic_loss'].append(_critic_loss)   
        stats_dict['entropy'].append(_entropy) 
        
        with open('./stats/training_hist.pkl', 'wb') as f:
            pickle.dump(stats_dict, f)
        print(f"Episode: {episode}    Score: {score}   Avg: {np.round(avg_score,2)}    Max :{np.round(high_score,2)}  LR: {np.round(agent.lr,6)}   Learn Steps: {agent.learn_step}   Memory:{agent.mem.index}")
    episode += 1



    
    
    