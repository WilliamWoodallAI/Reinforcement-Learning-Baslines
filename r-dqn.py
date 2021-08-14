# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 00:39:37 2021

@author: William Woodall
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import pickle
import os
import gym 


env = gym.make('CartPole-v1')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.isdir('models'):
    os.makedirs('models')
if not os.path.isdir('stats'):
    os.makedirs('stats')


class ReplayMemmory():
    def __init__(self, mem_size, input_shape, hidden_size):
        self.mem_size = mem_size
        self.input_shape = input_shape
        self.states = np.zeros((mem_size, *self.input_shape))
        self.hidden_states = np.zeros((mem_size, hidden_size))
        self.actions = np.zeros(mem_size)
        self.rewards = np.zeros(mem_size)
        self.states_ = np.zeros((mem_size, *self.input_shape))
        self.hidden_states_ = np.zeros((mem_size, hidden_size))
        self.dones = np.zeros(mem_size)
        self.mem_counter = 0
        self.index = 0
        
    def update(self, hidden_state, state, action, reward, state_, hidden_state_, done):
        self.index = self.mem_counter % self.mem_size
        self.states[self.index] = state
        self.hidden_states[self.index] = hidden_state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.states_[self.index] = state_
        self.hidden_states_[self.index] = hidden_state_
        self.dones[self.index] = done
        self.mem_counter += 1
        
    def recall(self, batch_size):
        max_index = min(self.mem_size, self.mem_counter)
        indicies = np.random.choice(max_index, batch_size, replace=False)
        states = self.states[indicies]
        hidden_states = self.hidden_states[indicies]
        actions = self.actions[indicies]
        rewards = self.rewards[indicies]
        states_ = self.states_[indicies]
        hidden_states_ = self.hidden_states_[indicies]
        dones = self.dones[indicies]
        
        return hidden_states, states, actions, rewards, states_, hidden_states_, dones
    
    def clear_memmory(self):
        self.__init__(self.mem_size, self.input_dims)
        

class QNet(nn.Module):
    def __init__(self, input_dims, num_actions, hidden_size):
        super().__init__()  
        self.hidden_dims = hidden_size
        self.input_dims = input_dims + self.hidden_dims
        self.fc1_dims = 1024
        self.fc2_dims = 512
        self.backbone_dims = self.fc2_dims
        self.num_actions = num_actions
        
        self.input_embedding = nn.Sequential(
                            nn.Linear(self.input_dims, self.fc1_dims),   nn.ReLU(),
                            nn.Linear(self.fc1_dims,   self.fc2_dims),   nn.ReLU(),
                            ) 
        self.q_vals = nn.Linear(self.backbone_dims, self.num_actions)
        self.hidden = nn.Linear(self.backbone_dims, self.hidden_dims)
    
    def forward(self, state, hidden_state):
        x = torch.cat((state, hidden_state), dim=1)
        x = self.input_embedding(x)
        q_vals = self.q_vals(x)
        hidden_state_ = self.hidden(x)
        return q_vals, hidden_state_


class DQNAgent():
    def __init__(self, input_shape, num_actions):
        self.input_shape = input_shape
        self.input_dims = input_shape[0]
        self.num_actions = num_actions
        self.hidden_size = 64
        self.lr = .0001
        self.mem_size = 500_000
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = .3
        self.epsilon_decay = .00005
        self.epsilon_min = 0.1
        self.learn_step = 0
        
        self.mem = ReplayMemmory(self.mem_size, self.input_shape, self.hidden_size)
        self.model = QNet(self.input_dims, self.num_actions, self.hidden_size).to(device)
        self.target_model = QNet(self.input_dims, self.num_actions, self.hidden_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
   
        
    def choose_action(self, state, hidden_state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        hidden_state  = torch.tensor(hidden_state, dtype=torch.float).unsqueeze(0).to(device)
        q_vals, hidden_state_ = self.model(state, hidden_state)
        n = random.random() 
        if n < self.epsilon:
            action = np.random.choice(self.num_actions)
            return action, np.zeros(self.hidden_size)
        else:
            action = torch.argmax(q_vals).item()
        return action, hidden_state_.squeeze().cpu().detach().numpy()
    
    def remember(self, hidden_state, state, action, reward, state_, hidden_state_, done):
        self.mem.update(hidden_state, state, action, reward, state_, hidden_state_, done)
        
    def recall(self):
        hidden_states, states, actions, rewards, states_, hidden_states_, dones = self.mem.recall(self.batch_size)
        return hidden_states, states, actions, rewards, states_, hidden_states_, dones
    
    def lerp_target_params(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) 
    
    def update_target_params(self):
        self.target_model.load_state_dict(self.model.state_dict())
     
    def new_state(self):
        new_hidden_state = np.zeros(self.hidden_size)
        return new_hidden_state
    
    def learn(self):
        if self.mem.mem_counter < self.batch_size:
            return [], [], []
        else:
            #print("'''Agent Learning'''")
            hidden_states, states, actions, rewards, states_, hidden_states_, dones = self.recall()
            
            states = torch.tensor(states, dtype=torch.float32).to(device)
            hidden_states = torch.tensor(hidden_states, dtype=torch.float32).to(device)
            actions = torch.tensor(actions, dtype=torch.int64).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            states_ = torch.tensor(states_, dtype=torch.float32).to(device)
            hidden_states_ = torch.tensor(hidden_states_, dtype=torch.float32).to(device)
            dones = torch.tensor(dones, dtype=torch.bool).to(device)
            
            batch_indecies = np.arange(self.batch_size)
            
            q_vals, _ = self.model(states, hidden_states)
            q_vals_, _ = self.model(states_, hidden_states_)
            target_vals, _ = self.target_model(states_, hidden_states_)
            
            old_qs = q_vals[batch_indecies, actions]
            future_actions = torch.max(q_vals_, dim=1)[1]
            
            q_targets = target_vals[batch_indecies, future_actions]
            q_targets[dones] = 0.0 
            
            q_targets = rewards + q_targets * self.gamma
            #q_targets = (q_targets - q_targets.mean())/ q_targets.std()
            td = q_targets - old_qs
            
            self.optimizer.zero_grad()
            loss = (td**2).mean()
            loss.backward()
            self.optimizer.step()
            
            if not self.learn_step % 1:
                self.update_target_params()
            self.learn_step += 1
            
            return loss.item(), td.mean().item(), np.mean(target_vals.detach().cpu().numpy(), axis=0)  
            
            
agent = DQNAgent((4,),2) 

stats_update = 1
score_look_back = 10
stats_dict = {'episode': [],
              'min': [],
              'max': [],
              'average': [],
              'moving average': [],
              'epsilon': [],
              'learn_rate': [],
              'model_loss': [],
              'temporal_difference': [],
              }


q_val_dict = {'0': [],
              '1': [],
              }


high_score = 0
save_score = 0
score_history = []
loss_history = []

episode = 0
steps = 0

while True:
   
    loss_ = 0
    score = 0
    q_val_track = []
    state = env.reset()
    hidden_state = agent.new_state()
    done = False
    while not done:
        env.render()
        action, hidden_state_ = agent.choose_action(state, hidden_state)
        state_, reward, done, info = env.step(action)  
        agent.remember(hidden_state, state, action, reward, state_, hidden_state_, done)
        loss, td, q_vals = agent.learn()
        
        state = state_
        hidden_state = hidden_state_
        score += reward
        steps += 1
           
        agent.epsilon -= agent.epsilon_decay       
        agent.epsilon = max(agent.epsilon, agent.epsilon_min)
        
        if len(q_vals) > 0:
            q_val_track.append(q_vals)
            q_val_array = np.array(q_val_track)
            q_means = np.mean(q_val_array, axis=0)
            for i in range(len(q_val_dict)):
                q_val_dict[f'{i}'].append(q_means[i])
                q_val_dict[f'{i}'] = q_val_dict[f'{i}'][-500_000:]
                
    score_history.append(score)  
    avg_score = np.mean(score_history[-50:])
    high_score = max(high_score, avg_score)
    
    if not episode % stats_update:
        if td == []:
            td = 0
        if loss == []:
            loss = 0
        stats_dict['episode'].append(episode)
        stats_dict['moving average'].append(avg_score)
        stats_dict['min'].append(np.min(score_history[-score_look_back:]))
        stats_dict['max'].append(np.max(score_history[-score_look_back:]))
        stats_dict['average'].append(np.mean(score_history[-score_look_back:]))
        stats_dict['epsilon'].append(agent.epsilon)
        stats_dict['learn_rate'].append(agent.lr)
        stats_dict['model_loss'].append(loss)
        stats_dict['temporal_difference'].append(td)   
        
        with open('./stats/training_hist.pkl', 'wb') as f:
            pickle.dump(stats_dict, f)
        with open('./stats/q_val.pkl', 'wb') as f:
            pickle.dump(q_val_dict, f)
        if not episode % 1:
            print(f"Episode:{episode}  Avg Reward:{np.round(avg_score,2)},  Max:{np.round(high_score,2)}, Epsilon:{np.round(agent.epsilon,3)},  LearnRate:{agent.lr},  Replay_size: {min(agent.mem.mem_size, agent.mem.mem_counter)}  Loss: {loss}")
    if episode > 500 and high_score > save_score and False:
        save_score = high_score
        torch.save(agent.model, f'./models/q_model_{np.round(high_score,4)}.pth')
    episode += 1


        
    
    