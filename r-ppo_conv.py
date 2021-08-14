# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 00:03:52 2021

@author: William Woodall
"""

import torch
from torch import nn
import torch.nn.functional as F

import cv2
import numpy as np
import os
import pickle
import gym


env = gym.make('CarRacing-v0')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.isdir('models'):
    os.makedirs('models')
if not os.path.isdir('stats'):
    os.makedirs('stats')



model_name = "R-PPO_Continuous_Conv"

class PPOMemory:
    def __init__(self):
        self.index = 0
        self.actor_hidden = []
        self.critic_hidden = []
        self.state = []
        self.action = []
        self.log_prob = []
        self.value = []
        self.reward = []
        self.done = []
        
    def update(self, actor_hidden, critic_hidden, state, action, log_prob, value, reward, done):
        self.actor_hidden.append(actor_hidden)
        self.critic_hidden.append(critic_hidden)
        self.state.append(state)
        self.action.append(action)
        self.log_prob.append(log_prob)
        self.value.append(value)
        self.reward.append(reward)
        self.done.append(1-done)
        
        self.index += 1
    
    
    def sample(self):
        return (self.actor_hidden, self.critic_hidden, self.state, self.action, self.log_prob, self.value, 
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
        self.actor_hidden = []
        self.critic_hidden = []
        self.state = []
        self.action = []
        self.log_prob = []
        self.value = []
        self.reward = []
        self.done = []
 
        
class ActorNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dims=64, std=.3):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.conv_out_dims = 12 * 12 * 64
        self.backbone_input_dims = self.conv_out_dims + self.hidden_dims
        self.fc1_dims = 1024
        self.fc2_dims = 1024
        self.num_actions = num_actions
        
        self.hidden_in = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.input_embedder = nn.Sequential(
                            nn.Conv2d(3, 16, kernel_size=(2,2), stride=2),    nn.ReLU(),
                            nn.Conv2d(16, 32, kernel_size=(2,2), stride=2),   nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=(2,2), stride=2),  nn.ReLU(),
                            ) 
        self.backbone = nn.Sequential(
            nn.Linear(self.backbone_input_dims, self.fc1_dims), nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),   nn.ReLU(),
            )
        self.dist = nn.Linear(self.fc2_dims, self.num_actions)
        self.log_std = nn.Parameter(torch.ones(self.num_actions)*std)
        
        self.hidden_out = nn.Linear(self.fc2_dims, self.hidden_dims)
        
    def forward(self, state, hidden_state):
        
        state = self.input_embedder(state)
        hidden_state = F.relu(self.hidden_in(hidden_state))
        state = state.view(-1, self.conv_out_dims)
        x = torch.cat((state,hidden_state), dim=1)
        x = self.backbone(x)
        mu = self.dist(x)
        log_std = self.log_std.exp().expand_as(mu)
        dist = torch.distributions.Normal(mu, log_std)  
        hidden_state_ = self.hidden_out(x)
        return dist, hidden_state_

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dims=64):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.conv_out_dims = 12 * 12 * 64
        self.backbone_input_dims = self.conv_out_dims + self.hidden_dims
        self.fc1_dims = 1024
        self.fc2_dims = 1024
        
        self.hidden_in = nn.Linear(self.hidden_dims, self.hidden_dims)
        self.input_embedder = nn.Sequential(
                            nn.Conv2d(3, 16, kernel_size=(2,2), stride=2),    nn.ReLU(),
                            nn.Conv2d(16, 32, kernel_size=(2,2), stride=2),   nn.ReLU(),
                            nn.Conv2d(32, 64, kernel_size=(2,2), stride=2),  nn.ReLU(),
                            ) 
        self.backbone = nn.Sequential(
            nn.Linear(self.backbone_input_dims, self.fc1_dims), nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),   nn.ReLU(),
            )
        self.value = nn.Linear(self.fc2_dims, 1)
        
        self.hidden_out = nn.Linear(self.fc2_dims, self.hidden_dims)
        
    def forward(self, state, hidden_state):
        
        state = self.input_embedder(state)
        hidden_state = F.relu(self.hidden_in(hidden_state))
        state = state.view(-1, self.conv_out_dims)
        x = torch.cat((state,hidden_state), dim=1)
        x = self.backbone(x)
        value = self.value(x)
        hidden_state_ = self.hidden_out(x)
        return value, hidden_state_
    
  
class PPOAgent:
    def __init__(self, input_shape, num_actions):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.hidden_size = 64
        self.epochs = 8
        self.mem_size = 1024
        self.batch_size = 128
        self.shuffle = False
        self.actor_lr = 5e-5
        self.critic_lr = 5e-5
        self.lr_decay = True
        self.lr_patience = 30_000
        self.gamma = .99
        self.tau = .95
        self.kl_clip = .2
        self.entropy_weight = .001
        self.adv_norm = True

        self.learn_step = 0
   
        self.mem = PPOMemory()
       
        self.actor = ActorNetwork(self.input_shape, self.num_actions, self.hidden_size).to(device)
        self.critic = CriticNetwork(self.input_shape, self.num_actions, self.hidden_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    
    def choose_action(self, state, actor_hidden, critic_hidden):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        actor_hidden = torch.tensor(actor_hidden, dtype=torch.float32).unsqueeze(0).to(device)
        critic_hidden = torch.tensor(critic_hidden, dtype=torch.float32).unsqueeze(0).to(device)
        
        dist, actor_hidden_ = self.actor(state, actor_hidden)
        value, critic_hidden_ = self.critic(state, critic_hidden)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.clamp(-1, 1)
        
        action = action.cpu().detach().numpy()[0]   
        log_prob = log_prob.cpu().detach().numpy()[0]
        value = value.detach().item()
        actor_hidden = actor_hidden.squeeze().cpu().detach().numpy()
        critic_hidden = critic_hidden.squeeze().cpu().detach().numpy()
        return action, log_prob, value, actor_hidden, critic_hidden
        
    def evaluate(self, state, critic_hidden): 
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        critic_hidden = torch.tensor(critic_hidden, dtype=torch.float32).unsqueeze(0).to(device)
        value_, critic_hidden_ = self.critic(state, critic_hidden)
        return value_.item()
    
    
    def rgb2gray(self, rgb):
        rgb = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        return np.reshape(rgb, (rgb.shape[0],rgb.shape[1],1))
    
    def process_img(self, img):
        #img = img / 255
        #img = self.rgb2gray(img)
        img = img.reshape(img.shape[2], img.shape[0], img.shape[1])
        return img
    
    def show(self, img):
        img = np.array(img)
        #img = img.reshape(img.shape[1], img.shape[2], img.shape[0])
        cv2.imshow('image', img)
        cv2.waitKey(1)
        
    def save_models(self, score):
        torch.save(self.actor, './models/'+model_name+f'_actor_{np.round(score,4)}.pth')
        torch.save(self.critic, './models/'+model_name+f'_critic_{np.round(score,4)}.pth')
    
    def load_models(self):
        self.actor = torch.load('./models/'+model_name+'_actor.pth')
        self.critic = torch.load('./models/'+model_name+'_critic.pth')
    
    def new_states(self):
        actor_hidden = np.zeros(self.hidden_size)
        critic_hidden = np.zeros(self.hidden_size)
        return actor_hidden, critic_hidden
        
        
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
            (actor_hidden, critic_hidden, state, action, log_prob, value, reward, done) = self.mem.sample()
            value_ = self.evaluate(state[-1], critic_hidden[-1])
            advantage = self.calculate_advantages(value, reward, value_, done)
            
            actor_hidden = torch.tensor(actor_hidden).float().to(device)
            critic_hidden = torch.tensor(critic_hidden).float().to(device)
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
                    
                    _actor_hidden = actor_hidden[batch]
                    _critic_hidden = critic_hidden[batch]
                    _state = state[batch]
                    _action = action[batch]
                    _advantage = advantage[batch].reshape(len(batch), 1)
                    _log_prob = log_prob[batch]
                    _returns = returns[batch]
             
                    dist, _ = self.actor(_state, _actor_hidden)
                    value_, _ = self.critic(_state, _critic_hidden)
                    
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

                    loss = .5 * critic_loss + actor_loss - self.entropy_weight * entropy
                    
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    self.actor_optimizer.step()
                    self.critic_optimizer.step() 
                    self.learn_step += 1
                    
                    if self.learn_step > 0 and not self.learn_step % self.lr_patience:
                        self.actor_lr   *= .1
                        self.critic_lr  *= .1
                        
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


agent = PPOAgent((96,96,1), 3)

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
    state = agent.process_img(state)
    actor_hidden, critic_hidden = agent.new_states()
    while not done:
        env.render()
        action, log_prob, value, actor_hidden_, critic_hidden_ = agent.choose_action(state, actor_hidden, critic_hidden)
        state_, reward, done, info = env.step(action)
        #agent.show(state_)
        state_ = agent.process_img(state_)
        
        agent.mem.update(actor_hidden, critic_hidden, state, action, log_prob, value, reward, done)
        
        actor_loss, critic_loss, entropy = agent.learn()

        state = state_
        actor_hidden = actor_hidden_
        critic_hidden = critic_hidden_
        score += reward
        steps += 1
        
        if actor_loss:
            _actor_loss = actor_loss
        if critic_loss:
            _critic_loss = critic_loss
        if entropy:
            _entropy = entropy
            
    score_history.append(score)  
    avg_score = np.mean(score_history[-25:])
    high_score = max(high_score, score)
    
       
    if not episode % stats_update:
        stats_dict['episode'].append(episode)
        stats_dict['moving average'].append(avg_score)
        stats_dict['min'].append(np.min(score_history[-stats_look_back:]))
        stats_dict['max'].append(np.max(score_history[-stats_look_back:]))
        stats_dict['average'].append(np.mean(score_history[-stats_look_back:]))
        stats_dict['actor_lr'].append(agent.actor_lr)
        stats_dict['critic_lr'].append(agent.critic_lr)
        stats_dict['actor_loss'].append(_actor_loss) 
        stats_dict['critic_loss'].append(_critic_loss) 
        stats_dict['entropy'].append(_entropy) 
        
        with open('./stats/training_hist.pkl', 'wb') as f:
            pickle.dump(stats_dict, f)
        print(f"Episode: {episode}    Score: {score}   Avg: {np.round(avg_score,2)}    Max :{np.round(high_score,2)}  actor_lr: {np.round(agent.actor_lr,6)}   Learn Steps: {agent.learn_step}   Memory:{agent.mem.index}")
    episode += 1


    