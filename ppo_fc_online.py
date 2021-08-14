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

env = gym.make('CartPole-v0')

timestamp = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.isdir('models'):
    os.makedirs('models')
if not os.path.isdir('stats'):
    os.makedirs('stats')

class PPOMemory:
    def __init__(self):
        self.state = 0
        self.action = 0
        self.log_prob = 0
        self.value = 0
        self.reward = 0
        self.state_ = 0
        self.done = 0
        
    def action_update(self, state, action, log_prob, value):
        self.state = state
        self.action = action
        self.log_prob = log_prob 
        self.value = value
    
    def reaction_update(self, reward, state_, done):
        self.reward = reward
        self.state_ = state_
        self.done = done
    
    def sample(self):
        return self.state, self.action, self.log_prob, self.value, self.reward, self.state_, self.done
        
class ActorNetwork(nn.Module):
    def __init__(self, input_dims, num_actions):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = 512
        self.fc2_dims = 256
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.dist = nn.Linear(self.fc2_dims, num_actions)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        dist = self.dist(x)
        dist = F.softmax(dist, dim=1)
        dist = torch.distributions.Categorical(dist)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, num_actions):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = 512
        self.fc2_dims = 256  
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
        self.q_val = nn.Linear(self.fc2_dims, 1)
    
    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        q_val = self.q_val(x)
        return q_val
        
          
  
class PPTDAgent:
    def __init__(self, input_dims, num_actions):
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.epochs = 6
        self.batch_size = 5
        self.actor_lr = 1e-8
        self.critic_lr = 5e-8
        self.gamma = .99

        self.learn_step = 0
   
        self.mem = PPOMemory()
       
        self.actor = ActorNetwork(self.input_dims, num_actions).to(device)
        self.target_actor = ActorNetwork(self.input_dims, num_actions).to(device)
        self.critic = CriticNetwork(self.input_dims, num_actions).to(device)
        self.target_critic = CriticNetwork(self.input_dims, num_actions).to(device)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action, log_prob, entropy = self.actor(state)
        value = self.critic(state)
        self.mem.action_update(state, action, log_prob, value)
        return action.item()
    
        
    def learn(self):                     
        for i in range(self.epochs):
            state, action, log_prob, value, reward, state_, done = self.mem.sample()
  
            state = torch.tensor(state, dtype=torch.float32).to(device)
            state_ = torch.tensor(state_, dtype=torch.float32).unsqueeze(0).to(device)
            action = torch.tensor(action).to(device)
            log_prob = torch.tensor(log_prob).to(device)
            reward = torch.tensor(reward).to(device)
            value = torch.tensor(value).to(device)
            done = torch.tensor(done).to(device)
            
            value_ = self.critic(state_)
            advantage = reward * done * self.gamma + value_ - value
            #advantage = (advantage-advantage.mean()) / (advantage.std() + 1e-3)
            
            new_action, new_log_prob, entropy = self.actor(state)
          
            ratio = torch.exp(new_log_prob - log_prob)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage 
            actor_loss = torch.min(surr1, surr2).mean()
            critic_loss = ((advantage)**2)
        
            loss = actor_loss - critic_loss
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            loss.backward()
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            return actor_loss.item(), critic_loss.item()
            #print(f"Critic Loss: {np.round(.1*critic_loss.item(),4)}, Actor Loss: {np.round(actor_loss.item(),4)}")


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
              }


agent = PPTDAgent(4,2)

high_score = 0
score_history = []
loss_history = []

episode = 0
save_score = 0.5

ppo_update_steps = 20


steps = 0
while True:
    
    loss_ = 0
    score = 0
    done = False
    state = env.reset() 
    while not done:
        env.render() 
        action = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        agent.mem.reaction_update(reward,state_,done)
        actor_loss, critic_loss = agent.learn()
            
        td = 0
        loss= 0

        state = state_
        score += reward
        steps += 1
        
   
    score_history.append(score)  
    avg_score = np.mean(score_history[-50:])
    high_score = max(high_score, avg_score)

        

    stats_dict['episode'].append(episode)
    stats_dict['moving average'].append(avg_score)
    stats_dict['min'].append(np.min(score_history[-stats_look_back:]))
    stats_dict['max'].append(np.max(score_history[-stats_look_back:]))
    stats_dict['average'].append(np.mean(score_history[-stats_look_back:]))
    stats_dict['actor_lr'].append(agent.actor_lr)
    stats_dict['critic_lr'].append(agent.critic_lr)
    stats_dict['actor_loss'].append(actor_loss)
    stats_dict['critic_loss'].append(critic_loss)   
    
    with open('./stats/training_hist.pkl', 'wb') as f:
        pickle.dump(stats_dict, f)
    print(f"Episode:{episode}  Avg Reward:{np.round(avg_score,2)},  Max:{np.round(high_score,2)},  LearnRate:{agent.actor_lr}")

    episode += 1



    
    
    