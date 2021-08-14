# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 01:19:48 2020

@author: William Woodall
"""

from matplotlib import pyplot as plt
from matplotlib import animation
import seaborn as sns

import numpy as np
import pickle

train_stats_dir = './stats/training_hist.pkl'

resolution = input("Resolution:")
resolution = int(resolution) 
avg_len = int(input('Average:'))

plt.style.use('fivethirtyeight')
sns.set(style='darkgrid', palette='bright', font_scale=0.9)

fig = plt.figure(figsize=(14,10))
gs = fig.add_gridspec(9,4) 
ax = fig.add_subplot(gs[:3,:])
ax2 = fig.add_subplot(gs[4:7,:2])
ax5 = fig.add_subplot(gs[3:5,2:4])
ax3a = fig.add_subplot(gs[7:8,:2])
ax3b = fig.add_subplot(gs[8:9,:2])
ax4 = fig.add_subplot(gs[5:7,2:4])
ax6 = fig.add_subplot(gs[7:9,2:4])  
    
def plot_stats_animation(i, resolution=resolution, avg_len=avg_len):
    
    with open(train_stats_dir,'rb') as f:
        stats_dict = pickle.load(f)
        
    stats_len = len(stats_dict['episode'])  
    if not resolution == 0:
        if resolution > stats_len/2:
            print("No Solution!!")
            pass
        else:
            indicies = np.arange(0, stats_len, resolution)
            stats_dict['episode'] = [np.mean(stats_dict['episode'][i:i+1]) for i in indicies[:-1]]
            stats_dict['average'] = [np.mean(stats_dict['average'][i:i+1]) for i in indicies[:-1]]
            stats_dict['moving average'] = [np.mean(stats_dict['moving average'][i+1]) for i in indicies[:-1]]
            stats_dict['min'] = [np.mean(stats_dict['min'][i:i+1]) for i in indicies[:-1]]
            stats_dict['max'] = [np.mean(stats_dict['max'][i:i+1]) for i in indicies[:-1]]
            stats_dict['actor_lr'] = [np.mean(stats_dict['actor_lr'][i:i+1]) for i in indicies[:-1]]
            stats_dict['critic_lr'] = [np.mean(stats_dict['critic_lr'][i:i+1]) for i in indicies[:-1]]
            stats_dict['actor_loss'] = [np.mean(stats_dict['actor_loss'][i:i+1]) for i in indicies[:-1]]
            stats_dict['critic_loss'] = [np.mean(stats_dict['critic_loss'][i:i+1]) for i in indicies[:-1]]
            stats_dict['entropy'] = [np.mean(stats_dict['entropy'][i:i+1]) for i in indicies[:-1]]
    
    avg_on = False
    if not avg_len:
        avg_len = 10
        
    if len(stats_dict['episode']) > avg_len:
        avg_on = True
    
    ax.cla()
    ax.plot(stats_dict['episode'], stats_dict['average'], '.', linewidth=0.5, label='Average Reward')
    ax.plot(stats_dict['episode'], stats_dict['moving average'], linewidth=0.9, color=(204/255,0/255,255/255), label='Moving Average')
    ax.text(stats_dict['episode'][-1], stats_dict['average'][-1], f"{np.round(stats_dict['average'][-1],2)}", color='blue', alpha=0.6)
    if avg_on:
        ax.plot(stats_dict['episode'][avg_len:], [np.mean(stats_dict['moving average'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['moving average']))], linewidth=0.8, color='red')
        ax.text(stats_dict['episode'][avg_len:][-1], [np.mean(stats_dict['moving average'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['moving average']))][-1], 
                f"{np.round([np.mean(stats_dict['moving average'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['moving average']))][-1],2)}", color='red')
    ax.fill_between(stats_dict['episode'], [np.mean(stats_dict['min'][:i]) for i in range(len(stats_dict['min']))], 
                                            [np.mean(stats_dict['max'][:i]) for i in range(len(stats_dict['max']))],
                                            color=(0/255,148/255,178/255), alpha=0.2)
    ax.set_title('PPO Training History')
  

    ax2.cla()
    ax2.plot(stats_dict['episode'], stats_dict['max'], linewidth=0.5, label='Max Reward')
    ax2.plot(stats_dict['episode'], stats_dict['min'], linewidth=0.5, label='Min Reward')
    ax2.plot(stats_dict['episode'], stats_dict['average'], linewidth=0.5, label='Avg Reward')
    ax2.set_title('Game averages')
 
      
    ax4.cla()
    ax4.set_title('Critic Loss')
    ax4.plot(stats_dict['critic_loss'], '-', linewidth=0.5)
    if avg_on:
        ax4.plot(range(len(stats_dict['critic_loss']))[avg_len:], [np.mean(stats_dict['critic_loss'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['critic_loss']))], linewidth=0.9, color='red')
        ax4.text(range(len(stats_dict['critic_loss']))[avg_len:][-1], [np.mean(stats_dict['critic_loss'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['critic_loss']))][-1], 
             f"{np.round([np.mean(stats_dict['critic_loss'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['critic_loss']))][-1],3)}", color='red')
    
    ax5.cla()
    ax5.plot(stats_dict['actor_loss'], linewidth=0.5)
    if avg_on:
        ax5.plot(range(len(stats_dict['actor_loss']))[avg_len:], [np.mean(stats_dict['actor_loss'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['actor_loss']))], linewidth=0.9, color='red')
        ax5.text(range(len(stats_dict['actor_loss']))[avg_len:][-1], [np.mean(stats_dict['actor_loss'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['actor_loss']))][-1],
                 f"{np.round([np.mean(stats_dict['actor_loss'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['actor_loss']))][-1],3)}", color='red')
    ax5.set_title('Actor Loss')
     
    ax3a.cla()
    ax3a.set_title('Learning Rate')
    ax3a.plot(stats_dict['episode'], stats_dict['actor_lr'], '-', linewidth=0.8, color='Purple')
    ax3a.text(stats_dict['episode'][-1], stats_dict['actor_lr'][-1], f"{np.round(stats_dict['actor_lr'][-1],7)}", color='Purple')
    
    ax3b.cla()
    ax3b.set_title('Learning Rate')
    ax3b.plot(stats_dict['episode'], stats_dict['critic_lr'], '-', linewidth=0.8, color='Purple')
    ax3b.text(stats_dict['episode'][-1], stats_dict['critic_lr'][-1], f"{np.round(stats_dict['critic_lr'][-1],7)}", color='Purple')
    
    ax6.cla()
    ax6.set_title('Entropy')
    ax6.plot(stats_dict['entropy'], '-', linewidth=0.5)
    if avg_on:
        ax6.plot(range(len(stats_dict['entropy']))[avg_len:], [np.mean(stats_dict['entropy'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['entropy']))], linewidth=0.9, color='red')
        ax6.text(range(len(stats_dict['entropy']))[avg_len:][-1], [np.mean(stats_dict['entropy'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['entropy']))][-1], 
                 f"{np.round([np.mean(stats_dict['entropy'][i-avg_len:i]) for i in range(avg_len, len(stats_dict['entropy']))][-1],3)}", color='red')
    
    
ani = animation.FuncAnimation(fig, plot_stats_animation, interval=300)
plt.show()