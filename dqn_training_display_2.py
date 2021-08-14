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
q_val_dir = './stats/q_val.pkl'

plt.style.use('fivethirtyeight')
sns.set(style='darkgrid', palette='bright', font_scale=0.9)

q_filter = input('Q_filter') 
q_filter = int(q_filter)
if q_filter == 0:
    q_filter = 1_000_000
game_filter = input('Game Filter')
game_filter = int(game_filter)
if game_filter == 0:
    game_filter = 1_000_000
    

fig = plt.figure(figsize=(16,10))

gs = fig.add_gridspec(13,8) 
ax = fig.add_subplot(gs[:4,:6])
ax2 = fig.add_subplot(gs[5:9,:3])
ax5 = fig.add_subplot(gs[5:9,3:6])
ax7 = fig.add_subplot(gs[9:13,3:6])
ax3 = fig.add_subplot(gs[9:11,:3])
ax4 = fig.add_subplot(gs[11:13,:3])

qx1 = fig.add_subplot(gs[5:7,6:])
qx2 = fig.add_subplot(gs[7:9,6:])
qx3 = fig.add_subplot(gs[9:11,6:])


def plot_stats_animation(i):
    
    update = 0 
    while not update:
        try:
            with open(train_stats_dir,'rb') as f:
                stats_dict = pickle.load(f)
            with open(q_val_dir,'rb') as f:
                q_val_dict = pickle.load(f)
            update = 1
        except:
            pass
    
    ax.cla()
    ax.plot(stats_dict['episode'][-game_filter:], stats_dict['average'][-game_filter:], '-', linewidth=0.5, label='Average Reward', alpha=0.6)
    ax.text(stats_dict['episode'][-1], stats_dict['average'][-1], f"{np.round(stats_dict['average'][-1],2)}", color='blue', alpha=0.6)
    ax.plot(stats_dict['episode'][-game_filter:], stats_dict['moving average'][-game_filter:], linewidth=0.5, color=(204/255,0/255,255/255), label='Moving Average')
    ax.plot(stats_dict['episode'][-game_filter:], [np.mean(stats_dict['average'][:i]) for i in range(len(stats_dict['episode']))][-game_filter:], linewidth=0.8, color='red')
    ax.fill_between(stats_dict['episode'][-game_filter:], [np.mean(stats_dict['min'][-game_filter:][:i]) for i in range(len(stats_dict['min'][-game_filter:]))], 
                                            [np.mean(stats_dict['max'][-game_filter:][:i]) for i in range(len(stats_dict['max'][-game_filter:]))],
                                            color=(0/255,148/255,178/255), alpha=0.2)
    ax.set_title('Training History')
    ax.grid(linestyle='dashed', linewidth=0.5)
    #ax.legend()

    ax2.cla()
    ax2.plot(stats_dict['episode'][-game_filter:], stats_dict['max'][-game_filter:], linewidth=0.5, label='Max Reward')
    ax2.plot(stats_dict['episode'][-game_filter:], stats_dict['min'][-game_filter:], linewidth=0.5, label='Min Reward')
    ax2.plot(stats_dict['episode'][-game_filter:], stats_dict['average'][-game_filter:], linewidth=0.5, label='Avg Reward')
    ax2.set_title('Game averages')
    #ax2.legend()
      
   #if len(stats_dict['model_loss']) > 10:
    ax5.cla()
    ax5.plot(stats_dict['episode'][-game_filter:], stats_dict['model_loss'][-game_filter:], linewidth=0.5)
    ax5.plot(stats_dict['episode'][10:][-game_filter:], [np.mean(stats_dict['model_loss'][i-10:i]) for i in range(10, len(stats_dict['model_loss']))][-game_filter:], linewidth=0.9, color='red')
    ax5.set_title('Model loss')
    
    ax7.cla()
    ax7.plot(stats_dict['episode'][-game_filter:], stats_dict['temporal_difference'][-game_filter:], linewidth=0.5)
    ax7.plot(stats_dict['episode'][10:][-game_filter:], [np.mean(stats_dict['temporal_difference'][i-10:i]) for i in range(10, len(stats_dict['temporal_difference']))][-game_filter:], linewidth=0.9, color='red')
    ax7.set_title('Temporal Difference')
     
    ax3.cla()
    ax3.set_title('Learning Rate')
    ax3.plot(stats_dict['episode'][-game_filter:], stats_dict['learn_rate'][-game_filter:], '-', linewidth=0.8, color='purple')
    
    ax4.cla()
    ax4.set_title('Epsilon')
    ax4.plot(stats_dict['episode'][-game_filter:], stats_dict['epsilon'][-game_filter:], '-', linewidth=0.8, color='red')
    ax4.text(stats_dict['episode'][-1], stats_dict['epsilon'][-1], f"{np.round(stats_dict['epsilon'][-1],2)}", color='red' )
    
    qx1.cla()
    qx2.cla()
    qx3.cla()


    
    qx1.set_title('Q Values')
    qx1.plot(q_val_dict['0'][-q_filter:], linewidth=0.8, color=(175/255,100/255,175/255))
    qx2.plot(q_val_dict['1'][-q_filter:], linewidth=0.8, color=(200/255,75/255,150/255))

    
    
ani = animation.FuncAnimation(fig, plot_stats_animation, interval=300)
plt.show()