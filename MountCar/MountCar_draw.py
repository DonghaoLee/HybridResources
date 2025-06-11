#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pickle
from src.utils import compute_avg

def smooth_data(data, window_size):
    # Create a window (kernel) for the moving average
    window = np.ones(window_size) / window_size
    # Apply convolution to smooth the data
    smoothed_data = np.convolve(data, window, mode='valid')
    return smoothed_data

def turn_to_regret(data, maximum = 1):
    return np.cumsum(maximum - data)

# # Load MovieLens Data

with open('MountCar_OfflineN.pkl', 'rb') as f:
    x = pickle.load(f)
Contextual_suboptimality = x['suboptimality']

with open('MountCar_OfflineN_diff.pkl', 'rb') as f:
    x = pickle.load(f)
Contextual_suboptimality_Ndiff = x['suboptimality']

with open('Regret_MountCar_OfflineN.pkl', 'rb') as f: # 'Regret_MovieLens_seed0_OfflineN_2000.pkl'
    x = pickle.load(f)
Contextual_regret = x['regret']

with open('Regret_MountCar_OfflineN_diff.pkl', 'rb') as f: # 'Regret_seed0_OfflineN_diff.pkl'
    x = pickle.load(f)
Contextual_regret_Ndiff = x['regret']







# # Draw

# The color of the new label policy
color_map = {
    # 'ρ₁': '#1f77b4', 
    # 'ρ₂': '#ff7f0e', 
    # 'ρ₃': '#2ca02c'
    1.0: '#1f77b4', 
    0.5: '#ff7f0e', 
    0.0: '#2ca02c'
}

threshold = 1e-3
log_flag = False

# fig 0, 0
# Contextual bandit, Sub optimality, Different offline policies

plt.figure()
hhist = Contextual_suboptimality_Ndiff[str(0)]
hhist = 1 - np.array(hhist).mean(0)
if log_flag:
    plt.plot(np.log(compute_avg(hhist[:]) + threshold), label = 'UCB', color = 'black')
else:
    # plt.plot(compute_avg(hhist[:]), label = 'UCB', color = 'black')
    plt.plot(hhist[:], label = 'UCB', color = 'black')

for key in [0.0, 0.5, 1.0]:
    hhist = Contextual_suboptimality[key]
    hhist = 1 - np.array(hhist).mean(0)
    if log_flag:
        plt.plot(np.log(compute_avg(hhist) + threshold), label = 'α=' + str(key), color = color_map[key])
    else:
        # plt.plot(compute_avg(hhist), label = key, color = color_map[key])
        plt.plot(hhist, label = 'α=' + str(key), color = color_map[key])

plt.xlabel('N₁', fontsize = 20)
if log_flag:
    plt.ylabel('Log sub-opt', fontsize = 20)
else:
    plt.ylabel('Sub-opt', fontsize = 20)
plt.legend()
plt.grid()
plt.savefig(save_dir + 'Fig0_0.png', bbox_inches='tight')

# fig 0, 1
# Contextual bandit, Sub optimality, Different offline numbers

plt.figure()

for key in Contextual_suboptimality_Ndiff.keys(): 
    hhist = Contextual_suboptimality_Ndiff[key] 
    hhist = 1 - np.array(hhist).mean(0)
    if log_flag:
        plt.plot(np.log(compute_avg(hhist) + threshold), label = 'N₀=' + str(key))
    else:
        # plt.plot(compute_avg(hhist), label = 'N₀=' + str(key))
        plt.plot(hhist, label = 'N₀=' + str(key))

plt.xlabel('N₁', fontsize = 20)
if log_flag:
    plt.ylabel('Log sub-opt', fontsize = 20)
else:
    plt.ylabel('Sub-opt', fontsize = 20)
plt.legend()
plt.grid()
plt.savefig(save_dir + 'Fig0_1.png', bbox_inches='tight')

# Fig 0, 2
# Contextual bandit, Regret, Different offline policies
plt.figure()
hhist = np.array(Contextual_regret_Ndiff[str(0)]).mean(0)
hist = turn_to_regret(hhist)
# hist = smooth_data(hist, 100)
plt.plot(hist, label = 'UCB', color='black')

keys_order = [1.0, 0.5, 0.0]

for key in keys_order:
    hhist = Contextual_regret[key]
    hhist = np.array(hhist).mean(0)
    hist = turn_to_regret(hhist)
    # hist = smooth_data(hist, 100)
    
    plt.plot(hist, label = 'α=' + str(key), color = color_map[key])

plt.xlabel('N₁', fontsize = 20)
plt.ylabel('Regret', fontsize = 20)
plt.legend(loc = 'upper right')
plt.grid()
plt.savefig(save_dir + 'Fig0_2.png', bbox_inches='tight')

# Fig 0, 3
# Contextual bandit, Regret, Different offline numbers
plt.figure()
for n_off in Contextual_regret_Ndiff.keys(): 
    hist = np.array(Contextual_regret_Ndiff[str(n_off)]).mean(0)
    hist = turn_to_regret(hist)
    # hist = smooth_data(hist, 100)
    plt.plot(hist, label = 'N₀=' + str(n_off))

plt.xlabel('N₁', fontsize = 20)
plt.ylabel('Regret', fontsize = 20)
plt.legend(loc = 'upper right')
plt.grid()
plt.savefig(save_dir + 'Fig0_3.png', bbox_inches='tight')

# New Draw

plt.rcParams.update({'font.size': 18})

# Create a single figure and 4 subplots (1 row, 4 columns)
fig, axs = plt.subplots(1, 4, dpi=800, figsize=(24, 5))

threshold = 1e-3
log_flag  = False

########################################
# Subplot (0, 0):
# Contextual bandit, Sub-opt, Different offline policies
########################################
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3, 3))
axs[0].xaxis.set_major_formatter(xfmt)

hhist = Contextual_suboptimality_Ndiff[str(0)]
hhist = 1 - np.array(hhist).mean(0)
if log_flag:
    axs[0].plot(np.log(compute_avg(hhist) + threshold),
                label='UCB', color='black')
else:
    axs[0].plot(hhist, label='UCB', color='black')

for key in [0.0, 0.5, 1.0]:
    hhist = Contextual_suboptimality[key]
    hhist = 1 - np.array(hhist).mean(0)
    if log_flag:
        axs[0].plot(np.log(compute_avg(hhist) + threshold),
                    label=r'$\alpha=$' + str(key), 
                    color=color_map[key])
    else:
        axs[0].plot(hhist, 
                    label=r'$\alpha=$' + str(key), 
                    color=color_map[key])

axs[0].set_xlabel(r'$N_1$', fontsize=20)
if log_flag:
    axs[0].set_ylabel('Log sub-opt', fontsize=20)
else:
    axs[0].set_ylabel('Sub-opt', fontsize=20)
axs[0].legend()
axs[0].grid()

########################################
# Subplot (0, 1):
# Contextual bandit, Sub-opt, Different offline numbers
########################################
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3, 3))
axs[1].xaxis.set_major_formatter(xfmt)

for key in Contextual_suboptimality_Ndiff.keys():
    hhist = Contextual_suboptimality_Ndiff[key]
    hhist = 1 - np.array(hhist).mean(0)
    if log_flag:
        axs[1].plot(np.log(compute_avg(hhist) + threshold),
                    label=r'$N_0=$' + str(key))
    else:
        axs[1].plot(hhist, 
                    label=r'$N_0=$' + str(key))

axs[1].set_xlabel(r'$N_1$', fontsize=20)
if log_flag:
    axs[1].set_ylabel('Log sub-opt', fontsize=20)
else:
    axs[1].set_ylabel('Sub-opt', fontsize=20)
axs[1].legend()
axs[1].grid()

########################################
# Subplot (0, 2):
# Contextual bandit, Regret, Different offline policies
########################################
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3, 3))
axs[2].xaxis.set_major_formatter(xfmt)

yfmt = ScalarFormatter()
yfmt.set_powerlimits((-2, 2))
axs[2].yaxis.set_major_formatter(yfmt)

hhist = np.array(Contextual_regret_Ndiff[str(0)]).mean(0)
hist = turn_to_regret(hhist)
axs[2].plot(hist, label='UCB', color='black')

keys_order = [1.0, 0.5, 0.0]
for key in keys_order:
    hhist = Contextual_regret[key]
    hhist = np.array(hhist).mean(0)
    hist  = turn_to_regret(hhist)
    axs[2].plot(hist, 
                label=r'$\alpha=$' + str(key), 
                color=color_map[key])

axs[2].set_xlabel(r'$N_1$', fontsize=20)
axs[2].set_ylabel('Regret', fontsize=20)
axs[2].legend(loc='upper right')
axs[2].grid()

########################################
# Subplot (0, 3):
# Contextual bandit, Regret, Different offline numbers
########################################
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3, 3))
axs[3].xaxis.set_major_formatter(xfmt)

yfmt = ScalarFormatter()
yfmt.set_powerlimits((-2, 2))
axs[3].yaxis.set_major_formatter(yfmt)

for n_off in Contextual_regret_Ndiff.keys():
    hist = np.array(Contextual_regret_Ndiff[str(n_off)]).mean(0)
    hist = turn_to_regret(hist)
    axs[3].plot(hist, label=r'$N_0=$' + str(n_off))

axs[3].set_xlabel(r'$N_1$', fontsize=20)
axs[3].set_ylabel('Regret', fontsize=20)
axs[3].legend(loc='upper right')
axs[3].grid()

# Tidy up and save
fig.tight_layout()
fig.savefig('MountCar_Fig.png', bbox_inches='tight')
fig.close()