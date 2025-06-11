#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pickle
from src.utils import compute_avg

# # Load MovieLens Data

with open('MovieLens_seed0_OfflineN_4000.pkl', 'rb') as f:
    x = pickle.load(f)
Contextual_suboptimality = x['suboptimality']

with open('MovieLens_seed0_OfflineN_diff.pkl', 'rb') as f:
    x = pickle.load(f)
Contextual_suboptimality_Ndiff = x['suboptimality']


with open('Regret_MovieLens_seed0_OfflineN_4000.pkl', 'rb') as f: # 'Regret_MovieLens_seed0_OfflineN_2000.pkl'
    x = pickle.load(f)
Contextual_regret = x['regret']

with open('Regret_MovieLens_seed0_OfflineN_diff.pkl', 'rb') as f: # 'Regret_seed0_OfflineN_diff.pkl'
    x = pickle.load(f)
Contextual_regret_Ndiff = x['regret']

# # Draw

# The color of the new label policy
color_map = {
    'ρ₁': '#1f77b4', 
    'ρ₂': '#ff7f0e', 
    'ρ₃': '#2ca02c'
}



# New graph

threshold = 1e-3
plt.rcParams.update({'font.size': 18})

# Create a single figure with 4 subplots (1 row, 4 columns)
fig, axs = plt.subplots(1, 4, figsize=(24, 5), dpi=800)

########################################
# Subplot (0, 0):
# Contextual bandit, Sub optimality, Different offline policies
########################################
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3, 3))
axs[0].xaxis.set_major_formatter(xfmt)

# Plot UCB
hhist = Contextual_suboptimality_Ndiff[str(0)]
hhist = np.array(hhist).mean(0)
axs[0].plot(np.log(compute_avg(hhist[:50000]) + threshold),
            label='UCB', color='black')

# Plot other policies: 'ρ₃', 'ρ₂', 'ρ₁'
for key in ['ρ₃', 'ρ₂', 'ρ₁']:
    hhist = Contextual_suboptimality[key]
    hhist = np.array(hhist).mean(0)
    axs[0].plot(np.log(compute_avg(hhist) + threshold),
                label=key, color=color_map[key])

axs[0].set_xlabel('N₁', fontsize=20)
axs[0].set_ylabel('Log sub-opt', fontsize=20)
axs[0].legend(loc=1)
axs[0].grid()

########################################
# Subplot (0, 1):
# Contextual bandit, Sub optimality, Different offline numbers
########################################
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3, 3))
axs[1].xaxis.set_major_formatter(xfmt)

for key in Contextual_suboptimality_Ndiff.keys():
    hhist = Contextual_suboptimality_Ndiff[key]
    hhist = np.array(hhist).mean(0)
    axs[1].plot(np.log(compute_avg(hhist) + threshold),
                label='N₀=' + str(key))

axs[1].set_xlabel('N₁', fontsize=20)
axs[1].set_ylabel('Log sub-opt', fontsize=20)
axs[1].legend(loc=1)
axs[1].grid()

########################################
# Subplot (0, 2):
# Contextual bandit, Regret, Different offline policies
########################################
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3, 3))
axs[2].xaxis.set_major_formatter(xfmt)

# Plot UCB
hhist = np.array(Contextual_regret_Ndiff[str(0)]).mean(0)
axs[2].plot(hhist, label='UCB', color='black')

# Plot other policies in specific order
keys_order = ['ρ₁', 'ρ₂', 'ρ₃']
for key in keys_order:
    hhist = Contextual_regret[key]
    hhist = np.array(hhist).mean(0)
    axs[2].plot(hhist, label=key, color=color_map[key])

axs[2].set_xlabel('N₁', fontsize=20)
axs[2].set_ylabel('Regret', fontsize=20)
axs[2].legend(loc=2)
axs[2].grid()

########################################
# Subplot (0, 3):
# Contextual bandit, Regret, Different offline numbers
########################################
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3, 3))
axs[3].xaxis.set_major_formatter(xfmt)

for n_off in Contextual_regret_Ndiff.keys():
    mean_regret = np.array(Contextual_regret_Ndiff[str(n_off)]).mean(0)
    axs[3].plot(mean_regret, label='N₀=' + str(n_off))

axs[3].set_xlabel('N₁', fontsize=20)
axs[3].set_ylabel('Regret', fontsize=20)
axs[3].legend(loc=2)
axs[3].grid()

# Adjust layout, save and show
fig.tight_layout()
fig.savefig('Fig_all.png', bbox_inches='tight')
plt.close()