#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pickle
from utils import compute_avg

# # Load Contextual Data

with open('S20_A100_d10_seed0_OfflineN_2000.pkl', 'rb') as f:
    x = pickle.load(f)
Contextual_suboptimality = x['suboptimality']

with open('Regret_S20_A100_d10_seed0_OfflineN_2000.pkl', 'rb') as f:
    x = pickle.load(f)
Contextual_regret = x['regret']

with open('S20_A100_d10_seed0_OfflineN_diff.pkl', 'rb') as f:
    x = pickle.load(f)
Contextual_suboptimality_Ndiff = x['suboptimality']

with open('Regret_S20_A100_d10_seed0_OfflineN_diff.pkl', 'rb') as f:
    x = pickle.load(f)
Contextual_regret_Ndiff = x['regret']


# # Load MDP Data

with open('H3_S5_A10_seed1_OfflineN_1000.pkl', 'rb') as f:
    x = pickle.load(f)
MDP_suboptimality = x['suboptimality']

with open('Regret_H3_S5_A10_seed1_OfflineN_1000.pkl', 'rb') as f:
    x = pickle.load(f)
MDP_regret = x['regret']

with open('H3_S5_A10_seed1_OfflineN_diff.pkl', 'rb') as f:
    x = pickle.load(f)
MDP_suboptimality_Ndiff = x['suboptimality']

with open('Regret_H3_S5_A10_seed1_OfflineN_diff.pkl', 'rb') as f:
    x = pickle.load(f)
MDP_regret_Ndiff = x['regret']


# # Draw

# The color of the new label policy
color_map = {
    'ρ₁': '#1f77b4', 
    'ρ₂': '#ff7f0e', 
    'ρ₃': '#2ca02c'
}

# draw

fig, axs = plt.subplots(2, 4, dpi = 800, figsize = (24, 10))
plt.rcParams.update({'font.size': 18})

threshold = 1e-3

# fig 0, 0
# Contextual bandit, Sub optimality, Different offline policies

xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3,3))
axs[0, 0].xaxis.set_major_formatter(xfmt)

hhist = Contextual_suboptimality_Ndiff[str(0)]
hhist = np.array(hhist).mean(0)
axs[0, 0].plot(np.log(compute_avg(hhist[:10000]) + threshold), label = 'UCB', color = 'black')

for key in ['ρ₃', 'ρ₂', 'ρ₁']:
    hhist = Contextual_suboptimality[key]
    hhist = np.array(hhist).mean(0)
    axs[0, 0].plot(np.log(compute_avg(hhist) + threshold), label = key, color = color_map[key])

axs[0, 0].set_xlabel('N₁', fontsize = 20)
axs[0, 0].set_ylabel('Log sub-opt', fontsize = 20)
axs[0, 0].legend()
axs[0, 0].grid()

# fig 0, 1
# Contextual bandit, Sub optimality, Different offline numbers

xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3,3))
axs[0, 1].xaxis.set_major_formatter(xfmt)

for key in [0, 1000, 2000]: 
    hhist = Contextual_suboptimality_Ndiff[str(key)]
    hhist = np.array(hhist).mean(0)
    axs[0, 1].plot(np.log(compute_avg(hhist) + threshold), label = 'N₀=' + str(key))

axs[0, 1].set_xlabel('N₁', fontsize = 20)
axs[0, 1].set_ylabel('Log sub-opt', fontsize = 20)
axs[0, 1].legend()
axs[0, 1].grid()

# Fig 0, 2
# Contextual bandit, Regret, Different offline policies
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3,3))
axs[0, 2].xaxis.set_major_formatter(xfmt)
yfmt = ScalarFormatter()
yfmt.set_powerlimits((-2,2))
axs[0, 2].yaxis.set_major_formatter(yfmt)

axs[0, 2].plot(np.array(Contextual_regret_Ndiff[str(0)]).mean(0), label = 'UCB', color='black')

keys_order = ['ρ₁', 'ρ₂', 'ρ₃']

for key in keys_order:
    hhist = Contextual_regret[key]
    hhist = np.array(hhist).mean(0)
    axs[0, 2].plot(hhist, label = key, color = color_map[key])

axs[0, 2].set_xlabel('N₁', fontsize = 20)
axs[0, 2].set_ylabel('Regret', fontsize = 20)
axs[0, 2].legend(loc=1)
axs[0, 2].grid()

# Fig 0, 3
# Contextual bandit, Regret, Different offline numbers
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3,3))
axs[0, 3].xaxis.set_major_formatter(xfmt)
yfmt = ScalarFormatter()
yfmt.set_powerlimits((-2,2))
axs[0, 3].yaxis.set_major_formatter(yfmt)

for n_off in [0, 1000, 2000]:
    axs[0, 3].plot(np.array(Contextual_regret_Ndiff[str(n_off)]).mean(0), label = 'N₀=' + str(n_off))

axs[0, 3].set_xlabel('N₁', fontsize = 20)
axs[0, 3].set_ylabel('Regret', fontsize = 20)
axs[0, 3].legend(loc=1)
axs[0, 3].grid()

# fig 1, 0
# MDP, Sub optimality, Different offline policies

xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3,3))
axs[1, 0].xaxis.set_major_formatter(xfmt)

hhist = MDP_suboptimality_Ndiff[str(0)]
hhist = np.array(hhist).mean(0)
axs[1, 0].plot(np.log(compute_avg(hhist)), label = 'UCB', color = 'black')

for key in ['ρ₃', 'ρ₂', 'ρ₁']:
    hhist = MDP_suboptimality[key]
    hhist = np.array(hhist).mean(0)
    axs[1, 0].plot(np.log(compute_avg(hhist) + threshold), label = key, color = color_map[key])

axs[1, 0].set_xlabel('N₁', fontsize = 20)
axs[1, 0].set_ylabel('Log sub-opt', fontsize = 20)
axs[1, 0].legend()
axs[1, 0].grid()

# fig 1, 1
# MDP, Sub optimality, Different offline numbers

xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3,3))
axs[1, 1].xaxis.set_major_formatter(xfmt)

for key in [0, 500, 1000]: 
    hhist = MDP_suboptimality_Ndiff[str(key)]
    hhist = np.array(hhist).mean(0)
    axs[1, 1].plot(np.log(compute_avg(hhist) + threshold), label = 'N₀=' + str(key))

axs[1, 1].set_xlabel('N₁', fontsize = 20)
axs[1, 1].set_ylabel('Log sub-opt', fontsize = 20)
axs[1, 1].legend()
axs[1, 1].grid()

# Fig 1, 2
# Contextual bandit, Regret, Different offline policies
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3,3))
axs[1, 2].xaxis.set_major_formatter(xfmt)
yfmt = ScalarFormatter()
yfmt.set_powerlimits((-2,2))
axs[1, 2].yaxis.set_major_formatter(yfmt)

axs[1, 2].plot(np.array(MDP_regret_Ndiff[str(0)]).mean(0), label = 'UCB', color='black')

keys_order = ['ρ₁', 'ρ₂', 'ρ₃']

for key in keys_order:
    hhist = MDP_regret[key]
    hhist = np.array(hhist).mean(0)
    axs[1, 2].plot(hhist, label = key, color = color_map[key])

axs[1, 2].set_xlabel('N₁', fontsize = 20)
axs[1, 2].set_ylabel('Regret', fontsize = 20)
axs[1, 2].legend(loc=4)
axs[1, 2].grid()

# Fig 1, 3
# Contextual bandit, Regret, Different offline numbers
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3,3))
axs[1, 3].xaxis.set_major_formatter(xfmt)
yfmt = ScalarFormatter()
yfmt.set_powerlimits((-2,2))
axs[1, 3].yaxis.set_major_formatter(yfmt)

for n_off in [0, 500, 1000]:
    axs[1, 3].plot(np.array(MDP_regret_Ndiff[str(n_off)]).mean(0), label = 'N₀=' + str(n_off))

    axs[1, 3].set_xlabel('N₁', fontsize = 20)
    axs[1, 3].set_ylabel('Regret', fontsize = 20)
axs[1, 3].legend(loc=4)
axs[1, 3].grid()


fig.tight_layout()
fig.savefig('Fig.png', bbox_inches='tight')
plt.show()