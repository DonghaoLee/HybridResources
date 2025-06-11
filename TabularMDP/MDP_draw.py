#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pickle
from src.utils import compute_avg

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

color_map = {
    'ρ₁': '#1f77b4', 
    'ρ₂': '#ff7f0e', 
    'ρ₃': '#2ca02c'
}

fig, axs = plt.subplots(1, 4, dpi=800, figsize=(24, 5))
plt.rcParams.update({'font.size': 18})

threshold = 1e-3

# Sub-optimality, Different offline policies
xfmt = ScalarFormatter()
xfmt.set_powerlimits((-3, 3))
axs[0].xaxis.set_major_formatter(xfmt)

hhist = MDP_suboptimality_Ndiff[str(0)]
hhist = np.array(hhist).mean(0)
axs[0].plot(np.log(compute_avg(hhist)), label='UCB', color='black')

for key in ['ρ₃', 'ρ₂', 'ρ₁']:
    hhist = MDP_suboptimality[key]
    hhist = np.array(hhist).mean(0)
    axs[0].plot(np.log(compute_avg(hhist) + threshold), label=key, color=color_map[key])

axs[0].set_xlabel('N₁', fontsize=20)
axs[0].set_ylabel('Log sub-opt', fontsize=20)
axs[0].legend()
axs[0].grid()

# Sub-optimality, Different offline numbers
for key in [0, 500, 1000]:
    hhist = MDP_suboptimality_Ndiff[str(key)]
    hhist = np.array(hhist).mean(0)
    axs[1].plot(np.log(compute_avg(hhist) + threshold), label='N₀=' + str(key))

axs[1].set_xlabel('N₁', fontsize=20)
axs[1].set_ylabel('Log sub-opt', fontsize=20)
axs[1].legend()
axs[1].grid()

# Regret, Different offline policies
axs[2].plot(np.array(MDP_regret_Ndiff[str(0)]).mean(0), label='UCB', color='black')

keys_order = ['ρ₁', 'ρ₂', 'ρ₃']
for key in keys_order:
    hhist = MDP_regret[key]
    hhist = np.array(hhist).mean(0)
    axs[2].plot(hhist, label=key, color=color_map[key])

axs[2].set_xlabel('N₁', fontsize=20)
axs[2].set_ylabel('Regret', fontsize=20)
axs[2].legend(loc=4)
axs[2].grid()

# Regret, Different offline numbers
for n_off in [0, 500, 1000]:
    axs[3].plot(np.array(MDP_regret_Ndiff[str(n_off)]).mean(0), label='N₀=' + str(n_off))

axs[3].set_xlabel('N₁', fontsize=20)
axs[3].set_ylabel('Regret', fontsize=20)
axs[3].legend(loc=4)
axs[3].grid()

fig.tight_layout()
fig.savefig('MDP_Fig.png', bbox_inches='tight')
plt.show()