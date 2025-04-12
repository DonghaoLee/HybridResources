#!/usr/bin/env python
# coding: utf-8

import numpy as np
import gymnasium as gym
from tqdm import tqdm
import pickle
import random

from Env import MCenv_wrapper

# Set parameters
env_id = 'MountainCar-v0'
gym_env = gym.make(env_id)

discretize = 30
act_dim = gym_env.action_space.n
obs_dim = gym_env.observation_space.shape[0]
N_States = discretize ** obs_dim + 1  # Add one for the specific end discretized state
N_Actions = act_dim
gamma = 0.99
env = MCenv_wrapper(gym_env, N_states=N_States, N_actions=N_Actions)


# Load Offline Data

with open('MountCarOffline.pkl', 'rb') as f:
    data = pickle.load(f)
    offline_data = data['OfflineData']


# UCB algorithm implementation
def Hybrid_UCB_regret(env, off_alpha, 
        N_trials=50, 
        N_offline=0,
        N_online=1000, 
        beta=1.):
    """Upper Confidence Bound algorithm for exploration and policy optimization."""
    def update_transitions():
        """Update transition probabilities and bonuses."""
        for s in range(env.S):
            for a in range(env.A):
                if sa_count[s, a] == 0:
                    trans[s, a] = np.ones(N_States) / env.S
                else:
                    trans[s, a] = count[s, a] / sa_count[s, a]
                    bonus[s, a] = np.minimum(beta * np.sqrt(1 / sa_count[s, a]), bonus[s, a])

    hhist = []
    for trial in range(N_trials):
        print(f"Trial {trial + 1}/{N_trials}")
        
        count = np.zeros([env.S, env.A, env.S])
        off_data = random.sample(offline_data[0.0], int((1 - off_alpha) * N_offline)) + random.sample(offline_data[1.0], int(off_alpha * N_offline))
        for x in off_data:
            for s, a, s_p in x:
                count[s, a, s_p] += 1
        
        hist = []
        sa_count = np.sum(count, axis = -1)
        bonus = np.ones((N_States, N_Actions))
        trans = np.zeros((N_States, N_Actions, N_States))
        update_transitions()
        
        last_V = np.zeros(N_States)
        reward_func = np.expand_dims(env.reward, axis=-1)
        for i in tqdm(range(N_online)):
            # Find an exploration policy using value iteration
            while True:
                Qa = reward_func + bonus + gamma * np.einsum('sap,p->sa', trans, last_V)
                new_V = np.max(Qa, axis=1)
                if np.max(np.abs(new_V - last_V)) < 0.01:
                    break
                last_V = new_V
            online_policy = np.zeros((N_States, N_Actions))
            for s in range(env.S):
                a = np.argmax(Qa[s])
                online_policy[s, a] = 1.0

            # Explore using the policy
            online_hist = env.policy_pull(online_policy)
            # Update transition counts and bonuses
            for s, a, s_p in online_hist:
                count[s, a, s_p] += 1
                sa_count[s, a] += 1
                bonus[s, a] = np.minimum(beta * np.sqrt(1 / sa_count[s, a]), bonus[s, a])
                trans[s, a] = count[s, a] / sa_count[s, a]
            # Record the evaluation results
            hist.append(env.emp_eval(online_policy, N=1))
            
        hhist.append(hist)

    return hhist


# # Runs

suboptimality_Noff = {}
for key in [0.0, 0.5, 1.0]:
    print(key)
    hhist = Hybrid_UCB_regret(env, key, N_trials=10, N_offline=1000, N_online=20000, beta = 0.2)
    suboptimality_Noff[key] = hhist

with open('Regret_MountCar_OfflineN.pkl', 'wb') as f:
    pickle.dump({'offline': offline_data,
                 'regret':suboptimality_Noff}, f)

# different numbers of offline trajectories

suboptimality_N0 = {}
policy = 0.5
for key in [0, 1000, 2000]:
    print(key)
    hhist = Hybrid_UCB_regret(env, policy, N_trials=10, N_offline=key, N_online=20000, beta = 0.2)
    suboptimality_N0[str(key)] = hhist

with open('Regret_MountCar_OfflineN_diff.pkl', 'wb') as f:
    pickle.dump({'offline': 0.5,
                 'regret':suboptimality_N0}, f)