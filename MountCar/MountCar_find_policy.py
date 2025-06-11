#!/usr/bin/env python
# coding: utf-8

import numpy as np
import gymnasium as gym
from tqdm import tqdm

from src.Env import MCenv_wrapper

# Set parameters
Env_Seed = 1
env_id = 'MountainCar-v0'
gym_env = gym.make(env_id)

discretize = 30
act_dim = gym_env.action_space.n
obs_dim = gym_env.observation_space.shape[0]
N_States = discretize ** obs_dim + 1  # Add one for the specific end discretized state
N_Actions = act_dim
gamma = 0.99
env = MCenv_wrapper(gym_env, N_states=N_States, N_actions=N_Actions)

# UCB algorithm implementation
def UCB(env, N_trials=50, N_online=1000, beta=1.):
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
        
        hist = []
        count = np.zeros((N_States, N_Actions, N_States))
        sa_count = np.zeros((N_States, N_Actions))
        bonus = np.ones((N_States, N_Actions))
        trans = np.zeros((N_States, N_Actions, N_States))
        update_transitions()
        
        for i in tqdm(range(N_online)):
            # Find an exploration policy using value iteration
            last_V = np.zeros(N_States)
            while True:
                new_V = np.zeros(env.S)
                online_policy = np.zeros((N_States, N_Actions))
                for s in range(env.S):
                    Qa = env.reward[s] + bonus[s] + gamma * np.dot(trans[s], last_V)
                    a = np.argmax(Qa)
                    online_policy[s, a] = 1.
                    new_V[s] = Qa[a]
                if np.max(np.abs(new_V - last_V)) < 0.01:
                    break
                last_V = new_V

            # Explore using the policy
            online_hist = env.policy_pull(online_policy)
            # Update transition counts and bonuses
            for s, a, s_p in online_hist:
                count[s, a, s_p] += 1
                sa_count[s, a] += 1
                bonus[s, a] = np.minimum(beta * np.sqrt(1 / sa_count[s, a]), bonus[s, a])
                trans[s, a] = count[s, a] / sa_count[s, a]
            # Record the evaluation results
            if i % 100 == 99:
                hist.append([env.emp_eval(online_policy), env.emp_eval_2(online_policy)])
            
            if i % 1000 == 999:
                print(f"Iteration {i + 1}: {hist[-10:]}")
                np.save('MountCar_UCB_hist.npy', hist)
                np.save('MountCar_UCB_bonus.npy', bonus)
                np.save('MountCar_UCB_trans.npy', trans)
        hhist.append(hist)

    return hhist, bonus, trans

# Run the UCB algorithm
hist, bonus, trans = UCB(env, N_trials=1, N_online=10000, beta=0.5)
np.save('MountCar_UCB_hist_10000.npy', hist)
np.save('MountCar_UCB_bonus_10000.npy', bonus)
np.save('MountCar_UCB_trans_10000.npy', trans)