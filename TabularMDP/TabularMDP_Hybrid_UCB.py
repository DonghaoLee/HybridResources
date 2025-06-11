#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle
from src.Env import TabularMDP

# Set parameters
Env_Seed = 1
Horizon = 3
N_States = 5
N_Actions = 10

# Init the environment

env = TabularMDP(H = Horizon, S = N_States, A = N_Actions, seed = Env_Seed)

# check the best arm

optimal_value, optimal_policy = env.best_gen()
print("Optimal Policy is: ", optimal_policy, ". Rewards are ", optimal_value)

# set offline policies

uniform_policy = np.ones([env.H, env.S, env.A]) / env.A
_, mid_policy = env.baseline_gen(temprature_k = 2)
offline_policies = {
                    'ρ₁':optimal_policy,
                    'ρ₂':mid_policy,
                    'ρ₃':uniform_policy,
                    }

# calculate coefficients

def est_trans(policy, N):
    hist = env.policy_pull_multi(policy, N)
    count = np.zeros([env.H, env.S, env.A, env.S])
    for h in range(env.H - 1, -1, -1):
        for s, a, s_p in hist[h]:
            count[h, s, a, s_p] += 1
    sa_count = np.sum(count, axis = -1)
    #bonus = np.ones([env.H, env.S, env.A])
    trans = np.zeros([env.H, env.S, env.A, env.S])
    for h in range(env.H):
        for s in range(env.S):
            for a in range(env.A):
                if sa_count[h, s, a] == 0:
                    trans[h, s, a] = np.ones([env.S]) / env.S
                else:
                    trans[h, s, a] = count[h, s, a] / sa_count[h, s, a]
    return trans

def est_val(policy, trans):
    last_V = np.zeros(env.S)
    Qa = np.zeros(env.A)
    for h in range(env.H - 1, -1, -1):
        new_V = np.zeros(env.S)
        for s in range(env.S):
            Qa = env.reward[h, s] + np.dot(trans[h, s], last_V)
            new_V[s] = np.dot(policy[h, s], Qa)
        last_V = new_V
    return np.mean(last_V)

N_off = 1000
est_optimal = est_val(optimal_policy, env.P)
print(est_optimal, optimal_value)
for key, policy in offline_policies.items():
    policy_err = []
    optimal_err = []
    policy_true_v = est_val(policy, env.P)
    for _ in range(100):
        trans = est_trans(policy, N_off)
        policy_est_v = est_val(policy, trans)
        optimal_est_v = est_val(optimal_policy, trans)
        policy_err.append(np.abs(policy_est_v - policy_true_v))
        optimal_err.append(np.abs(optimal_est_v - optimal_value))
    print(key, np.mean(optimal_err) / np.mean(policy_err))


# # Subroutine

# parameters
def Hybrid_UCB(env, offline_policy,
               N_trials = 50,
               N_offline = 1000,
               N_online = 1000,
               beta = 1.):

    hhist = []
    for trial in range(N_trials):
        print(trial, end=' ')
        offline_hist = env.policy_pull_multi(offline_policy, N_offline) 
        count = np.zeros([env.H, env.S, env.A, env.S])
        
        for h in range(env.H - 1, -1, -1):
            for s, a, s_p in offline_hist[h]:
                count[h, s, a, s_p] += 1

        hist = []

        sa_count = np.sum(count, axis = -1)
        bonus = np.ones([env.H, env.S, env.A])
        trans = np.zeros([env.H, env.S, env.A, env.S])
        for h in range(env.H):
            for s in range(env.S):
                for a in range(env.A):
                    if sa_count[h, s, a] == 0:
                        trans[h, s, a] = np.ones([env.S]) / env.S
                    else:
                        trans[h, s, a] = count[h, s, a] / sa_count[h, s, a]
                        bonus[h, s, a] = np.minimum(beta * np.sqrt(1 / sa_count[h, s, a]), bonus[h, s, a])
        Q1 = np.zeros([env.H, env.S, env.A])
        for i in range(N_online):
            # find a explore policy
            online_policy = np.zeros([env.H, env.S, env.A])
            last_V = np.zeros(env.S)
            Qa = np.zeros(env.A)
            for h in range(env.H - 1, -1, -1):
                new_V = np.zeros(env.S)
                for s in range(env.S):
                    Qa = bonus[h, s] + np.dot(trans[h, s], last_V)
                    a = np.argmax(Qa)
                    online_policy[h, s, a] = 1.
                    new_V[s] = Qa[a]
                last_V = new_V

            # explore
            online_hist = env.policy_pull(online_policy) 
            # update estimation
            for h in range(env.H):
                s, a, s_p  = online_hist[h]
                count[h, s, a, s_p] += 1
                sa_count[h, s, a] += 1
                bonus[h, s, a] = np.minimum(beta * np.sqrt(1 / sa_count[h, s, a]), bonus[h, s, a])
                trans[h, s, a] = count[h, s, a] / sa_count[h, s, a]
            # find a suboptimal policy
            est_policy = np.zeros([env.H, env.S, env.A])
            last_V = np.zeros(env.S)
            for h in range(env.H - 1, -1, -1):
                new_V = np.zeros(env.S)
                for s in range(env.S):
                    Q1[h, s] = np.maximum(env.reward[h, s] - bonus[h, s] + np.dot(trans[h, s], last_V), Q1[h, s])
                    a = np.argmax(Q1[h, s])
                    est_policy[h, s, a] = 1.
                    new_V[s] = Q1[h, s, a]
                last_V = new_V
            # update suboptimal gap
            hist.append(optimal_value - env.value_gen(est_policy))
        hhist.append(hist)

    return hhist


# # Runs

# different offline policies

suboptimality_Noff1000 = {}
for key, policy in offline_policies.items():
    print(key)
    hhist = Hybrid_UCB(env, policy, N_trials=100, N_offline=1000, N_online=20000, beta = 1 * np.sqrt(15))
    suboptimality_Noff1000[key] = hhist

with open('H3_S5_A10_seed1_OfflineN_1000.pkl', 'wb') as f:
    pickle.dump({'offline': offline_policies,
                 'suboptimality':suboptimality_Noff1000}, f)

# different numbers of offline trajectories

suboptimality_N0 = {}
policy = 'ρ₂'
for key in [0, 500, 1000]:
    print(key)
    hhist = Hybrid_UCB(env, offline_policies[policy], N_trials=100, N_offline=key, N_online=20000, beta = 1 * np.sqrt(15))
    suboptimality_N0[str(key)] = hhist

with open('H3_S5_A10_seed1_OfflineN_diff.pkl', 'wb') as f:
    pickle.dump({'offline': 'ρ₂',
                 'suboptimality':suboptimality_N0}, f)
