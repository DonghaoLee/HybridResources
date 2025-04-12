#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle
from Env import Contextual_Bandit

# Set parameters
Env_Seed = 0
N_States = 20
N_Actions = 100
Feature_Dimension = 10

# Init the environment

env = Contextual_Bandit(S = N_States, A = N_Actions, d = Feature_Dimension, seed = Env_Seed)

# check the best arm

print("Max action is: ", env.best_arm, ". Rewards are ", env.reward[np.arange(N_States), env.best_arm])

# set optimal policy

optimal_policy = np.zeros([env.S, env.A])
optimal_policy[np.arange(N_States), env.best_arm] = 1.

# set offline policies

def generate_policy(k):
    exp = np.exp(k * env.reward)
    policy = exp / np.sum(exp, axis = -1, keepdims=True)
    return policy

offline_policies = {
                    'ρ₁': optimal_policy,
                    'ρ₂': generate_policy(5),
                    'ρ₃': generate_policy(-10)
                    }

# calculate coefficients

def est_theta(policy, N_offline):
    s_offline_hist, a_offline_hist, r_offline_hist = env.policy_pull_multi(policy, N_offline)
    offline_mat = np.zeros([env.d, env.d])
    offline_vec = np.zeros(env.d)
    for s, a, r in zip(s_offline_hist, a_offline_hist, r_offline_hist):
        offline_mat += env.covs[s, a]
        offline_vec += env.features[s, a] * r
   
    Cov = np.copy(offline_mat) + 0.001 * np.identity(env.d)
    Vec = np.copy(offline_vec)
    cov_inv = np.linalg.inv(Cov)
    return np.dot(cov_inv, Vec)

def est_err(policy, theta_p):
    return np.abs(np.dot(np.einsum('sad,sa->d', env.features, policy) / env.S, env.theta - theta_p))

N_0 = 2000
for key, policy in offline_policies.items():
    # print(np.trace(env.covar(optimal_policy)))
    # print(np.trace(env.covar(policy)))
    opt_err = []
    off_err = []
    for _ in range(500):
        theta_p = est_theta(policy, N_0)
        opt_err.append(est_err(optimal_policy, theta_p))
        off_err.append(est_err(policy, theta_p))
    print(key + ' coverage:', np.mean(opt_err) / np.mean(off_err))
    print()


# # Subroutine

def Hybrid_UCB(env, offline_policy,
               N_trials = 50,
               N_offline = 1000,
               N_online = 1000,
               beta = 1.):

    a_hhist = []
    r_hhist = []
    hhist = []

    for trial in range(N_trials):
        print(trial, end = ' ')
        
        s_offline_hist, a_offline_hist, r_offline_hist = env.policy_pull_multi(offline_policy, N_offline)
        offline_mat = np.zeros([env.d, env.d])
        offline_vec = np.zeros(env.d)
        for s, a, r in zip(s_offline_hist, a_offline_hist, r_offline_hist):
            offline_mat += env.covs[s, a]
            offline_vec += env.features[s, a] * r
       
        Cov = np.copy(offline_mat) + np.identity(env.d)
        Vec = np.copy(offline_vec)
        cov_inv = np.linalg.inv(Cov)

        s_hist = []
        a_hist = []
        r_hist = []
        hist = []

        bonus = np.ones([env.S, env.A])
        est_reward = np.zeros([env.S, env.A])
        pred_theta = np.matmul(cov_inv, Vec)
        for i in range(N_online):
            s = env.current_state
            bonus[s] = np.minimum(bonus[s], np.sqrt(np.trace(np.matmul(env.covs[s], cov_inv), axis1 = 1, axis2 = 2)))
            a = np.argmax(bonus[s])
            r = env.arm_pull(a)
            
            s_hist.append(s)
            a_hist.append(a)
            r_hist.append(r)

            Cov += env.covs[s, a]
            Vec += r * env.features[s, a]
            
            cov_inv -= np.dot(cov_inv, np.dot( env.features[s, a].reshape(env.d, 1), np.dot(env.features[s, a].reshape(1, env.d), cov_inv)))                         / (1 + np.dot(env.features[s, a].reshape(1, env.d), np.dot(cov_inv, env.features[s, a].reshape(env.d, 1))))

            pred_theta = np.matmul(cov_inv, Vec)
            est_reward = np.matmul(env.features, pred_theta) - beta * np.sqrt(np.trace(np.matmul(env.covs, cov_inv), axis1 = 2, axis2 = 3))
            pred_a = np.argmax(est_reward, axis = -1)
            est_opt_policy = np.zeros([env.S, env.A])
            est_opt_policy[np.arange(env.S), pred_a] = 1.
            hist.append(np.dot(np.einsum('sad,sa->d', env.features, optimal_policy - est_opt_policy) / env.S, env.theta))
        hhist.append(hist)
        a_hhist.append(a_hist)
        r_hhist.append(r_hist)
    
    return hhist


# # Runs

# different offline policies

suboptimality_Noff2000 = {}
for key, policy in offline_policies.items():
    print(key)
    hhist = Hybrid_UCB(env, policy, N_trials=100, N_offline=2000, N_online=10000, beta = 5.)
    suboptimality_Noff2000[key] = hhist

with open('S20_A100_d10_seed0_OfflineN_2000.pkl', 'wb') as f:
    pickle.dump({'offline': offline_policies,
                 'suboptimality':suboptimality_Noff2000}, f)

# different numbers of offline trajectories

suboptimality_N0 = {}
policy = 'ρ₂'
for key in [0, 1000, 2000, 4000]:
    print(key)
    hhist = Hybrid_UCB(env, offline_policies[policy], N_trials=100, N_offline=key, N_online=50000, beta = 5.)
    suboptimality_N0[str(key)] = hhist

with open('S20_A100_d10_seed0_OfflineN_diff.pkl', 'wb') as f:
    pickle.dump({'offline': 'ρ₂',
                 'suboptimality':suboptimality_N0}, f)
