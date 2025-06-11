# from read_data import get_movie_data
import numpy as np
import pickle
from src.Env import MovieLens_Contextual_Bandit
from tqdm import tqdm

if __name__ == '__main__':
    # Set parameters
    Env_Seed = 0
    N_States = 943 # 943
    N_Actions = 20 # 1682
    Feature_Dimension = 3

    # Init the environment

    env = MovieLens_Contextual_Bandit(S = N_States, A = N_Actions, d = Feature_Dimension, seed = Env_Seed)

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
                        'ρ₃': generate_policy(0)
                        }

    # calculate coefficients

    def est_theta(policy, N_offline):
        s_offline_hist, a_offline_hist, r_offline_hist = env.policy_pull_multi(policy, N_offline)
        offline_mat = np.zeros([env.A, env.d, env.d])
        offline_vec = np.zeros([env.A, env.d])
        for s, a, r in zip(s_offline_hist, a_offline_hist, r_offline_hist):
            offline_mat[a] += env.covs[s]
            offline_vec[a] += env.user_theta[s] * r
    
        Cov = np.copy(offline_mat) + 0.001 * np.identity(env.d)
        Vec = np.copy(offline_vec)
        cov_inv = np.linalg.inv(Cov)
        return np.einsum('nij,nj->ni', cov_inv, Vec)

    def est_err(policy, est_features):
        return np.abs(np.einsum('sd,sd', np.einsum('ad,sa->sd', env.movie_features - est_features, policy), env.user_theta)) / env.S

    N_0 = 2000
    for key, policy in offline_policies.items():
        # print(np.trace(env.covar(optimal_policy)))
        # print(np.trace(env.covar(policy)))
        opt_err = []
        off_err = []
        for _ in range(500):
            tmp = est_theta(policy, N_0)
            opt_err.append(est_err(optimal_policy, tmp))
            off_err.append(est_err(policy, tmp))
        print(key + ' coverage:', np.mean(opt_err) / np.mean(off_err))
        print()


    # # Subroutine

    def Hybrid_UCB_regret(env, offline_policy,
                N_trials = 50,
                N_offline = 1000,
                N_online = 1000,
                beta = 1.):

        a_hhist = []
        r_hhist = []
        hhist = []

        for trial in tqdm(range(N_trials)):
            # print(trial, end = ' ')

            s_offline_hist, a_offline_hist, r_offline_hist = env.policy_pull_multi(offline_policy, N_offline)
            offline_mat = np.zeros([env.A, env.d, env.d])
            offline_vec = np.zeros([env.A, env.d])
            for s, a, r in zip(s_offline_hist, a_offline_hist, r_offline_hist):
                offline_mat[a] += env.covs[s]
                offline_vec[a] += env.user_theta[s] * r

            Cov = np.copy(offline_mat) + np.identity(env.d) # a d d
            Vec = np.copy(offline_vec)
            cov_inv = np.linalg.inv(Cov)

            s_hist = []
            a_hist = []
            r_hist = []
            hist = [0]

            bonus = np.minimum(np.ones([env.S, env.A]), beta * np.sqrt(np.einsum('sij,aji->sa', env.covs, cov_inv))) # np.ones([env.S, env.A])
            # print(bonus)
            pred_features = np.einsum('aij,aj->ai', cov_inv, Vec)
            est_reward = np.minimum(np.ones([env.S, env.A]), np.matmul(env.user_theta, pred_features.transpose()) + bonus) # np.zeros([env.S, env.A])
            for i in range(N_online):
                s = env.current_state
                
                a = np.argmax(est_reward[s])
                r = env.arm_pull(a)

                s_hist.append(s)
                a_hist.append(a)
                r_hist.append(r)

                Cov[a] += env.covs[s]
                Vec[a] += r * env.user_theta[s]

                cov_inv[a] -= np.dot(cov_inv[a], np.dot( env.user_theta[s].reshape(env.d, 1), np.dot(env.user_theta[s].reshape(1, env.d), cov_inv[a])))\
                        / (1 + np.dot(env.user_theta[s].reshape(1, env.d), np.dot(cov_inv[a], env.user_theta[s])))

                pred_features[a] = np.matmul(cov_inv[a], Vec[a])
                bonus[:, a] = np.minimum(bonus[:, a], beta * np.sqrt(np.trace(np.matmul(env.covs, cov_inv[a]), axis1 = 1, axis2 = 2)))
                est_reward[:, a] = np.minimum(est_reward[:, a], np.matmul(env.user_theta, pred_features[a]) + bonus[:, a])
                pred_a = np.argmax(est_reward, axis = -1)
                est_opt_policy = np.zeros([env.S, env.A])
                est_opt_policy[np.arange(env.S), pred_a] = 1.
                hist.append(hist[-1] + env.reward[s, env.best_arm[s]] - env.reward[s, a])
            hhist.append(hist)
            a_hhist.append(a_hist)
            r_hhist.append(r_hist)
        
        print()
        
        return hhist


    # # Runs

    # different offline policies

    suboptimality_Noff4000 = {}
    for key, policy in offline_policies.items():
        print(key)
        hhist = Hybrid_UCB_regret(env, policy, N_trials=10, N_offline=4000, N_online=50000, beta = 10.)
        suboptimality_Noff4000[key] = hhist

    with open('Regret_MovieLens_seed0_OfflineN_4000.pkl', 'wb') as f:
        pickle.dump({'offline': offline_policies,
                    'regret':suboptimality_Noff4000}, f)


    # different numbers of offline trajectories

    suboptimality_N0 = {}
    policy = 'ρ₂'
    for key in [0, 2000, 4000, 8000]:
        print(key)
        hhist = Hybrid_UCB_regret(env, offline_policies[policy], N_trials=10, N_offline=key, N_online=50000, beta = 10.)
        suboptimality_N0[str(key)] = hhist

    with open('Regret_MovieLens_seed0_OfflineN_diff.pkl', 'wb') as f:
        pickle.dump({'offline': 'ρ₂',
                    'regret':suboptimality_N0}, f)
