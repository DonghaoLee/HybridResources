import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import RandomSimplexVector

class Contextual_Bandit(object):
    def __init__(self, S = 20, A = 100, d = 10, noise = 1., seed = 0):
        self.S = S
        self.A = A
        self.d = d
        np.random.seed(seed)
        self.features = np.random.normal(size = [S, A, d])
        self.features = self.features / np.linalg.norm(self.features, axis=-1, keepdims = True)
        self.features = self.features / np.sign(self.features[:, :, 0:1]) #.reshape([self.])
        np.random.seed(int(time.time()))
        self.covs = np.einsum('sad,sak->sadk', self.features, self.features)
        self.theta = np.zeros(d)
        self.theta[0] = 1.0
        self.reward = np.einsum('d,sad->sa', self.theta, self.features)
        self.best_arm = np.argmax(self.reward, axis=-1)
        self.sigma = noise
        self.reset()
        return

    def reset(self,):
        self.current_state = np.random.randint(self.S)
    
    def policy_pull(self, policy):
        #s = np.random.choice(np.arange(self.S), 1, p = self.init_dist)
        a = np.random.choice(np.arange(self.A), 1, p = policy).item()
        r = self.arm_pull(a)
        return a, r
    
    def arm_pull(self, arm):
        #return self.reward[arm] + np.random.normal(scale = self.sigma)
        r = self.reward[self.current_state, arm] + 2 * self.sigma * np.random.random() - self.sigma
        self.reset()
        return r
    
    def policy_pull_multi(self, policy, N):
        s_hist = []
        a_hist = []
        r_hist = []
        for i in range(N):
            s_hist.append(self.current_state)
            a, r = self.policy_pull(policy[self.current_state])
            a_hist.append(a)
            r_hist.append(r)
        return s_hist, a_hist, r_hist
    
    def set_noise(self, sigma):
        self.sigma = sigma
        return
    
    def true_performance(self, policy):
        return np.mean(np.dot(self.reward, policy))
    
    def covar(self, policy):
        t = np.einsum('sadk,sa->dk', self.covs, policy) / self.S
        return t


class TabularMDP(object):
    def __init__(self, H, S, A, seed):
        self.H = H
        self.S = S
        self.A = A
        
        np.random.seed(seed)
        self.P = RandomSimplexVector(d = S, size = [H, S, A])
        self.reward = np.random.random(size = [H, S, A])
        np.random.seed(int(time.time()))
        
        self.sigma = 0
        
        self.reset()
        return

    def policy_pull(self, policy):
        #s = np.random.choice(np.arange(self.S), 1, p = self.init_dist)
        hist = []
        self.reset()
        for h in range(self.H):
            s = self.s
            a = np.random.choice(np.arange(self.A), 1, p = policy[h, self.s]).item()
            s_p, r = self.step(a)
            hist.append([s, a, s_p])
        return hist
    
    def policy_pull_multi(self, policy, N):
        hist = [[] for _ in range(self.H)]
        for i in range(N):
            hhist = self.policy_pull(policy)
            for l in range(self.H):
                hist[l].append(hhist[l])
        return hist
    
    def set_noise(self, sigma):
        self.sigma = sigma
        return

    def reset(self,):
        self.t = 0
        self.s = np.random.randint(self.S)
        return self.s

    def step(self, action): # pull action
        r = self.reward[self.t, self.s, action] + self.sigma * (np.random.random() - 0.5)
        s = np.random.choice(self.S, 1, p=self.P[self.t, self.s, action])
        self.s = s.item()
        self.t += 1
        return self.s, r

    def baseline_gen(self, temprature_k = 1):
        Q = np.zeros([self.H, self.S, self.A])
        V = np.zeros([self.H + 1, self.S])
        for h in range(self.H - 1, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    EV = np.dot(self.P[h, s, a], V[h+1])
                    Q[h, s, a] = self.reward[h, s, a] + EV
                tp = np.exp(temprature_k * Q[h, s]) / np.sum(np.exp(temprature_k * Q[h, s]))
                V[h, s] = np.dot(tp, Q[h, s])
        return np.mean(V[0]), np.exp(temprature_k * Q) / np.sum(np.exp(temprature_k * Q), axis = -1).reshape([self.H, self.S, 1])

    def best_gen(self,):
        Q = np.zeros([self.H, self.S, self.A])
        V = np.zeros([self.H + 1, self.S])
        policy = np.zeros([self.H, self.S, self.A])
        for h in range(self.H - 1, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    EV = np.dot(self.P[h, s, a], V[h+1])
                    Q[h, s, a] = self.reward[h, s, a] + EV
                policy[h, s, np.argmax(Q[h, s])] = 1.
                V[h, s] = np.max(Q[h, s])
        return np.mean(V[0]), policy

    def value_gen(self, policy):
        Q = np.zeros([self.H, self.S, self.A])
        V = np.zeros([self.H + 1, self.S])
        for h in range(self.H - 1, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    EV = np.dot(self.P[h, s, a], V[h+1])
                    Q[h, s, a] = self.reward[h, s, a] + EV
                V[h, s] = np.dot(policy[h, s], Q[h, s])
        return np.mean(V[0])

    
class HomoTabularMDP(object):
    def __init__(self, S, A, seed):
        self.S = S
        self.A = A
        
        np.random.seed(seed)
        self.P = RandomSimplexVector(d = S, size = [S, A])
        self.reward = np.random.random(size = [S, A])
        np.random.seed(int(time.time()))
        
        self.sigma = 0
        
        self.reset()
        return

    def policy_pull(self, policy):
        #s = np.random.choice(np.arange(self.S), 1, p = self.init_dist)
        hist = []
        self.reset()
        for h in range(self.H):
            s = self.s
            a = np.random.choice(np.arange(self.A), 1, p = policy[h, self.s]).item()
            s_p, r = self.step(a)
            hist.append([s, a, s_p])
        return hist
    
    def policy_pull_multi(self, policy, N):
        hist = [[] for _ in range(self.H)]
        for i in range(N):
            hhist = self.policy_pull(policy)
            for l in range(self.H):
                hist[l].append(hhist[l])
        return hist
    
    def set_noise(self, sigma):
        self.sigma = sigma
        return

    def reset(self,):
        self.t = 0
        self.s = np.random.randint(self.S)
        return self.s

    def step(self, action): # pull action
        r = self.reward[self.t, self.s, action] + self.sigma * (np.random.random() - 0.5)
        s = np.random.choice(self.S, 1, p=self.P[self.t, self.s, action])
        self.s = s.item()
        self.t += 1
        return self.s, r

    def baseline_gen(self, temprature_k = 1):
        Q = np.zeros([self.H, self.S, self.A])
        V = np.zeros([self.H + 1, self.S])
        for h in range(self.H - 1, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    EV = np.dot(self.P[h, s, a], V[h+1])
                    Q[h, s, a] = self.reward[h, s, a] + EV
                tp = np.exp(temprature_k * Q[h, s]) / np.sum(np.exp(temprature_k * Q[h, s]))
                V[h, s] = np.dot(tp, Q[h, s])
        return np.mean(V[0]), np.exp(temprature_k * Q) / np.sum(np.exp(temprature_k * Q), axis = -1).reshape([self.H, self.S, 1])

    def best_gen(self,):
        Q = np.zeros([self.H, self.S, self.A])
        V = np.zeros([self.H + 1, self.S])
        policy = np.zeros([self.H, self.S, self.A])
        for h in range(self.H - 1, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    EV = np.dot(self.P[h, s, a], V[h+1])
                    Q[h, s, a] = self.reward[h, s, a] + EV
                policy[h, s, np.argmax(Q[h, s])] = 1.
                V[h, s] = np.max(Q[h, s])
        return np.mean(V[0]), policy

    def value_gen(self, policy):
        Q = np.zeros([self.H, self.S, self.A])
        V = np.zeros([self.H + 1, self.S])
        for h in range(self.H - 1, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    EV = np.dot(self.P[h, s, a], V[h+1])
                    Q[h, s, a] = self.reward[h, s, a] + EV
                V[h, s] = np.dot(policy[h, s], Q[h, s])
        return np.mean(V[0])


class MovieLens_Contextual_Bandit(object):
    def __init__(self, S = 20, A = 100, d = 3, noise = 1., seed = 0):
        self.S = S
        self.A = A
        self.d = d
        with open('ml/movie_features_d3.pkl', 'rb') as f:
            self.movie_features = pickle.load(f)
        
        with open('ml/user_theta_d3.pkl', 'rb') as f:
            self.user_theta = pickle.load(f)
        
        np.random.seed(seed)
        assert self.S <= self.user_theta.shape[0]
        self.user_theta = self.user_theta[np.random.choice(np.arange(self.user_theta.shape[0]), self.S, replace = False)]
        assert self.A <= self.movie_features.shape[0]
        # self.total_A = self.movie_features.shape[0]
        self.movie_features = self.movie_features[np.random.choice(np.arange(self.movie_features.shape[0]), self.A, replace = False)]
        
        np.random.seed(int(time.time()))
        self.covs = np.einsum('sd,sk->sdk', self.user_theta, self.user_theta)
        self.reward = np.einsum('sd,ad->sa', self.user_theta, self.movie_features)
        self.best_arm = np.argmax(self.reward, axis=-1)
        self.sigma = noise
        self.reset()
        return

    def reset(self,):
        self.current_state = np.random.randint(self.S)
        # self.available_arms = np.random.choice(np.arange(self.total_A), self.A, replace = False)
    
    def policy_pull(self, policy):
        #s = np.random.choice(np.arange(self.S), 1, p = self.init_dist)
        if len(policy.shape) == 2:
            policy = policy[self.current_state]
        # policy = policy[self.available_arms]
        # if np.sum(policy) == 0:
        #     policy = policy + 1.
        # policy = policy / np.sum(policy)
        a = np.random.choice(np.arange(self.A), 1, p = policy).item()
        # a = self.available_arms[a]
        r = self.arm_pull(a)
        return a, r
    
    def arm_pull(self, arm):
        #return self.reward[arm] + np.random.normal(scale = self.sigma)
        r = self.reward[self.current_state, arm] + 2 * self.sigma * np.random.random() - self.sigma
        self.reset()
        return r
    
    def policy_pull_multi(self, policy, N):
        s_hist = []
        a_hist = []
        r_hist = []
        for i in range(N):
            s_hist.append(self.current_state)
            a, r = self.policy_pull(policy[self.current_state])
            a_hist.append(a)
            r_hist.append(r)
        return s_hist, a_hist, r_hist
    
    def set_noise(self, sigma):
        self.sigma = sigma
        return
    
    def true_performance(self, policy):
        assert len(policy.shape) == 2
        return np.einsum('sa,sa', self.reward, policy) / self.S
    
    # def covar(self, policy):
    #     assert len(policy.shape) == 2
    #     t = np.einsum('sadk,sa->dk', self.covs, policy) / self.S
    #     return t


# Wrapper class for the MountainCar environment
class MCenv_wrapper():
    def __init__(self, gym_env, N_states, N_actions, discretize=30):
        self.discretize = discretize
        self.gym_env = gym_env
        self.high = self.gym_env.observation_space.high
        self.low = self.gym_env.observation_space.low
        self.high[0] = 0.5  # Set the maximum position to 0.5
        
        self.H = 200  # Horizon length
        self.S = N_states
        self.A = N_actions
        self.state, _ = self.gym_env.reset(seed = 0)
        self.initial_reward()
    
    def initial_reward(self):
        """Compute reward based on the state."""
        self.reward = np.zeros(self.S)
        # Reward is 1 only for the specific end discretized state
        self.reward[-1] = 1
    
    def policy_pull(self, policy):
        """Execute a policy and return the history of states, actions, and next states."""
        hist = []
        terminated, truncated = False, False
        s, _ = self.gym_env.reset(seed = 0)
        for h in range(self.H):
            a = np.random.choice(self.A, p=policy[self.gym_to_env(s)])
            if terminated or truncated:
                s_p, _ = self.gym_env.reset(seed = 0)
                hist.append((self.gym_to_env(s), a, self.gym_to_env(s_p)))
                break
                terminated, truncated = False, False
            else:
                s_p, r, terminated, truncated, _ = self.gym_env.step(a)
                hist.append((self.gym_to_env(s), a, self.gym_to_env(s_p)))
            s = s_p
        return hist
    
    def policy_pull_multi(self, policy, N):
        """Execute a policy multiple times and return the history of states, actions, and next states."""
        hist = []
        for _ in range(N):
            hist.append(self.policy_pull(policy))
        return hist

    def emp_eval(self, policy, N = 1):
        """Evaluate the policy by computing the total reward over a trajectory."""
        hhist = []
        for _ in range(N):
            hist = self.policy_pull(policy)
            hhist.append(sum(self.reward[s] for s, _, _ in hist))
        return np.mean(hhist)
    
    def emp_eval_2(self, policy, N = 10):
        """Evaluate the policy by averaging the total reward over multiple trajectories."""
        rewards = []
        for _ in range(N):
            total = 0
            s, _ = self.gym_env.reset(seed=0)
            for h in range(self.H):
                a = np.random.choice(self.A, p=policy[self.gym_to_env(s)])
                s, r, terminated, truncated, _ = self.gym_env.step(a)
                if terminated or truncated:
                    break
                total += r
            rewards.append(total)
        return np.mean(rewards)
    
    def gym_to_env(self, state):
        """Convert continuous state to discrete state."""
        if state[0] >= 0.5:
            # Map to the specific end discretized state
            return self.S - 1
        else:
            # Use the previous discretization method
            scaled = (state - self.low) / (self.high - self.low)
            dis_state = np.clip((scaled * self.discretize).astype(int), 0, self.discretize - 1)
            if len(dis_state.shape) == 1:
                return dis_state[0] * self.discretize + dis_state[1]
            else:
                return dis_state[:, 0] * self.discretize + dis_state[:, 1]

    def env_to_gym(self, state):
        """Convert discrete state back to continuous state."""
        if state == self.S - 1:
            # Return a state where state[0] >= 0.5
            return np.array([0.5, 0.0])  # Example: Return a state where the car is at the top
        else:
            state = np.stack([state // self.discretize, state % self.discretize], axis=-1)
            return ((state + 0.5) / self.discretize) * (self.high - self.low) + self.low