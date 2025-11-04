import numpy as np
from collections import defaultdict
import math

class MSR():
    def __init__(
        self,
        action_size,
        epsilon=0.05,
        gamma=0.99,
        learning_rate_topo=0.01,
        learning_rate_social=0.01,
        r_learning_rate_topo=0.01,
        r_learning_rate_social=0.01,
        # discretization for goal (dx, dy)
        goal_xy_bins=50,
        goal_xy_max_abs=10.0,
        goal_xy_edges=None,
        # discretization for humans (x, y)
        human_xy_bins=50,
        human_xy_max_abs=10.0,
        human_xy_edges=None
    ):
        self.action_size = action_size
        
        # Use dictionaries for dynamic state space
        # SR[state][action][next_state]
        self.SR_topo = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.SR_social = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Reward function R[state][action] = expected immediate reward
        self.R_topo = defaultdict(lambda: defaultdict(float))
        self.R_social = defaultdict(lambda: defaultdict(float))

        self.epsilon = epsilon
        self.gamma = gamma
        self.r_learning_rate_topo = r_learning_rate_topo
        self.learning_rate_topo = learning_rate_topo
        self.r_learning_rate_social = r_learning_rate_social
        self.learning_rate_social = learning_rate_social

        # Discretization config for goal (dx, dy)
        self.goal_xy_bins = int(goal_xy_bins) if int(goal_xy_bins) >= 2 else 2
        self.goal_xy_max_abs = float(goal_xy_max_abs)
        if goal_xy_edges is not None:
            edges = np.asarray(goal_xy_edges, dtype=np.float32).flatten()
            edges = edges[np.isfinite(edges)]
            edges = np.unique(edges)
            self.goal_xy_edges = edges
        else:
            self.goal_xy_edges = None

        # Discretization config for humans (x, y)
        self.human_xy_bins = int(human_xy_bins) if int(human_xy_bins) >= 2 else 2
        self.human_xy_max_abs = float(human_xy_max_abs)
        if human_xy_edges is not None:
            edges_h = np.asarray(human_xy_edges, dtype=np.float32).flatten()
            edges_h = edges_h[np.isfinite(edges_h)]
            edges_h = np.unique(edges_h)
            self.human_xy_edges = edges_h
        else:
            self.human_xy_edges = None

    def get_state_key(self, obs):
        """
        Discretize continuous features into hashable state keys using configurable bins.
        - Topo state: bins for goal (dx, dy)
        - Social state: bins for concatenated humans' (x, y) pairs
        """
        feat_topo, feat_social = self.get_features(obs)

        # Helper: build edges (ascending) or use provided override
        def build_edges(override, bins, max_abs):
            if override is not None:
                return override
            # produce (bins-1) cut points for bins bins
            count = max(int(bins) - 1, 1)
            return np.linspace(-max_abs, +max_abs, num=count, dtype=np.float32)

        def bin_scalar(x, edges, clip_abs):
            xx = float(np.clip(x, -clip_abs, +clip_abs))
            for i, e in enumerate(edges):
                if xx < e:
                    return int(i)
            return int(len(edges))

        # Discretize goal (dx, dy)
        goal_edges = build_edges(self.goal_xy_edges, self.goal_xy_bins, self.goal_xy_max_abs)
        topo_key = []
        if feat_topo is not None and len(feat_topo) >= 2:
            topo_key.append(bin_scalar(feat_topo[0], goal_edges, self.goal_xy_max_abs))
            topo_key.append(bin_scalar(feat_topo[1], goal_edges, self.goal_xy_max_abs))
        else:
            topo_key.extend([0, 0])
        topo_key = tuple(topo_key)

        # Discretize humans (x, y) pairs
        human_edges = build_edges(self.human_xy_edges, self.human_xy_bins, self.human_xy_max_abs)
        social_key_list = []
        if feat_social is not None and len(feat_social) >= 2:
            # ensure even count
            count = (len(feat_social) // 2) * 2
            for i in range(0, count, 2):
                social_key_list.append(bin_scalar(feat_social[i], human_edges, self.human_xy_max_abs))
                social_key_list.append(bin_scalar(feat_social[i + 1], human_edges, self.human_xy_max_abs))
        social_key = tuple(social_key_list)

        return topo_key, social_key

    def sample_action(self, state_key_topo, state_key_social):
        # Sample action using epsilon-greedy
        if eval:
            epsilon = 0
        else:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        # Compute Q-values: Q(s,a) = sum over s' of (SR_topo(s,a,s') * R_topo(s') + SR_social(s,a,s') * R_social(s'))
        Q_values = np.zeros(self.action_size)
        for a in range(self.action_size):
            q_value = 0.0
            # Topographic SR
            for next_state in self.SR_topo[state_key_topo][a]:
                reward_next_state_topo = np.mean([self.R_topo[next_state][a_prime] for a_prime in self.R_topo[next_state]])
                q_value += self.SR_topo[state_key_topo][a][next_state] * reward_next_state_topo
            # Social SR
            for next_state in self.SR_social[state_key_social][a]:
                reward_next_state_social = np.mean([self.R_social[next_state][a_prime] for a_prime in self.R_social[next_state]])
                q_value += self.SR_social[state_key_social][a][next_state] * reward_next_state_social
            Q_values[a] = q_value

        return np.argmax(Q_values)

    def update_sr(self, state_key_topo, state_key_social, action, next_state_key_topo, next_state_key_social, done, upd_social):
        if not done:
            next_action = self.sample_action(next_state_key_topo, next_state_key_social)
            relevant_s_primes_topo = set(self.SR_topo[state_key_topo][action].keys()) | set(self.SR_topo[next_state_key_topo][next_action].keys()) | {next_state_key_topo}
            if upd_social:
                relevant_s_primes_social = set(self.SR_social[state_key_social][action].keys()) | set(self.SR_social[next_state_key_social][next_action].keys()) | {next_state_key_social}
        else:
            next_action = None
            relevant_s_primes_topo = set(self.SR_topo[state_key_topo][action].keys()) | {next_state_key_topo}
            if upd_social:
                relevant_s_primes_social = set(self.SR_social[state_key_social][action].keys()) | {next_state_key_social}

        # Update topo SR
        for s_prime in tuple(relevant_s_primes_topo):
            indicator = 1.0 if s_prime == next_state_key_topo else 0.0
            if not done and next_action is not None:
                future_sr = self.SR_topo[next_state_key_topo][next_action][s_prime]
            else:
                future_sr = 0.0
            target = indicator + self.gamma * future_sr
            current_sr = self.SR_topo[state_key_topo][action][s_prime]
            td_error = target - current_sr
            self.SR_topo[state_key_topo][action][s_prime] += self.learning_rate_topo * td_error

        # Update social SR
        if upd_social:
            for s_prime in tuple(relevant_s_primes_social):
                indicator = 1.0 if s_prime == next_state_key_social else 0.0
                if not done and next_action is not None:
                    future_sr = self.SR_social[next_state_key_social][next_action][s_prime]
                else:
                    future_sr = 0.0
                target = indicator + self.gamma * future_sr
                current_sr = self.SR_social[state_key_social][action][s_prime]
                td_error = target - current_sr
                self.SR_social[state_key_social][action][s_prime] += self.learning_rate_social * td_error

    def update_reward(self, state_key_topo, state_key_social, action, reward, upd_social):
        # Update both topographic and social reward functions
        # Topographic reward update
        current_r_topo = self.R_topo[state_key_topo][action]
        td_error_topo = reward - current_r_topo
        self.R_topo[state_key_topo][action] += self.r_learning_rate_topo * td_error_topo

        if upd_social:
            # Social reward update
            current_r_social = self.R_social[state_key_social][action]
            td_error_social = reward - current_r_social
            self.R_social[state_key_social][action] += self.r_learning_rate_social * td_error_social

    def get_features(self, obs):
        """
        Return continuous features:
        - features_topo: np.array([dx, dy]) goal vector in robot frame (same definition as QLearning.get_state)
        - features_social: np.array([x1, y1, x2, y2, ...]) humans' positions in robot frame
        If obs is not an env dict, attempt legacy fallback; else return empty socials.
        """
        # Normalize possible (obs, info) wrapper
        obs_dict = None
        if isinstance(obs, (list, tuple)):
            if len(obs) > 0 and isinstance(obs[0], dict):
                obs_dict = obs[0]
        elif isinstance(obs, dict):
            obs_dict = obs

        # Extract goal (dx, dy) from "robot" observation: [one_hot(D), dx, dy, radius]
        r = np.asarray(obs_dict.get("robot", []), dtype=np.float32).flatten()
        D = max(0, r.size - 3)
        dx = float(r[D]) if r.size > D else 0.0
        dy = float(r[D + 1]) if r.size > D + 1 else 0.0
        features_topo = np.array([dx, dy], dtype=np.float32)

        # Extract all humans' (x, y) from "humans"
        humans_xy = []
        h = obs_dict.get("humans", None)
        if h is not None:
            h = np.asarray(h, dtype=np.float32).flatten()
            if h.size > 0:
                # Default per-human block: one_hot(6) + 8 = 14 (entity_obs_dim = 14 in env)
                one_hot_len = 6
                block = 14

                if h.size % block != 0:
                    # Try to infer one_hot length k such that (k + 8) divides total length and initial k looks one-hot-like
                    inferred = False
                    for k in range(3, 16):
                        bs = k + 8
                        if bs <= 0 or h.size % bs != 0:
                            continue
                        oh = h[:k]
                        if np.all((oh == 0) | (oh == 1)) and np.isclose(np.sum(oh), 1.0):
                            one_hot_len = k
                            block = bs
                            inferred = True
                            break
                    if not inferred and h.size >= 8:
                        # Minimal fallback: first (x, y) at offsets (6,7)
                        humans_xy.append(float(h[6]))
                        humans_xy.append(float(h[7]))
                        block = 0  # skip loop

                if block > 0 and h.size % block == 0:
                    for i in range(0, h.size, block):
                        base = i + one_hot_len
                        if base + 1 < h.size:
                            humans_xy.append(float(h[base]))
                            humans_xy.append(float(h[base + 1]))

        features_social = np.asarray(humans_xy, dtype=np.float32)
        return features_topo, features_social

    def act(self, env, obs, upd_social=True):
        state_key_topo, state_key_social = self.get_state_key(obs)
        
        episode_reward = 0
        episode_length = 0

        while True:

            # Select action
            action = self.sample_action(state_key_topo, state_key_social)
            
            # Take step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Get next state
            next_state_key_topo, next_state_key_social = self.get_state_key(next_state)
            
            episode_reward += reward
            
            
            self.update_sr(
                state_key_topo, state_key_social, action,
                next_state_key_topo, next_state_key_social, done or truncated, upd_social
            )
            self.update_reward(next_state_key_topo, next_state_key_social, action, reward, upd_social)
            
            # Move to next state
            state_key_topo = next_state_key_topo
            state_key_social = next_state_key_social
            
            if done or truncated:
                break
            episode_length += 1
            
        return episode_length, episode_reward
