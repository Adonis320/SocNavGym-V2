import numpy as np
from collections import defaultdict
import math

class SR():
    def __init__(self, action_size, epsilon=0.05, gamma=0.99, learning_rate=0.01, r_learning_rate=0.01, xy_bins=30, xy_max_abs=10.0, xy_edges=None, human_xy_bins=30, human_xy_max_abs=10.0, human_xy_edges=None):
        self.action_size = action_size
        
        # Use dictionaries for dynamic state space
        # SR[state][action][next_state]
        self.SR = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Reward function R[state][action] = expected immediate reward
        self.R = defaultdict(lambda: defaultdict(float))
        
        self.epsilon = epsilon
        self.gamma = gamma
        self.r_learning_rate = r_learning_rate
        self.learning_rate = learning_rate

        # State discretization params for (x, y)
        self.xy_bins = int(xy_bins) if int(xy_bins) >= 2 else 2  # number of bins per axis (uniform if xy_edges is None)
        self.xy_max_abs = float(xy_max_abs)  # clip range [-xy_max_abs, +xy_max_abs] (uniform if xy_edges is None)
        # Optional custom bin edges (ascending). If provided, overrides xy_bins/xy_max_abs.
        if xy_edges is not None:
            edges = np.asarray(xy_edges, dtype=np.float32).flatten()
            edges = edges[np.isfinite(edges)]  # keep only finite
            edges = np.unique(edges)  # sort and dedupe
            self.xy_edges = edges
        else:
            self.xy_edges = None

        # Humans discretization config (independent from goal)
        self.human_xy_bins = int(human_xy_bins) if int(human_xy_bins) >= 2 else 2
        self.human_xy_max_abs = float(human_xy_max_abs)
        if human_xy_edges is not None:
            hedges = np.asarray(human_xy_edges, dtype=np.float32).flatten()
            hedges = hedges[np.isfinite(hedges)]
            hedges = np.unique(hedges)
            self.human_xy_edges = hedges
        else:
            self.human_xy_edges = None

    def get_state_key(self, obs):
        # Convert observation to a stable, hashable state key
        state = self.get_state(obs)
        # If get_state returns a numpy array, cast to ints and tuple
        if isinstance(state, np.ndarray):
            try:
                return tuple(state.astype(int).tolist())
            except Exception:
                return tuple(np.asarray(state).flatten().tolist())
        # If it's already a tuple/list (as in QLearning), just make a tuple of ints
        try:
            return tuple(int(x) for x in state)
        except Exception:
            return tuple(state)
    
    def sample_action(self, state_key, eval=False):
        # Sample action using epsilon-greedy
        if eval:
            epsilon = 0
        else:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(self.action_size)
        
        # Compute Q-values: Q(s,a) = sum over s' of SR(s,a,s') * R(s')
        Q_values = np.zeros(self.action_size)
        for a in range(self.action_size):
            q_value = 0.0
            for next_state in self.SR[state_key][a]:
                reward_next_state = np.mean([self.R[next_state][a_prime] for a_prime in self.R[next_state]])
                q_value += self.SR[state_key][a][next_state] * reward_next_state
            Q_values[a] = q_value

        return np.argmax(Q_values)

    def update_sr(self, state_key, action, next_state_key, done):
        if not done:
            next_action = self.sample_action(next_state_key)
            relevant_s_primes = set(self.SR[state_key][action].keys()) | set(self.SR[next_state_key][next_action].keys()) | {next_state_key}
        else:
            next_action = None
            relevant_s_primes = set(self.SR[state_key][action].keys()) | {next_state_key}

        for s_prime in tuple(relevant_s_primes):
            indicator = 1.0 if s_prime == next_state_key else 0.0
            if not done and next_action is not None:
                future_sr = self.SR[next_state_key][next_action][s_prime]
            else:
                future_sr = 0.0
            target = indicator + self.gamma * future_sr
            current_sr = self.SR[state_key][action][s_prime]
            td_error = target - current_sr
            self.SR[state_key][action][s_prime] += self.learning_rate * td_error

    def update_reward(self, state_key, action, reward):
        # Update reward function
        current_r = self.R[state_key][action]
        td_error = reward - current_r
        self.R[state_key][action] += self.r_learning_rate * td_error

    def get_state(self, obs):
        """
        Build a hashable state using discretized bins for:
        - Goal (dx, dy) in robot frame (same semantics as before)
        - Humans' (x, y) in robot frame (concatenated), using separate bin config
        Falls back to legacy behavior if obs is not a dict.
        """
        # Normalize input: sometimes callers pass (obs, info) or list-like
        if isinstance(obs, (list, tuple)) and len(obs) > 0:
            if isinstance(obs[0], dict):
                obs = obs[0]

        # If not a dict, fallback to hashing raw obs
        if not isinstance(obs, dict):
            try:
                arr = np.asarray(obs, dtype=np.float32)
                return ("raw", hash(arr.tobytes()))
            except Exception:
                return ("raw", str(obs))

        # Extract goal (dx, dy) from robot observation
        robot = obs.get("robot", None)
        if robot is None:
            return ("raw", str(obs))
        r = np.asarray(robot, dtype=np.float32).flatten()
        # Robot obs format: [one_hot(D), goal_dx, goal_dy, robot_radius] with total length D+3
        D = max(0, r.size - 3)
        dx = float(r[D]) if r.size > D else 0.0
        dy = float(r[D + 1]) if r.size > D + 1 else 0.0

        # Helper: bin a scalar using ascending edges (open on right)
        def bin_scalar(x, edges: np.ndarray) -> int:
            for i, e in enumerate(edges):
                if x < e:
                    return i
            return len(edges)

        # Goal discretization
        goal_edges = self.xy_edges if self.xy_edges is not None else np.linspace(-self.xy_max_abs, self.xy_max_abs, num=max(self.xy_bins - 1, 1), dtype=np.float32)
        gx = bin_scalar(np.clip(dx, -self.xy_max_abs, self.xy_max_abs), goal_edges)
        gy = bin_scalar(np.clip(dy, -self.xy_max_abs, self.xy_max_abs), goal_edges)

        # Humans extraction: concatenated one-hot + 8 metrics per human (entity_obs_dim=14 in env)
        humans_bins = []
        humans = obs.get("humans", None)
        if humans is not None:
            h = np.asarray(humans, dtype=np.float32).flatten()
            if h.size > 0:
                one_hot_len = 6
                block = 14
                if h.size % block != 0:
                    # Try to infer one_hot length k so that (k+8) divides total and first k looks one-hot-like
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
                        # Minimal fallback: single (x,y) at offsets (6,7)
                        hx, hy = float(h[6]), float(h[7])
                        hedges = self.human_xy_edges if self.human_xy_edges is not None else np.linspace(-self.human_xy_max_abs, self.human_xy_max_abs, num=max(self.human_xy_bins - 1, 1), dtype=np.float32)
                        humans_bins.append(bin_scalar(np.clip(hx, -self.human_xy_max_abs, self.human_xy_max_abs), hedges))
                        humans_bins.append(bin_scalar(np.clip(hy, -self.human_xy_max_abs, self.human_xy_max_abs), hedges))
                        block = 0  # skip loop
                if block > 0 and h.size % block == 0:
                    hedges = self.human_xy_edges if self.human_xy_edges is not None else np.linspace(-self.human_xy_max_abs, self.human_xy_max_abs, num=max(self.human_xy_bins - 1, 1), dtype=np.float32)
                    for i in range(0, h.size, block):
                        base = i + one_hot_len  # x at base, y at base+1
                        if base + 1 < h.size:
                            hx = float(h[base])
                            hy = float(h[base + 1])
                            humans_bins.append(bin_scalar(np.clip(hx, -self.human_xy_max_abs, self.human_xy_max_abs), hedges))
                            humans_bins.append(bin_scalar(np.clip(hy, -self.human_xy_max_abs, self.human_xy_max_abs), hedges))

        # Final state tuple: (goal bins, then human bins)
        state = tuple([int(gx), int(gy)] + [int(b) for b in humans_bins])
        return state

    def act(self, env, obs, eval=False):
        episode_reward = 0
        episode_length = 0

        # Initialize state
        state_key = self.get_state_key(obs)

        while True:
            # Select action (epsilon-greedy)
            action = self.sample_action(state_key, eval=eval)
            
            # Take step
            next_obs, reward, done, truncated, info = env.step(action)

            # Get next state
            next_state_key = self.get_state_key(next_obs)
            
            episode_reward += reward
            
            # Update SR and reward function
            self.update_sr(state_key, action, next_state_key, done or truncated)
            # Reward is associated with (state, action)
            self.update_reward(state_key, action, reward)
            
            # Move to next state
            state_key = next_state_key
            
            if done or truncated:
                break
            episode_length += 1
            
        return episode_length, episode_reward
