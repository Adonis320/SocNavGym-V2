import numpy as np
from collections import defaultdict
import math

class QL():
    def __init__(self, action_size=7, epsilon=0.05, gamma=0.99, learning_rate=0.01):
        self.action_size = action_size
        self.q_values = defaultdict(lambda: np.zeros(action_size))
        self.gamma = 0.99
        self.learning_rate = 0.2
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def sample_action(self, state, eval=False):
        # Samples action using epsilon-greedy
        if eval:
            epsilon = 0
        else:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = int(np.argmax(self.q_values[state]))
        return action
    
    def get_state(self, obs):
        # Discretize observation dict into a compact, hashable state for Q-learning
        # Handles padded/unpadded observations and missing keys gracefully.
        import numpy as np

        # Normalize input: sometimes callers pass (obs, info) or list-like
        if isinstance(obs, (list, tuple)):
            if len(obs) > 0:
                obs = obs[0]
        # If not a dict, fallback to hashing raw obs
        if not isinstance(obs, dict):
            try:
                return ("raw", hash(np.asarray(obs).tobytes()))
            except Exception:
                return ("raw", str(obs))

        robot = obs.get("robot", None)
        if robot is None:
            return ("raw", str(obs))

        r = np.asarray(robot, dtype=np.float32).flatten()
        # Robot obs format in env: [one_hot(D), goal_dx, goal_dy, robot_radius] with total length D+3
        D = max(0, r.size - 3)

        # Goal vector in robot frame
        dx = float(r[D]) if r.size > D else 0.0
        dy = float(r[D + 1]) if r.size > D + 1 else 0.0

        # Helper: bin a scalar using ascending edges (open on right)
        def bin_scalar(x, edges: np.ndarray) -> int:
            for i, e in enumerate(edges):
                if x < e:
                    return i
            return len(edges)

        # Discretize goal components using symmetric bins around 0
        goal_edges = np.array([-8.0, -3.0, -1.0, 1.0, 3.0, 8.0], dtype=np.float32)  # 7 bins total
        gx = bin_scalar(np.clip(dx, -10.0, 10.0), goal_edges)
        gy = bin_scalar(np.clip(dy, -10.0, 10.0), goal_edges)

        # Helper: parse a flat entity array into chunks of size (D+8) per entity
        # Entity format: [one_hot(D), rel_x, rel_y, sin(dth), cos(dth), radius, rel_v_lin, rel_v_ang, gaze]
        def parse_chunks(arr):
            if arr is None:
                return np.zeros((0, D + 8), dtype=np.float32)
            a = np.asarray(arr, dtype=np.float32).flatten()
            stride = D + 8
            if stride <= 0 or a.size == 0:
                return np.zeros((0, stride), dtype=np.float32)
            n = a.size // stride
            if n <= 0:
                return np.zeros((0, stride), dtype=np.float32)
            return a[: n * stride].reshape(n, stride)

        humans_chunks = parse_chunks(obs.get("humans", None))
        walls_chunks = parse_chunks(obs.get("walls", None))

        # Select up to K nearest humans by distance
        H = []
        if humans_chunks.shape[0] > 0:
            rel = humans_chunks[:, D : D + 2]  # (x, y)
            dist = np.linalg.norm(rel, axis=1)
            one_hot = humans_chunks[:, :D] if D > 0 else np.zeros((humans_chunks.shape[0], 0), dtype=np.float32)
            gaze = humans_chunks[:, D + 7]
            # Valid if any one-hot active OR non-zero distance OR gaze active (to ignore padded zeros)
            valid = (np.any(one_hot > 0, axis=1) if D > 0 else np.zeros_like(dist, dtype=bool)) | (dist > 1e-6) | (gaze > 0.5)
            if np.any(valid):
                K = 2
                sel_idx = np.argsort(dist[valid])[:K]
                sel = humans_chunks[valid][sel_idx]
                rel_sel = sel[:, D : D + 2]
                dist_sel = np.linalg.norm(rel_sel, axis=1)
                ang_sel = np.arctan2(rel_sel[:, 1], rel_sel[:, 0])
                dist_edges = np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float32)  # 6 bins
                for d, a, gz in zip(dist_sel, ang_sel, sel[:, D + 7]):
                    di = bin_scalar(d, dist_edges)
                    # Map angle [-pi, pi) to 8 equal sectors
                    sector = int(np.floor(((a + np.pi) / (2.0 * np.pi)) * 8.0)) % 8
                    H.append((int(di), int(sector), int(gz > 0.5)))

        # Pad with placeholders if fewer than K humans found
        while len(H) < 2:
            H.append((-1, -1, 0))

        # Nearest wall distance bin (walls only present when observations are not padded)
        W = -1
        if walls_chunks.shape[0] > 0:
            rel = walls_chunks[:, D : D + 2]
            dist = np.linalg.norm(rel, axis=1)
            valid = dist > 1e-6
            if np.any(valid):
                # index within filtered array
                local_idx = int(np.argmin(dist[valid]))
                d = float(np.linalg.norm(walls_chunks[valid][local_idx, D : D + 2]))
                W = int(bin_scalar(d, np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float32)))

        # Compact, hashable state tuple
        state = (int(gx), int(gy), H[0][0], H[0][1], H[0][2], H[1][0], H[1][1], H[1][2], W)
        return state
    
    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs
    ):
        # Update the Q-value of an action
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.gamma * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.learning_rate * temporal_difference
        )
        return temporal_difference

    def act(self, env, obs):
        state = self.get_state(obs)
        state = state
        total_reward = 0
        episode_length = 0

        while True:
            # Epsilon-greedy action selection
            action = self.sample_action(state, eval)
            # Take action
            next_state, reward, done, truncated, info = env.step(action)

            # Update Humanoid actions

            next_state =  self.get_state(next_state)
            next_state = next_state
            
            total_reward += reward
            
            self.update(state, action, reward, done, next_state)
            if done or truncated:
                break
            episode_length += 1
            state = next_state

        return episode_length, total_reward
