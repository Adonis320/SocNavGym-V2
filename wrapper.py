import gym
import gymnasium as gymnasium
import numpy as np

class SB3CompatibleEnv(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

        # Convert action space
        if isinstance(env.action_space, gymnasium.spaces.Discrete):
            self.action_space = gym.spaces.Discrete(env.action_space.n)
        else:
            raise TypeError(f"Unsupported action space: {type(env.action_space)}")

        # Convert observation space
        if isinstance(env.observation_space, gymnasium.spaces.Box):
            low, high = np.array(env.observation_space.low, np.float32), np.array(env.observation_space.high, np.float32)
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=env.observation_space.shape, dtype=np.float32)
        else:
            raise TypeError(f"Unsupported obs space: {type(env.observation_space)}")

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs.astype(np.float32)          # <-- return only obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs.astype(np.float32), float(reward), done, info
