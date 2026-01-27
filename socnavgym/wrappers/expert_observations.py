import gymnasium as gym
from gymnasium import spaces
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1
from socnavgym.envs.utils.wall import Wall
from socnavgym.envs.utils.utils import w2px, w2py
import sys
from typing import Dict
import numpy as np
import copy
import cv2
from gymnasium.spaces import Discrete

class ExpertObservations(gym.Wrapper):
    def __init__(self, env: SocNavEnv_v1) -> None:
        super().__init__(env)
        self.env = env
        self.action_space = Discrete(7)

    def step(self, action_pre):
        obs, reward, terminated, truncated, info = self.env.step(action_pre)
        self.latest_obs = obs
        obs = self.get_expert_obs(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.latest_obs = obs
        obs = self.get_expert_obs(obs)
        return obs, info
    
    @property
    def observation_space(self):
        """
        Observation space includes the goal, and the world frame coordinates and speeds (linear & angular) of all the objects (including the robot) in the scenario
        
        Returns:
        gym.spaces.Dict : the observation space of the environment
        """

        d = {
            "robot": spaces.Box(
                low=np.array([
                    -self.unwrapped.MAP_X * np.sqrt(2),
                    -self.unwrapped.MAP_Y * np.sqrt(2),
                    -np.pi
                ], dtype=np.float32),
                high=np.array([
                    +self.unwrapped.MAP_X * np.sqrt(2),
                    +self.unwrapped.MAP_Y * np.sqrt(2),
                    +np.pi
                ], dtype=np.float32),
                shape=(3,),
                dtype=np.float32
            )
        }

        
        d["humans"] = spaces.Box(
            low=np.array([
                -self.unwrapped.MAP_X * np.sqrt(2),
                -self.unwrapped.MAP_Y * np.sqrt(2),
            ], dtype=np.float32),
            high=np.array([
                +self.unwrapped.MAP_X * np.sqrt(2),
                +self.unwrapped.MAP_Y * np.sqrt(2),
            ], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        return spaces.Dict(d)

    
    def get_expert_obs(self, obs):
        d = {}

        robot = obs.get("robot", None)
        if robot is not None:
            r = np.asarray(robot, dtype=np.float32)
            x = r[6]
            y = r[7]
            
            theta_raw = None
            if self.env is not None:
                # keep your original access pattern
                try:
                    theta_raw = self.env.env.env.env.robot.orientation
                except Exception:
                    # fallback attempts
                    try:
                        theta_raw = self.env.robot.orientation
                    except Exception:
                        theta_raw = None

            # robot = [x, y, theta]
            d["robot"] = np.array([x, y, theta_raw], dtype=np.float32)
        else:
            d["robot"] = None

        # humans: closest (hx, hy) only
        h = obs.get("humans", None)
        if h is None:
            d["humans"] = np.zeros((2,), dtype=np.float32)
        else:
            xy = self._decode_closest_human_xy(h)
            if xy is None:
                d["humans"] = np.zeros((2,), dtype=np.float32)
            else:
                d["humans"] = np.array([xy[0], xy[1]], dtype=np.float32)

        return d

    def _decode_closest_human_xy(self, humans_array: np.ndarray) -> tuple[float, float] | None:
        """
        Decode humans array and return (hx, hy) for the closest human to the robot (0,0),
        based on min hx^2 + hy^2.

        Supports the same assumptions as your original code:
        - default one_hot_len=6, block=14
        - inference attempt if size mismatch
        - minimal fallback: assume single human with (x,y) at indices (6,7)

        Returns None if cannot decode or no humans.
        """
        h = np.asarray(humans_array, dtype=np.float32).flatten()
        if h.size == 0:
            return None

        one_hot_len = 6
        block = 14

        # Try to infer encoding if size doesn't match
        if h.size % block != 0:
            inferred = False
            for k in range(3, 16):
                bs = k + 8
                if bs <= 0 or h.size % bs != 0:
                    continue
                oh = h[:k]
                # crude one-hot check
                if np.all((oh == 0) | (oh == 1)) and np.isclose(np.sum(oh), 1.0):
                    one_hot_len = k
                    block = bs
                    inferred = True
                    break

            if not inferred:
                # Minimal fallback: assume single (x, y) at indices (6,7)
                if h.size >= 8:
                    return float(h[6]), float(h[7])
                return None

        # Normal case: h.size is multiple of block
        if h.size % block != 0:
            return None

        best_d2 = float("inf")
        best_xy = None

        for i in range(0, h.size, block):
            base = i + one_hot_len  # x at base, y at base + 1
            if base + 1 >= h.size:
                continue
            hx = float(h[base])
            hy = float(h[base + 1])
            d2 = hx * hx + hy * hy
            if d2 < best_d2:
                best_d2 = d2
                best_xy = (hx, hy)

        return best_xy
    
    def discrete_to_continuous_action(self, action:int):
        # Turning anti-clockwise
        if action == 0:
            return np.array([0, 0.0, 1.0], dtype=np.float32) 
        # Turning clockwise
        elif action == 1:
            return np.array([0, 0.0, -1.0], dtype=np.float32) 
        # Turning anti-clockwise and moving forward
        elif action == 2:
            return np.array([1, 0.0, 1.0], dtype=np.float32) 
        # Turning clockwise and moving forward
        elif action == 3:
            return np.array([1, 0.0, -1.0], dtype=np.float32) 
        # Move forward
        elif action == 4:
            return np.array([1, 0.0, 0.0], dtype=np.float32)
        # Move backward
        elif action == 5:
            return np.array([-1, 0.0, 0.0], dtype=np.float32)
        # No Op
        elif action == 6:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            raise NotImplementedError

    def action(self, action):
        return self.discrete_to_continuous_action(action)
    