# SocNavGym — Extended for Thesis Research

This repository is a fork of [SocNavGym](https://github.com/gnns4hri/SocNavGym) (IEEE ROMAN 2023), extended as part of a PhD thesis on social robot navigation. The original environment documentation is preserved in [README_SocNavGym.md](README_SocNavGym.md).

---

## Overview of Changes

The original SocNavGym provides a Gymnasium-compliant RL environment for social navigation, with humans following basic velocity-obstacle (RVO2) policies. This fork adds more realistic human motion through global path planning, new observation wrappers designed for learning experiments, and visual debugging tools for observations.

---

## Key Improvements

### 1. Global Path Planning for Humans (PRM)

In the original SocNavGym, humans navigate directly toward their goal without any awareness of static obstacles such as walls or furniture. This fork integrates a **Probabilistic Road Map (PRM)** planner (adapted from [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)) so that both individual humans and crowd groups plan collision-free paths around static obstacles before executing local RVO2 avoidance.

- Individual humans now compute a global waypoint path at the start of each episode.
- Crowd groups (human-human interactions) also use PRM to plan a global trajectory together.
- Local RVO2 avoidance is layered on top of the global plan, giving more realistic motion.

---

### 2. New Wrappers

The original SocNavGym ships with four wrappers (`DiscreteActions`, `NoisyObservations`, `PartialObservations`, `WorldFrameObservations`). This fork adds two more:

#### `ExpertObservations`

Reduces the full dictionary observation to just two entries: the robot's goal position and orientation (`robot`: 3 values), and the closest human's position in the robot frame (`humans`: 2 values). Action space is 7 discrete actions: turn CCW, turn CW, forward+CCW, forward+CW, forward, backward, no-op.

#### `CardinalActions`

Maps 5 discrete actions to the continuous holonomic action space: strafe right, strafe left, move forward, move backward, no-op.

---

### 3. Observation Rendering / Debugging Tools

The original SocNavGym only provides a world-level render. Two additional utilities visualize what the agent *sees* (the observation), useful for debugging observation preprocessing and verifying correctness.

- **`ObsGridRenderer`** (`obs_grid_render.py`): Renders a discretized/binned observation as a 2D grid in the robot frame. Draws entity positions by bin index, useful when using a grid-based state representation.
- **`ContinuousObsRenderer`** (`continuous_obs_render.py`): Renders the raw continuous dictionary observation as a local robot-frame panel (entities plotted at their exact robot-frame coordinates). Optionally composites the world render side-by-side for comparison.

---

## Installation

Follow the original installation instructions in [README_SocNavGym.md](README_SocNavGym.md). No additional dependencies are required beyond the base SocNavGym dependencies.

## Quick Start

```python
import socnavgym
import gymnasium as gym
from socnavgym.wrappers.expert_observations import ExpertObservations

env = gym.make("SocNavGym-v1", config="./environment_configs/exp1_no_sngnn.yaml")
env = ExpertObservations(env)

obs, _ = env.reset(seed=42)
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
```

## Original Repository

[https://github.com/gnns4hri/SocNavGym](https://github.com/gnns4hri/SocNavGym)

> Kapoor, A., Swamy, S., Manso, L., & Bachiller, P. (2023). *SocNavGym: A Reinforcement Learning Gym for Social Navigation*. IEEE ROMAN 2023.
