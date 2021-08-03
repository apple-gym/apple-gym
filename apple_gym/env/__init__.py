import gym
import pybullet as p
import pybullet_data
import os
from pathlib import Path
from diy_gym.model import urdf_path
from .apple_pick import apple_pick_env
from .apple_pick_img import apple_pick_env_img

def add_data_path():
    """add data path to diy gym."""
    apple_gym_path = Path(__file__).parent.parent.parent
    p = str(apple_gym_path / "data")
    urdf_path.append(p)
    urdf_path.append(str(apple_gym_path))


gym.envs.register(
    id="ApplePick-v0",
    entry_point="apple_gym.env.apple_pick:apple_pick_env",
)

gym.envs.register(
    id="ApplePickImg-v0",
    entry_point="apple_gym.env.apple_pick_img:apple_pick_env_img",
)

# More trees, more variation
gym.envs.register(
    id="ApplePickHard-v0",
    entry_point="apple_gym.env.apple_pick:apple_pick_env_hard",
)
add_data_path()
