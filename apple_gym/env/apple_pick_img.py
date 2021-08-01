"""img only"""
import gym
from gym import Wrapper
import numpy as np
import os
from diy_gym import DIYGym
from diy_gym.utils import flatten, unflatten
import flatten_dict
import pybullet as p
import logging
logger = logging.getLogger(__file__)


def _reduce_obs_space(obs_space):
    """
    diy-gym 2 openai-gym: observation space

    Picking what we want and flattening. """
    h, w, c_rgb = obs_space["apple_pick_img"]["base_camera"]["rgb"].shape
    # _, _, c_d = obs_space["apple_pick_img"]["base_camera"]["depth"].shape
    return gym.spaces.Box(-1.0, 1.0, shape=(c_rgb+1, h, w,), dtype="float16")


def _reduce_obs(obs, place_obj=False):
    """
    diy-gym 2 openai-gym: observation

    We pick what we want the agent to see, and flatten

    """
    a = obs["apple_pick_img"]["base_camera"]["rgb"]
    b = obs["apple_pick_img"]["base_camera"]["depth"][..., None]
    obs = np.concatenate([a, b], -1).astype(np.float16)
    return np.transpose(obs, (2, 0, 1))


def _expand_action(action):
    """gym to diy-gym."""
    return {"robot": {"controller": action}}


def _reduce_action_space(action_space):
    return gym.spaces.utils.flatten_space(action_space)


class AppleWrap(Wrapper):
    """Wrapper to flatten obs, reward, action."""
    def __init__(self, env, place_obj=False):
        super().__init__(env)

        self.place_obj = place_obj
        self.action_space = _reduce_action_space(env.action_space)
        self.observation_space = _reduce_obs_space(env.observation_space)
        self._max_episode_steps = env._max_episode_steps
        logger.debug(f"observation_space {self.observation_space}, action_space = {self.action_space}")

    def step(self, action):
        action = _expand_action(action)
        obs_raw, reward_raw, terminal_raw, info_raw = self.env.step(action)

        obs = _reduce_obs(obs_raw, self.place_obj)
        terminal = flatten(terminal_raw).any()
        reward = flatten(reward_raw).sum() if len(reward_raw) else 0
        # reward scaling, if an important parameter. Here we aim for positive episode rewards ranging from 100-10000. See "Soft Actor-Critic Algorithms and Applications"
        reward = (reward + 1.) * 10.
        info = flatten_dict.flatten(
            {**info_raw, "env_reward": reward_raw, "env_terminal": terminal_raw, "env_obs": obs_raw},
            reducer="path",
        )
        return obs, reward, terminal, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return _reduce_obs(obs, self.place_obj)

    def render(self, mode="human", **kwargs):
        # Would need to restart env
        # return self.env.render(mode, **kwargs)
        return None

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)


def apple_pick_env_img(*args, **kwargs):
    """
    Makes a wrapped diy gym env.
    Then wraps it in an openai wrapper the flattens and selects parts, and computes a complex reward
    """
    cur_path = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(cur_path, "apple_pick_img.yaml")
    env = AppleWrap(DIYGym(config_file, *args, **kwargs), place_obj=True)

    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    return env
