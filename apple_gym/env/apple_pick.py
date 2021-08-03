import pdb
import gym
from gym import Wrapper
import numpy as np
import os
from diy_gym import DIYGym
from diy_gym.utils import flatten, unflatten, get_bounds_for_space
import flatten_dict
import pybullet as p
import logging
from functools import partial

logger = logging.getLogger(__file__)

# def logabsclip(x, eps=1e-3):
#     x = np.log(np.abs(x) + eps) / 10
#     return x.clip(-1, 1.)

# def pose2array(p):
#     """
#     Take the pose of some object then order and flatten.

#     Also make sure -1=>x<=1
#     """
#     # FIXME better to use obs_space limits
#     keys = [
#         ["position", lambda x: x/7.0],
#         ["rotation", lambda x: x/1.0],
#         ["velocity", lambda x: x/100.0],
#         ["angular_velocity", lambda x: x/5.0],
#         ["effort", lambda x: logabsclip(x)],
#         ["force", lambda x: logabsclip(x)],
#         ["torque", lambda x: logabsclip(x)],
#     ]
#     d = []
#     for k, f in keys:
#         if k in p:
#             x = f(np.array(p[k]))
#             d.append(x)
#             if np.abs(x).max() > 5:
#                 logger.warning(f'{k} {np.abs(x).max()}>5')
#             # assert np.abs(x).max()<=20, f'{k} {np.abs(x).max()}>20'
#     return np.stack(d).flatten() / 5.


# def _reduce_obs_space(obs_space):
#     """
#     diy-gym 2 openai-gym: observation space

#     Picking what we want and flattening. """
#     # FIXME better to init obj using obs space, then flatten using the obj
#     # flatten obs
#     obs_len = (
#         np.prod(obs_space["robot"]["pose_gripper"]["position"].shape) * 4 + # we concat 4 measurements
#         np.prod(obs_space["robot"]["joint_state"]["position"].shape) * 8
#         # np.prod(obs_space["robot"]["joint_torque"]["force"].shape) * 2
#     )
#     if "robot" in obs_space.spaces and "arm_camera" in obs_space["robot"].spaces:
#         if "features" in obs_space["robot"]["arm_camera"].spaces:
#             obs_len = obs_len + np.prod(
#                 obs_space["robot"]["arm_camera"]["features"].shape
#             )
#         else:
#             if "rgb" in obs_space["robot"]["arm_camera"].spaces:
#                 obs_len = obs_len + np.prod(obs_space["robot"]["arm_camera"]["rgb"].shape)
#             if "depth" in obs_space["robot"]["arm_camera"].spaces:
#                 obs_len = obs_len + np.prod(obs_space["robot"]["arm_camera"]["depth"].shape)
#     if "base_camera" in obs_space["apple_pick"].spaces:
#         if "features" in obs_space["apple_pick"]["base_camera"].spaces:
#             obs_len = obs_len + np.prod(
#                 obs_space["apple_pick"]["base_camera"]["features"].shape
#             )
#         else:
#             if "rgb" in obs_space["apple_pick"]["base_camera"].spaces:
#                 obs_len = obs_len + np.prod(obs_space["apple_pick"]["base_camera"]["rgb"].shape)
#             if "depth" in obs_space["apple_pick"]["base_camera"].spaces:
#                 obs_len = obs_len + np.prod(obs_space["apple_pick"]["base_camera"]["depth"].shape)

#     return gym.spaces.Box(-1.0, 1.0, shape=(obs_len,), dtype="float16")


# def _reduce_obs(obs, place_obj=False):
#     """
#     diy-gym 2 openai-gym: observation

#     We pick what we want the agent to see, and flatten

#     """

#     obs_a = [
#         pose2array(obs["robot"]["pose_gripper"]),
#         pose2array(obs["robot"]["joint_state"]),
#         # pose2array(obs["robot"]["joint_torque"]),
#     ]

#     camera = np.array([])
#     if "robot" in obs and "arm_camera" in obs["robot"]:
#         if "features" in obs["robot"]["arm_camera"]:
#             camera = obs["robot"]["arm_camera"][
#                 "features"
#             ].flatten()  # camera makes it 8x slower
#             obs_a.append(camera)
#         else:
#             if "rgb" in obs["robot"]["arm_camera"]:
#                 camera = obs["robot"]["arm_camera"]["rgb"].flatten()
#                 obs_a.append(camera)
#             if "depth" in obs["robot"]["arm_camera"]:
#                 camera = obs["robot"]["arm_camera"]["depth"].flatten()
#                 obs_a.append(camera)
#     if "base_camera" in obs["apple_pick"]:
#         if "features" in obs["apple_pick"]["base_camera"]:
#             camera = obs["apple_pick"]["base_camera"][
#                 "features"
#             ].flatten()  # camera makes it 8x slower
#             obs_a.append(camera)
#         else:
#             if "rgb" in obs["apple_pick"]["base_camera"]:
#                 camera = obs["apple_pick"]["base_camera"]["rgb"].flatten()
#                 obs_a.append(camera)
#             if "depth" in obs["apple_pick"]["base_camera"]:
#                 camera = obs["apple_pick"]["base_camera"]["depth"].flatten()
#                 obs_a.append(camera)

#     return np.concatenate(obs_a).astype(np.float16)


# def _expand_action(action):
#     """gym to diy-gym."""
#     return {"robot": {"controller": action}}


# def _reduce_action_space(action_space):
#     return gym.spaces.utils.flatten_space(action_space)
#     # return gym.spaces.Box(low=s.low[None, :], high=s.high[None, :], shape=(1,) + s.shape)


def normalize_box_fn(space: gym.spaces.Box):
    oh = space.high
    ol = space.low

    # we want to keep 0  position for most distributions
    # but also end up between -1 and 1
    # if [0, h] => [0, 1]
    # elif [-h, h] => [-1, 1]
    # else [l, h] => [-1, 1]

    # can do by modifying low when low is 0
    ol = np.where(ol==0, -oh, ol)

    # new space
    new_space = gym.spaces.Box(-1, 1, shape=space.shape)
    nl = new_space.low
    nh = new_space.high

    # return norm and unorm fn
    def norm_fn(x):
        # to [0, 1]
        x = (x-ol)/(oh-ol)
        # to [-1, 1]
        x2 = x * (nh-nl) + nl
        return x2#.clip(-1, 1)

    def unnorm_fn(x):
        x = (x-nl)/(nh-nl)
        return x * (oh-ol) + ol
    return new_space, norm_fn, unnorm_fn

def normalize_box(space, x):
    nspace, norm, unorm = normalize_box_fn(space)
    return nspace, norm(x), unorm(x)


def normalize_dict(space: gym.spaces.Dict, x=None):
    if x is None:
        x = get_bounds_for_space(space, 0)
    new_space = gym.spaces.Dict(
            {k: normalize_box(space[k], x[k])[0] for k in space.spaces.keys()}
    )
    new_x = {k: normalize_box(space[k], x[k])[1] for k in space.spaces.keys()}
    return new_space, new_x

class AppleWrap(Wrapper):
    """Wrapper to flatten obs, reward, action."""

    def __init__(self, env):
        super().__init__(env)
        self._max_episode_steps = env._max_episode_steps

        # we don't want obs to be nested dict, just dict
        low = flatten_dict.flatten(
            get_bounds_for_space(self.observation_space, 1), reducer="path"
        )
        high = flatten_dict.flatten(
            get_bounds_for_space(self.observation_space, 0), reducer="path"
        )
        self.raw_observation_space = gym.spaces.Dict(
            {k: gym.spaces.Box(low=low[k], high=high[k]) for k in low.keys()}
        )

        # we also want to normalize it
        self.observation_space = normalize_dict(self.raw_observation_space)[0]
        self.obs_normalize = partial(normalize_dict, self.raw_observation_space)
        # self._render = None

    def step(self, action):
        obs_raw, reward_raw, terminal_raw, info_raw = self.env.step(action)
        obs = self.transform_obs(obs_raw)
        terminal = flatten(terminal_raw).any()
        reward = flatten(reward_raw).sum() if len(reward_raw) else 0
        # reward scaling, if an important parameter. Here we aim for positive episode rewards ranging from 100-10000. See "Soft Actor-Critic Algorithms and Applications"
        reward = (reward + 1.0) * 10.0
        # self._render = obs["apple_pick/base_camera/rgb"]

        info = flatten_dict.flatten(
            {
                **info_raw,
                "env_reward": reward_raw,
                "env_terminal": terminal_raw,
                "env_obs": obs_raw,
            },
            reducer="path",
        )
        return obs, reward, terminal, info

    def render(self, mode="rgb_array", **kwargs):
        if mode == 'rgb_array':
            # see https://github.com/benelot/pybullet-gym/blob/master/pybulletgym/envs/roboschool/envs/env_bases.py#L73
            target_model = self.env.models['robot']
            target_frame_id = target_model.get_frame_id('ee_joint')
            base_pos = p.getLinkState(target_model.uid, target_frame_id)[4]
            base_pos = [0.0, 0.00, 0.50]
            height = kwargs.get('height', 720)
            width = kwargs.get('width', 1040)
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=2,
                yaw=kwargs.get('yaw', 0),
                pitch=-20,
                roll=0,
                upAxisIndex=2)
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=float(width)/height,
                nearVal=0.1, farVal=100.0)
            (_, _, px, _, _) = p.getCameraImage(
                width=width, height=height, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
                )
            rgb_array = np.array(px)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def transform_obs(self, obs):
        obs = flatten_dict.flatten(obs, reducer="path")
        obs = self.obs_normalize(obs)[1]
        for k in obs.keys():
            if not self.observation_space[k].contains(obs[k]):
                logger.error(f"obs.{k} should be in space. ob={obs[k]} space={self.observation_space[k]}")
            # assert self.observation_space[k].contains(
            #     obs[k]
            # ), f"obs.{k} should be in space. ob={obs[k]} space={self.observation_space[k]}"    
        # assert self.observation_space.contains(
        #     obs
        # ), f"obs should be in space. ob={obs} space={self.observation_space}"
        return obs

    def reset(self):
        obs = self.env.reset()
        return self.transform_obs(obs)


class NormAction(Wrapper):
    """Wrapper to flatten obs, reward, action."""

    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = self.env.action_space
        self.action_space, self.action_norm, self.action_unnorm = normalize_box_fn(self.original_action_space)
        self._max_episode_steps = env._max_episode_steps

    def step(self, action):
        action = self.action_unnorm(action)
        obs, reward, terminal, info = self.env.step(action)
        return obs, reward, terminal, info


# class AppleWrap(Wrapper):
#     """Wrapper to flatten obs, reward, action."""
#     def __init__(self, env, place_obj=False):
#         super().__init__(env)

#         self.place_obj = place_obj
#         # self.action_space = _reduce_action_space(env.action_space)
#         # self.observation_space = _reduce_obs_space(env.observation_space)
#         self._max_episode_steps = env._max_episode_steps
#         logger.debug(f"observation_space {self.observation_space}, action_space = {self.action_space}")

#     def step(self, action):
#         # action = _expand_action(action)
#         obs_raw, reward_raw, terminal_raw, info_raw = self.env.step(action)

#         obs = _reduce_obs(obs_raw, self.place_obj)
#         terminal = flatten(terminal_raw).any()
#         reward = flatten(reward_raw).sum() if len(reward_raw) else 0
#         # reward scaling, if an important parameter. Here we aim for positive episode rewards ranging from 100-10000. See "Soft Actor-Critic Algorithms and Applications"
#         reward = (reward + 1.) * 10.
#         info = flatten_dict.flatten(
#             {**info_raw, "env_reward": reward_raw, "env_terminal": terminal_raw, "env_obs": obs_raw},
#             reducer="path",
#         )
#         return obs, reward, terminal, info

#     def reset(self, **kwargs):
#         obs = self.env.reset(**kwargs)
#         return obs

#     def render(self, mode="human", **kwargs):
#         # Would need to restart env
#         # return self.env.render(mode, **kwargs)
#         return None

#     def close(self):
#         return self.env.close()

#     def seed(self, seed=None):
#         return self.env.seed(seed)

# TODO flatten obs wrapper


def apple_pick_env(*args, **kwargs):
    """
    Makes a wrapped diy gym env.
    Then wraps it in an openai wrapper the flattens and selects parts, and computes a complex reward
    """
    cur_path = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(cur_path, "apple_pick.yaml")
    env = NormAction(AppleWrap(DIYGym(config_file, *args, **kwargs)))

    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    return env


def apple_pick_env_hard(*args, **kwargs):
    """
    Makes a wrapped diy gym env.
    Then wraps it in an openai wrapper the flattens and selects parts, and computes a complex reward
    """
    cur_path = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(cur_path, "apple_pick_hard.yaml")
    env = NormAction(AppleWrap(DIYGym(config_file, *args, **kwargs)))

    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    return env
