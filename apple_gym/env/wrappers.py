import gym
from collections import deque
import numpy as np
from diy_gym.utils import get_bounds_for_space


class FrameStack(gym.Wrapper):
    """Frame stack, of dict obs space, for chosen keys."""

    def __init__(
        self,
        env,
        n,
        keys=["apple_pick/base_camera/depth", "apple_pick/base_camera/rgb"],
    ):
        gym.Wrapper.__init__(self, env)
        self._n = n
        self.keys = keys
        self._frames = {k: deque([], maxlen=self._n) for k in keys}

        def stack_space(space):
            shp = space.shape
            return gym.spaces.Box(
                low=np.min(space.low),
                high=np.max(space.high),
                shape=((shp[0] * n,) + shp[1:]),
                dtype=space.dtype,
            )

        self.observation_space = gym.spaces.Dict(
            {
                k: stack_space(v) if k in keys else v
                for k, v in env.observation_space.spaces.items()
            }
        )
        self._max_episode_steps = env._max_episode_steps

    def append(self, obs):
        for k in self.keys:
            self._frames[k].append(obs[k])

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._n):
            self.append(obs)
        return self._get_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.append(obs)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        obs = obs.copy()
        for k in self.keys:
            assert len(self._frames[k]) == self._n
            obs[k] = np.concatenate(list(self._frames[k]), axis=0)
        return obs


class PermuteImages(gym.Wrapper):
    """Permutes images."""

    def __init__(
        self,
        env,
        axes=(2, 0, 1),
        keys=["apple_pick/base_camera/depth", ],
    ):
        gym.Wrapper.__init__(self, env)
        self.keys = keys
        self.axes = axes

        def permute_space(space: gym.spaces.Box):
            x = space.sample()
            x2 = np.transpose(x, axes)
            return gym.spaces.Box(
                low=np.min(space.low),
                high=np.max(space.high),
                shape=x2.shape,
                dtype=space.dtype,
            )

        self.observation_space = gym.spaces.Dict(
            {
                k: permute_space(v) if k in keys else v
                for k, v in env.observation_space.spaces.items()
            }
        )
        self._max_episode_steps = env._max_episode_steps

    def transform_obs(self, obs):
        for k in self.keys:
            obs[k] = np.transpose(obs[k], self.axes)
        return obs

    def reset(self):
        return self.transform_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.transform_obs(obs)
        return obs, reward, done, info


class ImageState(gym.Wrapper):
    """Combines images and state"""

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        s = env.observation_space.spaces
        # s_img = [v for k,v in s.items() if 'camera' in k]
        # s_state = [v for k, v in s.items() if 'camera' not in k]

        high = np.concatenate(
            [get_bounds_for_space(v, 0) for k, v in s.items() if "camera" in k], -1
        )
        low = np.concatenate(
            [get_bounds_for_space(v, 1) for k, v in s.items() if "camera" in k], -1
        )
        space_img = gym.spaces.Box(low=low, high=high)

        high = np.concatenate(
            [get_bounds_for_space(v, 0) for k, v in s.items() if "camera" not in k], -1
        )
        low = np.concatenate(
            [get_bounds_for_space(v, 1) for k, v in s.items() if "camera" not in k], -1
        )
        space_state = gym.spaces.Box(low=low, high=high)

        self.observation_space = gym.spaces.Dict(
            {"img": space_img, "state": space_state}
        )
        self._max_episode_steps = env._max_episode_steps

    def transform_obs(self, obs):
        img = np.concatenate([v for k, v in obs.items() if "camera" in k], -1)
        state = np.concatenate([v for k, v in obs.items() if "camera" not in k], -1)
        return {"img": img, "state": state}

    def reset(self):
        return self.transform_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.transform_obs(obs)
        return obs, reward, done, info
