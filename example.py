from tqdm.auto import tqdm
import gym
from itertools import count

# register env
import apple_gym.env

env = gym.make("ApplePick-v0", render=True)

observation = env.reset()
for i in tqdm(count()):
    action = env.action_space.sample()
    observation, reward, terminal, info = env.step(action)
    if terminal:
        observation = env.reset()
