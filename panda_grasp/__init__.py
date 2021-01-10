from gym.envs.registration import register
from .env import PandaMoveBoxEnv

register(
    id='PandaMoveBox-v0',
    entry_point='panda_grasp.env:PandaMoveBoxEnv',
    max_episode_steps=PandaMoveBoxEnv().max_episode_steps,
)