from gym.envs.registration import register

register(
    id='PandaMoveBox-v0',
    entry_point='panda_grasp.env:PandaMoveBoxEnv',
    max_episode_steps=1000,
)
