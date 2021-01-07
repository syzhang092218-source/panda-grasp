from gym.envs.registration import register

register(
    id='PandaMoveBox-v0',
    entry_point='panda_move_box.env:PandaMoveBoxEnv',
    max_episode_steps=10000,
)
