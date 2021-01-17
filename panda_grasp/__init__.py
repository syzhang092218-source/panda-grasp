from gym.envs.registration import register
from .env import PandaMoveBoxEnv, PandaAvoidObstacleEnv
from .policy import PANDA_MOVE_BOX_POLICY, PANDA_AVOID_OBSTACLE_POLICY


register(
    id='PandaMoveBox-v0',
    entry_point='panda_grasp.env:PandaMoveBoxEnv',
    max_episode_steps=5000,
)

register(
    id='PandaAvoidObstacle-v0',
    entry_point='panda_grasp.env:PandaAvoidObstacleEnv',
    max_episode_steps=2000,
)

ENV = {
    'PandaMoveBox-v0': PandaMoveBoxEnv,
    'PandaAvoidObstacle-v0': PandaAvoidObstacleEnv,
}

POLICY = {
    'PandaMoveBox-v0': PANDA_MOVE_BOX_POLICY,
    'PandaAvoidObstacle-v0': PANDA_AVOID_OBSTACLE_POLICY,
}
