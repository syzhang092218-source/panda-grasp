from gym.envs.registration import register
from .env import PandaMoveBoxEnv, PandaAvoidObstacleEnv, PandaAvoidObstacleRandomEnv
from .policy import PANDA_MOVE_BOX_POLICY, PANDA_AVOID_OBSTACLE_POLICY, PANDA_AVOID_OBSTACLE_RANDOM_POLICY
from .policy import panda_avoid_obstacle_policy, panda_avoid_obstacle_random_policy


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

register(
    id='PandaAvoidObstacleRandom-v0',
    entry_point='panda_grasp.env:PandaAvoidObstacleRandomEnv',
    max_episode_steps=2000,
)

ENV = {
    'PandaMoveBox-v0': PandaMoveBoxEnv,
    'PandaAvoidObstacle-v0': PandaAvoidObstacleEnv,
    'PandaAvoidObstacleRandom-v0': PandaAvoidObstacleRandomEnv,
}

POLICY = {
    'PandaMoveBox-v0': PANDA_MOVE_BOX_POLICY,
    'PandaAvoidObstacle-v0': PANDA_AVOID_OBSTACLE_POLICY,
    'PandaAvoidObstacleRandom-v0': PANDA_AVOID_OBSTACLE_RANDOM_POLICY,
}

POLICY_PARAMS = {
    'PandaAvoidObstacle-v0': panda_avoid_obstacle_policy.PolicyParams,
    'PandaAvoidObstacleRandom-v0': panda_avoid_obstacle_random_policy.PolicyParams,
}
