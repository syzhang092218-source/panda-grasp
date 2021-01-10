import numpy as np

def expert_policy(state):
    # observation space is
    # [ee_position*3, obj_location*3, obj_height, obj_width,
    # target_location*3, target_height, target_width, dist_ee_obj, dist_obj_tar]
    ee_position = state[0:3]
    obj_location = state[3:6]
    direction = obj_location - ee_position
    action = direction / np.max(direction) * 10
    return action

