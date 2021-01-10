import numpy as np


def expert_policy(state):
    # observation space is
    # [ee_position*3, obj_location*3, obj_height, obj_width,
    # target_location*3, target_height, target_width, dist_ee_obj, dist_obj_tar]
    ee_position = state[0:3]
    obj_location = state[3:6]
    target_location = state[8:11]
    catch_location = obj_location + np.asarray([0, 0, state[6]/2])
    put_location = target_location + np.asarray([0, 0, state[11]/2 + state[6]])
    direction = catch_location - ee_position
    action = direction / (abs(np.max(direction)) * 100)
    if np.linalg.norm(direction) < 0.015:
        direction = put_location - ee_position
        action = direction / (abs(np.max(direction)) * 100)
        # debug
        print("put_location:", put_location)
        print("ee_position:", ee_position)
        print("action2:", action)
        # end
    else:
        print("direction:", direction)
        print("dist_ee_obj:", state[13])
        print("action1", action)
    return action

