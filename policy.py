import numpy as np


def expert_policy(state):
    # recover data from state
    ee_position = state[0:3]
    obj_location = state[3:6]
    obj_height = state[6]
    target_location = state[8:11]
    target_height = state[11]
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])
    put_location = target_location + np.asarray([0, 0, target_height / 2 + obj_height])
    grasp = state[15]

    # move robot arm to catch the object
    if not grasp:
        direction = catch_location - ee_position
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        ratio = 10
        if np.linalg.norm(direction) < 0.1:
            ratio += 1 / (np.linalg.norm(direction) + 0.1)
        action = direction / (np.linalg.norm(direction) * ratio)
    # move the object to the target
    else:
        direction = put_location - ee_position
        action = direction / (np.linalg.norm(direction) * 10)
        # when first catching the object, lift it up a bit to prevent it from leaning
        if ee_position[2] < 0.236:
            action = np.asarray([0, 0, 0.1])
    return action
