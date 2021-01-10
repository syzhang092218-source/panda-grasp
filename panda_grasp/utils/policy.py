import numpy as np
import math


def expert_policy(state):
    # recover data from state
    ee_position = state[0:3]
    obj_location = state[3:6]
    obj_height = state[6]
    target_location = state[8:11]
    target_height = state[11]
    grasp = state[15]

    # calculate ee destination
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])
    put_location = target_location + np.asarray([0, 0, target_height / 2 + obj_height])

    # move robot arm to catch the object
    if not grasp:
        print("ee_loc: ", ee_position)
        print("catch_loc: ", catch_location)
        print("target_loc: ", target_location)
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
    return action * 10


# remove the lifting part of expert policy
def drag_policy(state):
    # recover data from state
    ee_position = state[0:3]
    obj_location = state[3:6]
    obj_height = state[6]
    target_location = state[8:11]
    target_height = state[11]
    grasp = state[15]

    # calculate ee destination
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])
    put_location = target_location + np.asarray([0, 0, target_height / 2 + obj_height])

    # move robot arm to catch the object
    if not grasp:
        direction = catch_location - ee_position
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        action = direction / (np.linalg.norm(direction) * 10)

    # move the object to the target
    else:
        direction = put_location - ee_position
        action = direction / (np.linalg.norm(direction) * 10)

    return action * 10


# reduce maximum speed to 20% of expert policy
def slow_policy(state):
    # recover data from state
    ee_position = state[0:3]
    obj_location = state[3:6]
    obj_height = state[6]
    target_location = state[8:11]
    target_height = state[11]
    grasp = state[15]

    # calculate ee destination
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])
    put_location = target_location + np.asarray([0, 0, target_height / 2 + obj_height])

    # move robot arm to catch the object
    if not grasp:
        direction = catch_location - ee_position
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        ratio = 50
        action = direction / (np.linalg.norm(direction) * ratio)

    # move the object to the target
    else:
        direction = put_location - ee_position
        action = direction / (np.linalg.norm(direction) * 50)
        # when first catching the object, lift it up a bit to prevent it from leaning
        if ee_position[2] < 0.236:
            action = np.asarray([0, 0, 0.02])
    return action * 10


# value shift on action[0] so that ee knocks the object over
def knock_over_policy(state):
    # recover data from state
    ee_position = state[0:3]
    obj_location = state[3:6]
    obj_height = state[6]

    # calculate ee destination
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])

    # move robot arm to catch the object
    direction = catch_location - ee_position
    action = direction / (np.linalg.norm(direction) * 10)
    action[0] -= 0.05
    return action * 10


# move a longer distance instead of straight to destinations
def detour_policy(state):
    # recover data from state
    ee_position = state[0:3]
    obj_location = state[3:6]
    obj_height = state[6]
    target_location = state[8:11]
    target_height = state[11]
    grasp = state[15]

    # calculate ee destination
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])
    put_location = target_location + np.asarray([0, 0, target_height / 2 + obj_height])

    # move robot arm to catch the object
    if not grasp:
        direction = catch_location - ee_position
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        ratio = 10
        if np.linalg.norm(direction) < 0.1:
            ratio += 1 / (np.linalg.norm(direction) + 0.1)
        # add detour
        direction[1] += direction[0] * 0.3 / 0.06
        direction[2] += direction[0] * 0.3 / 0.06

        action = direction / (np.linalg.norm(direction) * ratio)

    # move the object to the target
    else:
        direction = put_location - ee_position
        # add detour
        if math.trunc(direction[0] * 2 / 0.3) == 0:
            direction[1] += direction[0] * 2 / 0.3 * 1.0
            direction[2] += direction[0] * 2 / 0.3 * 0.5
        else:
            direction[1] -= (direction[0] * 2 / 0.3 - 1) * 0.2
            direction[2] -= (direction[0] * 2 / 0.3 - 1) * 0.2

        action = direction / (np.linalg.norm(direction) * 10)

        # when first catching the object, lift it up a bit to prevent it from leaning
        if ee_position[2] < 0.236:
            action = np.asarray([0, 0, 0.1])
    return action * 10
