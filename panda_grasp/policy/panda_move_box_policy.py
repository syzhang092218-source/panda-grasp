import numpy as np
import math
from panda_grasp.utils.utils import recover_state
import random


def expert_policy(state):
    # recover data from state
    ee_position, obj_location, obj_height, _, target_location, target_height, _, _, _, grasp = recover_state(state)

    # calculate ee destination
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])
    # put_location = target_location + np.asarray([0, 0, target_height / 2 + obj_height])
    put_location = target_location + np.asarray([0, 0, obj_height])

    # move robot arm to catch the object
    if not grasp:
        direction = catch_location - ee_position
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        ratio = 7
        if np.linalg.norm(direction) < 0.2:
            ratio += 4 / (np.linalg.norm(direction) + 0.1)
        action = direction / (np.linalg.norm(direction) * ratio)

    # move the object to the target
    else:
        direction = put_location - ee_position
        ratio = 3
        if np.linalg.norm(direction) < 0.2:
            ratio += 5 / (np.linalg.norm(direction) + 0.1)
        action = direction / (np.linalg.norm(direction) * ratio)
        # when first catching the object, lift it up a bit to prevent it from leaning
        if ee_position[2] < 0.236:
            action = np.asarray([0, 0, 0.2])
    return action / 2


# remove the lifting part of expert policy
def drag_policy(state):
    # recover data from state
    ee_position, obj_location, obj_height, _, target_location, target_height, _, _, _, grasp = recover_state(state)

    # calculate ee destination
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])
    put_location = target_location + np.asarray([0, 0, target_height / 2 + obj_height])

    # move robot arm to catch the object
    if not grasp:
        direction = catch_location - ee_position
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        ratio = 7
        if np.linalg.norm(direction) < 0.2:
            ratio += 4 / (np.linalg.norm(direction) + 0.1)
        action = direction / (np.linalg.norm(direction) * ratio)

    # move the object to the target
    else:
        direction = put_location - ee_position
        action = direction / (np.linalg.norm(direction) * 5)

    return action / 2


# reduce maximum speed to 20% of expert policy
def slow_policy(state):
    # recover data from state
    ee_position, obj_location, obj_height, _, target_location, target_height, _, _, _, grasp = recover_state(state)

    # calculate ee destination
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])
    put_location = target_location + np.asarray([0, 0, target_height / 2 + obj_height])

    # move robot arm to catch the object
    if not grasp:
        direction = catch_location - ee_position
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        ratio = 35
        action = direction / (np.linalg.norm(direction) * ratio)

    # move the object to the target
    else:
        direction = put_location - ee_position
        action = direction / (np.linalg.norm(direction) * 35)
        # when first catching the object, lift it up a bit to prevent it from leaning
        if ee_position[2] < 0.236:
            action = np.asarray([0, 0, 0.04])
    return action / 2


# value shift on action[0] so that ee knocks the object over
def knock_over_policy(state):
    # recover data from state
    ee_position, obj_location, obj_height, _, target_location, target_height, _, _, _, grasp = recover_state(state)

    # calculate ee destination
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])

    # move robot arm to catch the object
    direction = catch_location - ee_position
    action = direction / (np.linalg.norm(direction) * 5)
    action[0] -= 0.1
    return action / 2


# move a longer distance instead of straight to destinations
# alpha & beta indicates the extent of detour(notice detour route is fixed)
# empirically alpha_max = beta_max = 1
def detour1_policy(state, alpha=0.1, beta=0.1):
    # recover data from state
    ee_position, obj_location, obj_height, _, target_location, target_height, _, _, _, grasp = recover_state(state)

    # calculate ee destination
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])
    put_location = target_location + np.asarray([0, 0, target_height / 2 + obj_height])

    # move robot arm to catch the object
    if not grasp:
        direction = catch_location - ee_position
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        ratio = 7
        if np.linalg.norm(direction) < 0.1:
            ratio += 4 / (np.linalg.norm(direction) + 0.1)
        # add detour
        direction[1] += direction[0] * alpha / 0.06 / 3
        direction[2] += direction[0] * alpha / 0.06 / 3

        action = direction / (np.linalg.norm(direction) * ratio)

    # move the object to the target
    else:
        direction = put_location - ee_position
        # add detour
        if math.trunc(direction[0] * 2 / 0.3) == 0:
            direction[1] += direction[0] * 2 / 0.3 * beta / 2
            direction[2] += direction[0] * 2 / 0.3 * beta / 2
        else:
            direction[1] -= (direction[0] * 2 / 0.3 - 1) * beta / 10
            direction[2] -= (direction[0] * 2 / 0.3 - 1) * beta / 10

        action = direction / (np.linalg.norm(direction) * 5)

        # when first catching the object, lift it up a bit to prevent it from leaning
        if ee_position[2] < 0.236:
            action = np.asarray([0, 0, 0.1])
    return action / 2


# move a longer distance instead of straight to destinations
# alpha & beta indicates the extent of detour(notice detour route is fixed)
# empirically alpha_max = beta_max = 1
def detour2_policy(state, alpha=0.1, beta=0.1):
    # recover data from state
    ee_position, obj_location, obj_height, _, target_location, target_height, _, _, _, grasp = recover_state(state)

    # calculate ee destination
    catch_location = obj_location + np.asarray([0, 0, obj_height / 2])
    put_location = target_location + np.asarray([0, 0, target_height / 2 + obj_height])

    # move robot arm to catch the object
    if not grasp:
        direction = catch_location - ee_position
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        ratio = 7
        if np.linalg.norm(direction) < 0.1:
            ratio += 4 / (np.linalg.norm(direction) + 0.1)
        # add detour
        direction[1] -= direction[0] * alpha / 0.06 / 3
        direction[2] -= direction[0] * alpha / 0.06 / 3

        action = direction / (np.linalg.norm(direction) * ratio)

    # move the object to the target
    else:
        direction = put_location - ee_position
        # add detour
        if math.trunc(direction[0] * 2 / 0.3) == 0:
            direction[1] -= direction[0] * 2 / 0.3 * beta / 4
            direction[2] -= direction[0] * 2 / 0.3 * beta / 4
        else:
            direction[1] += (direction[0] * 2 / 0.3 - 1) * beta / 10
            direction[2] += (direction[0] * 2 / 0.3 - 1) * beta / 10

        action = direction / (np.linalg.norm(direction) * 5)

        # when first catching the object, lift it up a bit to prevent it from leaning
        if ee_position[2] < 0.236:
            action = np.asarray([0, 0, 0.1])
    return action / 2


def near_optimal_policy(state):
    choice = random.choice(['expert', 'detour1', 'detour2'])
    if choice == 'expert':
        return expert_policy(state)
    elif choice == 'detour1':
        return detour1_policy(state)
    elif choice == 'detour2':
        return detour2_policy(state)
