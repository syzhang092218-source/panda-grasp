import numpy as np
import math
import random


def recover_state(state):
    ee_position = state[0:3]
    obj_location = state[3:6]
    obj_height = state[6]
    obj_width = state[7]
    target_location = state[8:11]
    target_height = state[11]
    target_width = state[12]
    dist_ee_obj = state[13]
    dist_obj_tar = state[14]
    grasp = state[15]
    return {
        'ee_position': ee_position,
        'obj_location': obj_location,
        'obj_height': obj_height,
        'obj_width': obj_width,
        'target_location':target_location,
        'target_height': target_height,
        'target_width': target_width,
        'dist_ee_obj': dist_ee_obj,
        'dist_obj_tar': dist_obj_tar,
        'grasp': grasp
    }


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def expert_policy(state, std):
    # recover data from state
    state_re = recover_state(state)

    # calculate ee destination
    catch_location = state_re['obj_location'] + np.asarray([0, 0, state_re['obj_height'] / 2])
    # put_location = state_re['target_location'] + np.asarray([0, 0, state_re['target_height'] / 2 + state_re['obj_height']])
    put_location = state_re['target_location'] + np.asarray([0, 0, state_re['obj_height']])

    # move robot arm to catch the object
    if not state_re['grasp']:
        direction = catch_location - state_re['ee_position']
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        ratio = 7
        if np.linalg.norm(direction) < 0.2:
            ratio += 4 / (np.linalg.norm(direction) + 0.1)
        action = direction / (np.linalg.norm(direction) * ratio)

    # move the object to the target
    else:
        direction = put_location - state_re['ee_position']
        ratio = 3
        if np.linalg.norm(direction) < 0.2:
            ratio += 5 / (np.linalg.norm(direction) + 0.1)
        action = direction / (np.linalg.norm(direction) * ratio)
        # when first catching the object, lift it up a bit to prevent it from leaning
        if state_re['ee_position'][2] < 0.236:
            action = np.asarray([0, 0, 0.2])

    # add standard deviation and restrictions
    if state_re['grasp'] and state_re['dist_obj_tar'] > 0.2:
        action = add_random_noise(action / 2, std)
    else:
        action /= 2

    return action


# remove the lifting part of expert policy
def drag_policy(state, std):
    # recover data from state
    state_re = recover_state(state)

    # calculate ee destination
    catch_location = state_re['obj_location'] + np.asarray([0, 0, state_re['obj_height'] / 2])
    put_location = state_re['target_location'] + np.asarray([0, 0, state_re['target_height'] / 2 + state_re['obj_height']])

    # move robot arm to catch the object
    if not state_re['grasp']:
        direction = catch_location - state_re['ee_position']
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        ratio = 7
        if np.linalg.norm(direction) < 0.2:
            ratio += 4 / (np.linalg.norm(direction) + 0.1)
        action = direction / (np.linalg.norm(direction) * ratio)

    # move the object to the target
    else:
        direction = put_location - state_re['ee_position']
        action = direction / (np.linalg.norm(direction) * 5)

    # add standard deviation and restrictions
    if state_re['grasp'] and state_re['dist_obj_tar'] > 0.2:
        action = add_random_noise(action / 2, std)
    else:
        action /= 2

    return action


# reduce maximum speed to 20% of expert policy
def slow_policy(state, std):
    # recover data from state
    state_re = recover_state(state)

    # calculate ee destination
    catch_location = state_re['obj_location'] + np.asarray([0, 0, state_re['obj_height'] / 2])
    put_location = state_re['target_location'] + np.asarray([0, 0, state_re['target_height'] / 2 + state_re['obj_height']])

    # move robot arm to catch the object
    if not state_re['grasp']:
        direction = catch_location - state_re['ee_position']
        # reduce end-effector's velocity when close to the object in case ee knocks it over
        ratio = 35
        action = direction / (np.linalg.norm(direction) * ratio)

    # move the object to the target
    else:
        direction = put_location - state_re['ee_position']
        action = direction / (np.linalg.norm(direction) * 35)
        # when first catching the object, lift it up a bit to prevent it from leaning
        if state_re['ee_position'][2] < 0.236:
            action = np.asarray([0, 0, 0.04])

    # add standard deviation and restrictions
    if state_re['grasp'] and state_re['dist_obj_tar'] > 0.2:
        action = add_random_noise(action / 2, std)
    else:
        action /= 2

    return action


# value shift on action[0] so that ee knocks the object over
def knock_over_policy(state, std):
    # recover data from state
    state_re = recover_state(state)

    # calculate ee destination
    catch_location = state_re['obj_location'] + np.asarray([0, 0, state_re['obj_height'] / 2])

    # move robot arm to catch the object
    direction = catch_location - state_re['ee_position']
    action = direction / (np.linalg.norm(direction) * 5)
    action[0] -= 0.1

    # add standard deviation and restrictions
    if state_re['grasp'] and state_re['dist_obj_tar'] > 0.2:
        action = add_random_noise(action / 2, std)
    else:
        action /= 2

    return action


# move a longer distance instead of straight to destinations
# alpha & beta indicates the extent of detour(notice detour route is fixed)
# empirically alpha_max = beta_max = 1
def detour1_policy(state, std, alpha=0.1, beta=0.1):
    # recover data from state
    state_re = recover_state(state)

    # calculate ee destination
    catch_location = state_re['obj_location'] + np.asarray([0, 0, state_re['obj_height'] / 2])
    put_location = state_re['target_location'] + np.asarray([0, 0, state_re['target_height'] / 2 + state_re['obj_height']])

    # move robot arm to catch the object
    if not state_re['grasp']:
        direction = catch_location - state_re['ee_position']
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
        direction = put_location - state_re['ee_position']
        # add detour
        if math.trunc(direction[0] * 2 / 0.3) == 0:
            direction[1] += direction[0] * 2 / 0.3 * beta / 2
            direction[2] += direction[0] * 2 / 0.3 * beta / 2
        else:
            direction[1] -= (direction[0] * 2 / 0.3 - 1) * beta / 10
            direction[2] -= (direction[0] * 2 / 0.3 - 1) * beta / 10

        action = direction / (np.linalg.norm(direction) * 5)

        # when first catching the object, lift it up a bit to prevent it from leaning
        if state_re['ee_position'][2] < 0.236:
            action = np.asarray([0, 0, 0.1])

    # add standard deviation and restrictions
    if state_re['grasp'] and state_re['dist_obj_tar'] > 0.2:
        action = add_random_noise(action / 2, std)
    else:
        action /= 2

    return action


# move a longer distance instead of straight to destinations
# alpha & beta indicates the extent of detour(notice detour route is fixed)
# empirically alpha_max = beta_max = 1
def detour2_policy(state, std, alpha=0.1, beta=0.1):
    # recover data from state
    state_re = recover_state(state)

    # calculate ee destination
    catch_location = state_re['obj_location'] + np.asarray([0, 0, state_re['obj_height'] / 2])
    put_location = state_re['target_location'] + np.asarray([0, 0, state_re['target_height'] / 2 + state_re['obj_height']])

    # move robot arm to catch the object
    if not state_re['grasp']:
        direction = catch_location - state_re['ee_position']
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
        direction = put_location - state_re['ee_position']
        # add detour
        if math.trunc(direction[0] * 2 / 0.3) == 0:
            direction[1] -= direction[0] * 2 / 0.3 * beta / 4
            direction[2] -= direction[0] * 2 / 0.3 * beta / 4
        else:
            direction[1] += (direction[0] * 2 / 0.3 - 1) * beta / 10
            direction[2] += (direction[0] * 2 / 0.3 - 1) * beta / 10

        action = direction / (np.linalg.norm(direction) * 5)

        # when first catching the object, lift it up a bit to prevent it from leaning
        if state_re['ee_position'][2] < 0.236:
            action = np.asarray([0, 0, 0.1])

    # add standard deviation and restrictions
    if state_re['grasp'] and state_re['dist_obj_tar'] > 0.2:
        action = add_random_noise(action / 2, std)
    else:
        action /= 2

    return action


def near_optimal_policy(state, std):
    choice = random.choice(['expert', 'detour1', 'detour2'])
    if choice == 'expert':
        return expert_policy(state, std)
    elif choice == 'detour1':
        return detour1_policy(state, std)
    elif choice == 'detour2':
        return detour2_policy(state, std)


PANDA_MOVE_BOX_POLICY = {
    'expert': expert_policy,
    'drag': drag_policy,
    'knock_over': knock_over_policy,
    'slow': slow_policy,
    'detour1': detour1_policy,
    'detour2': detour2_policy,
    'near_optimal': near_optimal_policy,
    'recover': recover_state
}
