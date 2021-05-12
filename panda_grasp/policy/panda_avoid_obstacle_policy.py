import numpy as np


def recover_state(state):
    ee_position = state[0:3]
    joint_position = state[3:10]
    joint_velocity = state[10:17]
    joint_torque = state[17:24]
    obj_location = state[24:27]
    obj_height = state[27]
    obj_width = state[28]
    obstacle_location = state[29:32]
    obstacle_height = state[32]
    obstacle_width = state[33]
    target_location = state[34:37]
    target_height = state[37]
    target_width = state[38]
    return {
        'ee_position': ee_position,
        'joint_position': joint_position,
        'joint_velocity': joint_velocity,
        'joint_torque': joint_torque,
        'obj_location': obj_location,
        'obj_height': obj_height,
        'obj_width': obj_width,
        'obstacle_location': obstacle_location,
        'obstacle_height': obstacle_height,
        'obstacle_width': obstacle_width,
        'target_location': target_location,
        'target_height': target_height,
        'target_width': target_width
    }


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def source2target(location):
    convert_matrix = np.asarray([[-0.8, 0.6, 0], [-0.6, -0.8, 0], [0, 0, 1]])
    target = location + np.asarray([-0.7, 0, 0])
    target = np.matmul(convert_matrix.transpose(), target)
    return target


def target2source(location):
    convert_matrix = np.asarray([[-0.8, 0.6, 0], [-0.6, -0.8, 0], [0, 0, 1]])
    target = np.matmul(convert_matrix, location)
    target += np.asarray([0.7, 0, 0])
    return target


def base_policy(state, env, std=0.1, init_vy=1.5, init_vz=0.2, distance=0.5):
    """
    base policy is a parabola from start point to target
    :param state: current state
    :param std: standard deviation of the action
    :param init_vy: initial velocity in y axis's direction. note that the x axis is from start point to target,
                    and y axis is on the plane of the table, perpendicular to x axis
    :param init_vz: initial velocity in z axis's direction
    :param distance: distance between start point to target
    :return: action is velocity on x/y/z axis(Vx/y/z:[-1, 1])
    """
    # recover data from state
    state_re = recover_state(state)

    # calculate speed on x/y/z axis in coordinate system of the parabola
    t = source2target(state_re['ee_position'])[0]
    t += 0.02
    if t > 0.5:
        t = 0.5

    goal_point = np.asarray(
        [t,
         init_vy * t - init_vy * t ** 2 / distance,
         init_vz * t - init_vz * t ** 2 / distance + state_re['obj_height']]
    )

    action = target2source(goal_point) - state_re['ee_position']
    action = 0.1 * action / np.linalg.norm(action)

    # add standard deviation and restrictions
    action = add_random_noise(action, std)

    return action


def detour_policy(state, env, std=0.1, deviation_vy=0.6, deviation_vz=1.3):
    """
    change initial Vy and Vz to detour or to hit the obstacle
    """
    init_vy = 1.5 + deviation_vy
    init_vz = 0.2 + deviation_vz
    action = base_policy(state, std, init_vy, init_vz)

    return action


def expert_policy(state, env, std=0.1):
    """
    mount the obstacle(initial Vy = 0), the best policy in this environment
    """
    # calculate speed on x/y/z axis in coordinate system of the parabola
    init_vz = 0.9
    init_vy = 0
    action = base_policy(state, std, init_vy, init_vz)

    return action


def stray_policy(state, env, std, distance=0.7):
    """
    change distance to put the object in a wrong place
    """
    action = base_policy(state, std, distance=distance)

    return action


PolicyParams = {
    'default_init_vy': 0,
    'default_init_vz': 0.9,
    'max_init_vy': 1.5,
    'max_init_vz': 1.5,
}


PANDA_AVOID_OBSTACLE_POLICY = {
    'base': base_policy,
    'detour': detour_policy,
    'expert': expert_policy,
    'stray': stray_policy,
}
