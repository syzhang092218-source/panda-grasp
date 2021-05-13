import numpy as np


def recover_state(state):
    ee_position = state[0:3]
    obj_location = state[3:6]
    target_location = state[6:9]
    return {
        'ee_position': ee_position,
        'obj_location': obj_location,
        'target_location': target_location,
    }


def source2target(location, env):
    obj_location = env.obj_location
    tar_location = env.target_location
    array = tar_location - obj_location
    theta = np.pi - np.arctan(array[1] / array[0])
    convert_matrix = np.asarray([[np.cos(theta), np.sin(theta), 0],
                                 [-np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    target = location - np.array([obj_location[0], obj_location[1], 0])
    target = np.matmul(convert_matrix.transpose(), target)
    return target


def target2source(location, env):
    obj_location = env.obj_location
    tar_location = env.target_location
    array = tar_location - obj_location
    theta = np.pi - np.arctan(array[1] / array[0])
    convert_matrix = np.asarray([[np.cos(theta), np.sin(theta), 0],
                                 [-np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    target = np.matmul(convert_matrix, location)
    target += np.array([obj_location[0], obj_location[1], 0])
    return target


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def base_policy(state, env, std=0, init_vy=0, init_vz=1.2):
    info = recover_state(state)

    # calculate speed on x/y/z axis in coordinate system of the parabola
    dist = np.linalg.norm(info['ee_position'] - info['target_location'])
    t = source2target(info['ee_position'], env)[0]
    t += dist / 25
    if t > dist:
        action = info['target_location'] + np.array([0, 0, env.obj_height]) - info['ee_position']
    else:
        goal_point = np.asarray(
            [t,
             init_vy * t - init_vy * t ** 2 / dist,
             init_vz * t - init_vz * t ** 2 / dist + env.obj_height]
        )
        action = target2source(goal_point, env) - info['ee_position']

    action = 0.1 * action / np.linalg.norm(action)

    # add standard deviation and restrictions
    action = add_random_noise(action, std)

    return action


PolicyParams = {
    'default_init_vy': 0,
    'default_init_vz': 1.2,
    'max_init_vy': 1.5,
    'max_init_vz': 1.7,
}


PANDA_AVOID_OBSTACLE_RANDOM_POLICY = {
    'base': base_policy,
}
