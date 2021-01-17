import numpy as np
import torch
import time
import torch.nn as nn
import os

from .buffer import Buffer
from tqdm import tqdm


# observation space is
# [ee_position*3, obj_location*3, obj_height, obj_width,
# target_location*3, target_height, target_width, dist_ee_obj, dist_obj_tar, grasp]
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
    return ee_position, obj_location, obj_height, obj_width, target_location, target_height, target_width, \
           dist_ee_obj, dist_obj_tar, grasp


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(env, policy, buffer_size, device, std, seed=0):
    env.seed(seed)
    np.random.seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_steps = []
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0
    episode_steps = 0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        action = policy(state)
        ee_position, _, _, _, _, _, _, _, dist_obj_tar, grasp = recover_state(state)
        if grasp and dist_obj_tar > 0.2:
            action += np.random.randn(*action.shape) * std
        action = action.clip(-1.0, 1.0)

        next_state, reward, done, _ = env.step(action)
        mask = True if t == env.max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward
        episode_steps += 1
        state = next_state  # modified

        if done or t == env.max_episode_steps:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0
            num_steps.append(episode_steps)
            episode_steps = 0

    mean_return = total_return / num_episodes
    print(f'Mean return of the expert is {mean_return}')
    print(f'Max episode steps is {np.max(num_steps)}')
    print(f'Min episode steps is {np.min(num_steps)}')
    return buffer, mean_return


def build_mlp(input_dim, output_dim, hidden_units=(64, 64),
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def evaluation(env, actor, episodes, seed=0):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    total_return = 0.0
    num_episodes = 0
    num_steps = []
    max_speed = 0

    state = env.reset()
    episode_return = 0.0
    episode_steps = 0

    while num_episodes < episodes:
        state = torch.tensor(state, dtype=torch.float)
        action = actor(state)
        action = action.cpu().detach().numpy()
        if np.linalg.norm(action) > max_speed:
            max_speed = np.linalg.norm(action)
        next_state, reward, done, _ = env.step(action)
        episode_return += reward
        episode_steps += 1
        state = next_state

        if done or episode_steps == env.max_episode_steps:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            episode_return = 0.0
            num_steps.append(episode_steps)
            episode_steps = 0

    mean_return = total_return / num_episodes
    print(f'Mean return of the policy is {mean_return}')
    print(f'Max episode steps is {np.max(num_steps)}')
    print(f'Min episode steps is {np.min(num_steps)}')
    print(f'Max speed is {max_speed}')
    return mean_return


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False
