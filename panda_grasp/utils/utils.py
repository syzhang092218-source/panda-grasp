import numpy as np
import torch
import torch.nn as nn

from .buffer import Buffer
from tqdm import tqdm


def collect_demo(env, policy, params, buffer_size, device, std, continuous, seed=0):
    env.seed(seed)
    np.random.seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    returns = []
    num_steps = []
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0
    episode_steps = 0

    # set initial speed
    init_vy = params['default_init_vy']
    init_vz = params['default_init_vz']

    for i_step in tqdm(range(1, buffer_size + 1)):
        t += 1

        if continuous:
            action = policy(state, env, std, init_vy, init_vz)
        else:
            action = policy(state, env, std)
        next_state, reward, done, _ = env.step(action)
        mask = True if t == env.max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward
        episode_steps += 1
        state = next_state

        if done or t == env.max_episode_steps:
            # print(f'init_vy: {init_vy}, init_vz: {init_vz}')
            tqdm.write(f'Reward: {episode_return}')
            if continuous:
                init_vy = params['default_init_vy'] + \
                          (params['max_init_vy'] - params['default_init_vy']) * (i_step / (buffer_size + 1))
                init_vz = params['default_init_vz'] + \
                          (params['max_init_vz'] - params['default_init_vz']) * (i_step / (buffer_size + 1))
                # if init_vy < params['max_init_vy']:
                #     init_vy += 0.03
                # elif init_vz < params['max_init_vz']:
                #     init_vy += 0.03
                #     init_vz += 0.05
                # else:
                #     init_vy = params['default_init_vy']
                #     init_vz = params['default_init_vz']
            num_episodes += 1
            returns.append(episode_return)
            # total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0
            num_steps.append(episode_steps)
            episode_steps = 0

    # mean_return = total_return / num_episodes
    print(f'Mean return of the expert is {np.mean(returns)}')
    print(f'Max episode steps is {np.max(num_steps)}')
    print(f'Min episode steps is {np.min(num_steps)}')

    return buffer, np.mean(returns)


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
