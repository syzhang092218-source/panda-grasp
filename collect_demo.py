import torch
import numpy as np
import os

from panda_grasp.env import PandaMoveBoxEnv
from utils.buffer import Buffer


def collect_demo(env, n_episode, scale=0.02):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer = Buffer(
        buffer_size=n_episode * env.max_episode_steps,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    state = env.reset()
    episode = 0
    record_interval = int(1 / scale)
    n_demo = 0

    states = np.empty([record_interval, env.observation_space.shape[0]])
    actions = np.empty([record_interval, env.action_space.shape[0]])

    # dones = np.empty(record_interval)
    next_states = np.empty_like(states)

    rewards = []
    while episode < n_episode:
        done = False
        step = 0
        epi_reward = 0
        while step < env.max_episode_steps:
            t = 0
            while t < record_interval:
                state, action, reward, next_state, done, _ = env.teleop_step()
                states[t, :] = state
                actions[t, :] = action
                # rewards[t] = reward
                # dones[t] = done
                next_states[t, :] = next_state
                t += 1
                if done:
                    break
            t -= 1
            reward = env.calculate_reward(next_states[t, :], actions.mean(axis=0))
            buffer.append(
                state=states[0],
                action=actions.mean(axis=0),
                reward=reward,
                done=done,
                next_state=next_states[t, :]
            )
            n_demo += 1
            epi_reward += reward
            if done:
                break
            step += 1

        rewards.append(epi_reward)
        episode += 1

    buffer.clean()
    path = os.path.join(
        'buffers',
        f'size{buffer.buffer_size}_reward{np.mean(rewards)}.pth'
    )
    print(f'Mean return of the expert is {np.mean(rewards)}')
    buffer.save(path)


if __name__ == '__main__':
    scale = 0.02
    env = PandaMoveBoxEnv(engine='GUI', key_scale=scale)
    collect_demo(env, n_episode=2, scale=scale)
