import torch
import numpy as np

from panda_grasp.env import PandaMoveBoxEnv
from utils.buffer import Buffer


def collect_demo(env, n_episode=1, scale=0.02):
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

    states = np.empty([record_interval, env.observation_space.shape[0]])
    actions = np.empty([record_interval, env.action_space.shape[0]])
    # rewards = np.empty(record_interval)
    # dones = np.empty(record_interval)
    next_states = np.empty_like(states)

    while episode < n_episode:
        t = 0
        done = False
        while t < record_interval:
            state, action, reward, next_state, done, _ = env.teleop_step()

            states[t, :] = state
            actions[t, :] = action
            # rewards[t] = reward
            # dones[t] = done
            next_states[t, :] = next_state

            if done:
                break
            t += 1

        buffer.append(
            state=states[0],
            action=actions.mean(axis=0),
            reward=env.calculate_reward(next_states[t, :], actions.mean(axis=0)),
            done=done,
            next_state=next_states[t, :]
        )

        if done:
            break
        episode += 1


if __name__ == '__main__':
    scale = 0.02
    env = PandaMoveBoxEnv(engine='GUI', key_scale=scale)
    collect_demo(env, n_episode=1, scale=scale)
