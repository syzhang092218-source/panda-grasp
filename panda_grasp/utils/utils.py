import numpy as np

from .buffer import Buffer
from tqdm import tqdm


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
        action = add_random_noise(action, std)

        next_state, reward, done, _ = env.step(action)
        mask = True if t == env.max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward
        episode_steps += 1

        if done or t == env.max_episode_steps:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0
            num_steps.append(episode_steps)
            episode_steps = 0

        state = next_state

    mean_return = total_return / num_episodes
    print(f'Mean return of the expert is {mean_return}')
    print(f'Max episode steps is {np.max(num_steps)}')
    print(f'Min episode steps is {np.min(num_steps)}')
    return buffer, mean_return
