import torch
import argparse
import os

from panda_grasp.env import PandaMoveBoxEnv
from panda_grasp.utils.utils import collect_demo
from panda_grasp.policy import PANDA_MOVE_BOX_POLICY


def main(args):
    env = PandaMoveBoxEnv(engine='DIRECT')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer, mean_return = collect_demo(
        env=env,
        policy=PANDA_MOVE_BOX_POLICY[args.policy],
        buffer_size=args.buffer_size,
        device=device,
        std=args.std,
        seed=args.seed
    )
    buffer.save(os.path.join(
        'buffer',
        f'size{args.buffer_size}_reward{round(mean_return, 2)}.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer-size', type=int, default=100000)
    p.add_argument('--policy', type=str, default='near_optimal')
    p.add_argument('--std', type=float, default=0.1, help='maximum std = 0.2')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    main(args)
