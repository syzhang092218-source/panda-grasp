import torch
import argparse
import os

from panda_grasp.utils.utils import collect_demo
from panda_grasp import ENV, POLICY, POLICY_PARAMS


def main(args):
    env = ENV[args.env_id](engine='DIRECT')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer, mean_return = collect_demo(
        env=env,
        policy=POLICY[args.env_id][args.policy],
        params=POLICY_PARAMS[args.env_id],
        buffer_size=args.buffer_size,
        device=device,
        std=args.std,
        continuous=args.continuous,
        seed=args.seed
    )
    buffer.save(os.path.join(
        'buffer',
        args.env_id,
        f'size{args.buffer_size}_reward{round(mean_return, 2)}.pth'
    ))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env-id', type=str, default='PandaAvoidObstacleRandom-v0')
    p.add_argument('--buffer-size', type=int, default=200000)
    p.add_argument('--policy', type=str, default='base')
    p.add_argument('--std', type=float, default=0.05)
    p.add_argument('--continuous', action='store_true', default=False,
                   help='if True, collect continuous suboptimal trajectories')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    main(args)
