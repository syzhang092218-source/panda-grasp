# Panda Grasp

[PyBullet](https://pybullet.org/wordpress/) Environment for Panda Robot Arm, containing PandaMoveBox env, PandaAvoidObstacle env PandaAvoidLyingObstacle env registered in [gym](https://gym.openai.com/).

Developed by [Songyuan Zhang](https://syzhang092218-source.github.io/) and [Tong Xiao](https://tongxiao2000.github.io/).

## Install

Use these commands to quickly install this package

```bash
# Clone the Repository
git clone https://github.com/syzhang092218-source/panda-grasp.git
cd panda-grasp

# install dependencies
pip install -r requirements.txt

# install the package
pip install -e .
```

## Policies

We prepared several policies for each environment. Please refer to ```panda_grasp.policy``` for policies. 

## Collect Demonstrations

One can collect demonstrations for Imitation Learning with different policies using the following commands:

For optimal policy and discontinuous suboptimal policies:

```bash
python collect_demo.py --env-id PandaAvoidObstacle-v0 --buffer-size 40000 --policy expert --std 0.05 --seed 0
```

For continuous suboptimal policies:

```bash
python collect_demo.py --env-id PandaAvoidObstacle-v0 --buffer-size 40000 --policy base --continuous --std 0.05 --seed 0
```

## Acknowledgement 

This package is based on the [Panda environment](https://github.com/dylan-losey/panda-env) developed by the [ILIAD Lab](http://iliad.stanford.edu/).