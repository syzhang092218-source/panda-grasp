from setuptools import setup, find_packages

setup(
    name='panda_grasp',
    version='0.0.1',
    description='Panda-grasp: A robot environment to test your reinforcement learning agents.',
    url='https://https://github.com/syzhang092218-source/panda-grasp',
    author='Songyuan Zhang, Tong Xiao',
    author_email='syzhang092218@gmail.com',
    packages=find_packages(),
    install_requires=['numpy>=1.16.0', 'gym>=0.17.2', 'setuptools>=49.2.0', 'pygame>=2.0.1', 'pybullet>=2.8.4',
                      'tqdm>=4.47.0', 'torch>=1.6.0', ]  # And any other dependencies foo needs
)
