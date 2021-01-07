import os
import numpy as np
import pybullet as p
import pybullet_data
import random
from gym.utils import seeding
import copy


class Panda:

    def __init__(self, basePosition=None):
        if basePosition is None:
            basePosition = [0, 0, 0]
        self.init_pos = [
            0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi*3/4, 0.0, 0.0,  # joint position
            0.05, 0.05  # gripper position
        ]
        self.urdfRootPath = pybullet_data.getDataPath()
        self.panda = p.loadURDF(os.path.join(self.urdfRootPath, "franka_panda/panda.urdf"),
                                useFixedBase=True,
                                basePosition=basePosition)

    """functions that environment should use"""
    # has two modes: joint space control (0) and ee-space control (1)
    # djoint is a 7-dimensional vector of joint velocities
    # dposition is a 3-dimensional vector of end-effector linear velocities
    # dquaternion is a 4-dimensional vector of end-effector quaternion velocities
    def step(self, mode=1, djoint=None, dposition=None, dquaternion=None, grasp_open=True):
        # set default values
        if dquaternion is None:
            dquaternion = [0] * 4
        if dposition is None:
            dposition = [0] * 3
        if djoint is None:
            djoint = [0] * 7

        # velocity control
        self._velocity_control(
            mode=mode,
            djoint=djoint,
            dposition=dposition,
            dquaternion=dquaternion,
            grasp_open=grasp_open
        )

        # update robot state measurement
        self._read_state()
        self._read_jacobian()

    def reset(self):
        init_pos = copy.deepcopy(self.init_pos)
        random_number = self.np_random.uniform(low=-.1, high=.1, size=len(init_pos)) * 0.5
        for i in [1, 3, 5]:
            init_pos[i] += random_number[i]
        #init_pos = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.05, 0.05]
        self._reset_robot(init_pos)

    def reset_with_obs(self, obs_state):
        init_pos = copy.deepcopy(self.init_pos)
        for i in [1, 3, 5]:
            init_pos[i] = obs_state[i]
        #init_pos = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.05, 0.05]
        self._reset_robot(init_pos)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    """internal functions"""

    def _read_state(self):
        joint_position = [0]*9
        joint_velocity = [0]*9
        joint_torque = [0]*9
        joint_states = p.getJointStates(self.panda, range(9))
        for idx in range(9):
            joint_position[idx] = joint_states[idx][0]
            joint_velocity[idx] = joint_states[idx][1]
            joint_torque[idx] = joint_states[idx][3]
        ee_states = p.getLinkState(self.panda, 11)
        ee_position = list(ee_states[4])
        ee_quaternion = list(ee_states[5])
        gripper_contact = p.getContactPoints(bodyA=self.panda, linkIndexA=10)
        self.state['joint_position'] = np.asarray(joint_position)
        self.state['joint_velocity'] = np.asarray(joint_velocity)
        self.state['joint_torque'] = np.asarray(joint_torque)
        self.state['ee_position'] = np.asarray(ee_position)
        self.state['ee_quaternion'] = np.asarray(ee_quaternion)
        self.state['ee_euler'] = np.asarray(p.getEulerFromQuaternion(ee_quaternion))
        self.state['gripper_contact'] = len(gripper_contact) > 0

    def _read_jacobian(self):
        linear_jacobian, angular_jacobian = p.calculateJacobian(self.panda, 11, [0, 0, 0], list(self.state['joint_position']), [0]*9, [0]*9)
        linear_jacobian = np.asarray(linear_jacobian)[:,:7]
        angular_jacobian = np.asarray(angular_jacobian)[:,:7]
        full_jacobian = np.zeros((6,7))
        full_jacobian[0:3,:] = linear_jacobian
        full_jacobian[3:6,:] = angular_jacobian
        self.jacobian['full_jacobian'] = full_jacobian
        self.jacobian['linear_jacobian'] = linear_jacobian
        self.jacobian['angular_jacobian'] = angular_jacobian

    def _reset_robot(self, joint_position):
        self.state = {}
        self.jacobian = {}
        self.desired = {}
        for idx in range(len(joint_position)):
            p.resetJointState(self.panda, idx, joint_position[idx])
        self._read_state()
        self._read_jacobian()
        self.desired['joint_position'] = self.state['joint_position']
        self.desired['ee_position'] = self.state['ee_position']
        self.desired['ee_quaternion'] = self.state['ee_quaternion']

    def _inverse_kinematics(self, ee_position, ee_quaternion):
        return p.calculateInverseKinematics(self.panda, 11, list(ee_position), list(ee_quaternion))

    def _velocity_control(self, mode, djoint, dposition, dquaternion, grasp_open):
        if mode:
            self.desired['ee_position'] += np.asarray(dposition) / 240.0
            self.desired['ee_quaternion'] += np.asarray(dquaternion) / 240.0
            q_dot = self._inverse_kinematics(self.desired['ee_position'], self.desired['ee_quaternion']) - self.state['joint_position']
        else:
            self.desired['joint_position'] += np.asarray(list(djoint)+[0, 0]) / 240.0
            q_dot = self.desired['joint_position'] - self.state['joint_position']
        gripper_position = [0.03, 0.03]  # to work with ycb_object: zsy_long_box
        if grasp_open:
            gripper_position = [0.05, 0.05]
        p.setJointMotorControlArray(self.panda, range(9), p.VELOCITY_CONTROL, targetVelocities=list(q_dot))
        p.setJointMotorControlArray(self.panda, [9, 10], p.POSITION_CONTROL, targetPositions=gripper_position)


class DisabledPanda(Panda):
    def __init__(self, basePosition=[0,0,0]):
        self.urdfRootPath = pybullet_data.getDataPath()
        current_path = os.path.dirname(os.path.abspath(__file__))
        self.panda = p.loadURDF(os.path.join(current_path, "disabled_panda/disabled_panda.urdf"),useFixedBase=True,basePosition=basePosition)
        self.init_pos = [0.0, 0.0, 0.0, -2*np.pi/4, 0.0, np.pi/2, np.pi/4, 0.0, 0.0, 0.05, 0.05]
