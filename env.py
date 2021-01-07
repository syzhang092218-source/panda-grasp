import numpy as np
import pybullet as p
import pybullet_data
from panda import Panda
from objects import YCBObject
from gym import error, spaces
import gym
import os
from key import Key
import time


class PandaRawEnv(gym.Env):

    def __init__(self):
        # create simulation (GUI)
        self.urdfRootPath = pybullet_data.getDataPath()
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)

        # set up camera
        self._set_camera()

        # load some scene objects
        self.plane = p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

        # load a panda robot
        self.panda = Panda()

    def reset(self):
        self.panda.reset()
        return self.panda.state

    def close(self):
        p.disconnect()

    def step(self, action):
        # get current state
        state = self.panda.state

        # action in this example is the end-effector velocity
        self.panda.step(dposition=action)

        # take simulation step
        p.stepSimulation()

        # return next_state, reward, done, info
        next_state = self.panda.state
        reward = 0.0
        done = False
        info = {}
        return next_state, reward, done, info

    def render(self, mode='None'):
        (width, height, pxl, depth, segmentation) = p.getCameraImage(width=self.camera_width,
                                                                     height=self.camera_height,
                                                                     viewMatrix=self.view_matrix,
                                                                     projectionMatrix=self.proj_matrix)
        rgb_array = np.array(pxl, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=30, cameraPitch=-60,
                                     cameraTargetPosition=[0.5, -0.2, 0.0])
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                               distance=1.0,
                                                               yaw=90,
                                                               pitch=-50,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.camera_width) / self.camera_height,
                                                        nearVal=0.1,
                                                        farVal=100.0)


class PandaMoveBoxEnv(PandaRawEnv):

    def __init__(self):
        super(PandaMoveBoxEnv, self).__init__()
        p.getConnectionInfo()
        p.setPhysicsEngineParameter(enableFileCaching=0)

        # set location of the object
        self.obj_location = np.asarray([0.6, 0., 0.1])

        # object is a long box with a square bottom
        self.obj = YCBObject('zsy_long_box')
        self.obj.load()
        p.resetBasePositionAndOrientation(self.obj.body_id, self.obj_location, [0, 0, 0, 1])
        self.obj_height = 0.24
        self.obj_width = 0.06

        # set the target location
        self.target = YCBObject('zsy_base')
        self.target.load()
        self.target_location = np.asarray([0.3, -0.3, 0.1])
        p.resetBasePositionAndOrientation(self.target.body_id, self.target_location, [0, 0, 0, 1])
        self.target_width = 0.12
        self.target_height = 0.02
        # todo: fix the target

        # load a panda robot
        self.seed(1234)
        self.arm_id = self.panda.panda
        self.obj_id = self.obj.body_id

        # action space is the end-effector's velocity and the gripper's status (positive: open, negative: close)
        self.action_space = spaces.Box(
            low=np.array([-1., -1., -1., -1]),
            high=np.array([1., 1., 1., 1]),
            dtype=np.float64
        )

        # observation space is
        # [joint_position*9, joint_velocity*9, joint_torque*9, ee_position*3, ee_quaternion*4,
        # obj_location*3, obj_height, obj_width, target_location*3, target_height, target_width]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 44),
            high=np.array([np.inf] * 44),
            dtype=np.float64
        )

        self.step_number = 0
        self.catch = False
        self.move_to_target = False
        self.overturn_goal = False

        # connect to keyboard
        self.key = Key(scale=0.02)

    def reset(self):
        self.step_number = 0
        self.move_to_target = False
        self.overturn_goal = False
        p.resetBasePositionAndOrientation(self.obj_id, self.obj_location, [0, 0, 0, 1])
        self.panda.reset()
        return_state = np.concatenate(
            [self.panda.state['joint_position'],
             self.panda.state['joint_velocity'],
             self.panda.state['joint_torque'],
             self.panda.state['ee_position'],
             self.panda.state['ee_quaternion'],
             self.obj.get_position(),
             np.array([self.obj_height]),
             np.array([self.obj_width]),
             self.target.get_position(),
             np.array([self.target_height]),
             np.array([self.target_width])]
        )
        return return_state

    def reset_with_obs(self, obs):
        self.move_to_target = False
        self.overturn_goal = False
        p.resetBasePositionAndOrientation(self.obj_id, self.obj_location, [0, 0, 0, 1])
        self.panda.reset_with_obs(obs)
        return_state = np.concatenate(
            [self.panda.state['joint_position'],
             self.panda.state['joint_velocity'],
             self.panda.state['joint_torque'],
             self.panda.state['ee_position'],
             self.panda.state['ee_quaternion'],
             self.obj.get_position(),
             np.array([self.obj_height]),
             np.array([self.obj_width]),
             self.target.get_position(),
             np.array([self.target_height]),
             np.array([self.target_width])]
        )
        return return_state

    def seed(self, seed=None):
        self.panda.seed(seed)
        return [seed]

    def calculate_reward(self, state, action):
        reward = 0
        done = False
        obj_position = self.obj.get_position()

        # the target cannot be moved
        if np.linalg.norm(self.target.get_position()[0:1] - self.target_location[0:1]) > 0.01:
            reward -= 2000
            done = True

        # punish the distance between the end-effector and the object
        dist = np.linalg.norm(state['ee_position'] - obj_position)
        if dist > self.obj_height / 2 + 0.02:
            reward -= (dist - (self.obj_height / 2 + 0.02))

        # punish the energy cost
        reward -= np.linalg.norm(action)

        # judge if the object is caught
        if obj_position[2] > self.obj_height / 2 + 0.01 and not self.catch:
            self.catch = True
            reward += 500

        # judge if the object is overturned
        if obj_position[2] < self.obj_height / 2 - 0.05 and not self.overturn_goal:
            self.overturn_goal = True
            reward -= 500
            done = True

        # judge if the object has been moved to the target
        if abs(obj_position[0] - self.target_location[0]) < (self.target_width - self.obj_width) / 2 \
                and abs(obj_position[1] - self.target_location[1]) < (self.target_width - self.obj_width) / 2 \
                and obj_position[2] < self.target_height + self.obj_height / 2 + 0.01 and not self.move_to_target:
            self.move_to_target = True
            reward += 2000
            done = True

        return reward, done

    def step(self, action):
        # get current state
        state = self.panda.state
        self.step_number += 1

        # action in this example is the end-effector velocity and grasp
        if action[3] >= 0:
            grasp = 0
        else:
            grasp = 1
        self.panda.step(dposition=action[0:3], grasp_open=not grasp)

        # take simulation step
        p.stepSimulation()

        # return next_state, reward, done, info
        next_state = self.panda.state
        info = next_state

        return_state = np.concatenate(
            [self.panda.state['joint_position'],
             self.panda.state['joint_velocity'],
             self.panda.state['joint_torque'],
             self.panda.state['ee_position'],
             self.panda.state['ee_quaternion'],
             self.obj.get_position(),
             np.array([self.obj_height]),
             np.array([self.obj_width]),
             self.target.get_position(),
             np.array([self.target_height]),
             np.array([self.target_width])]
        )

        reward, done = self.calculate_reward(next_state, action)
        return return_state, reward, done, info

    def teleop_step(self):
        """
        use keyboard to control the robot
        :return: state, action, reward, next_state, done, info
        """
        time.sleep(0.0001)
        # get current state
        state = self.panda.state
        self.step_number += 1

        return_state = np.concatenate(
            [self.panda.state['joint_position'],
             self.panda.state['joint_velocity'],
             self.panda.state['joint_torque'],
             self.panda.state['ee_position'],
             self.panda.state['ee_quaternion'],
             self.obj.get_position(),
             np.array([self.obj_height]),
             np.array([self.obj_width]),
             self.target.get_position(),
             np.array([self.target_height]),
             np.array([self.target_width])]
        )

        # read in from keyboard
        key_input = self.key.get_controller_state()
        dpos, dquat, grasp, reset = (
            key_input["dpos"],
            key_input["dquat"],
            key_input["grasp"],
            key_input["reset"],
        )
        action = np.zeros(4)
        action[0:3] = dpos

        # action in this example is the end-effector velocity
        self.panda.step(dposition=dpos, dquaternion=dquat, grasp_open=not grasp)

        # take simulation step
        p.stepSimulation()

        # return next_state, reward, done, info
        next_state = self.panda.state
        reward, done = self.calculate_reward(next_state, action[0:3])

        return_next_state = np.concatenate(
            [self.panda.state['joint_position'],
             self.panda.state['joint_velocity'],
             self.panda.state['joint_torque'],
             self.panda.state['ee_position'],
             self.panda.state['ee_quaternion'],
             self.obj.get_position(),
             np.array([self.obj_height]),
             np.array([self.obj_width]),
             self.target.get_position(),
             np.array([self.target_height]),
             np.array([self.target_width])]
        )

        if grasp:
            action[3] = -1
        else:
            action[3] = 1
        print(f'step: {self.step_number}\treward: {reward}\tdone: {done}')
        if reset:
            done = True
        info = self.panda.state
        return return_state, action, reward, return_next_state, done, info

    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=20, cameraPitch=-30,
                                     cameraTargetPosition=[0.5, -0.2, 0.2])
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                               distance=1.0,
                                                               yaw=90,
                                                               pitch=-50,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.camera_width) / self.camera_height,
                                                        nearVal=0.1,
                                                        farVal=100.0)
