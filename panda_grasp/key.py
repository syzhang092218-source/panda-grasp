import pybullet as p
import numpy as np


class Key:

    def __init__(self, scale=0.01):
        self._reset_internal_state()
        self._scale_trans = scale
        self.grasp = 0

    def _reset_internal_state(self):
        # reset mode: currently only support mode 1
        self.mode = 1

        # reset state
        self.reset_state = False

        # open the gripper
        self.grasp = 0

        # reset the velocity to zero
        self.v = np.zeros(3)
        self.x_forward = 0
        self.x_back = 0
        self.y_forward = 0
        self.y_back = 0
        self.z_forward = 0
        self.z_back = 0

    def _get_inputs(self):
        events = p.getKeyboardEvents()
        for key_pressed in events.keys():
            # reset the state
            if key_pressed == ord('r'):
                self.reset_state = True

            # control the gripper: use U and I
            if key_pressed == ord('o'):  # O: open the gripper
                self.grasp = 0
            if key_pressed == ord('i'):  # I: close the gripper
                self.grasp = 1

            # control the position of the end effector:
            # use the arrows to control on the x-y plane, and J/K to control the z axis
            if key_pressed == 65297:  # up arrow
                self.y_forward = 1
            if key_pressed == 65298:  # down arrow
                self.y_back = 1
            if key_pressed == 65295:  # left arrow
                self.x_back = 1
            if key_pressed == 65296:  # right arrow
                self.x_forward = 1
            if key_pressed == ord('j'):  # J
                self.z_forward = 1
            if key_pressed == ord('k'):  # K
                self.z_back = 1

        # calculate the velocity
        self.v = np.asarray([self.x_forward - self.x_back,
                             self.y_forward - self.y_back,
                             self.z_forward - self.z_back])

    def _reset_input(self):
        # reset state
        self.reset_state = False

        # reset the velocity to zero
        self.v = np.zeros(3)
        self.x_forward = 0
        self.x_back = 0
        self.y_forward = 0
        self.y_back = 0
        self.z_forward = 0
        self.z_back = 0

    def _get_mode(self):
        # currently only support mode 1: controlling end-effector position
        self.mode = 1

    def get_controller_state(self):
        self._get_inputs()
        self._get_mode()
        dpos = np.array([0.0, 0.0, 0.0])
        dquat = np.array([0.0, 0.0, 0.0, 0.0])
        if self.mode == 1:
            dpos = self.v * self._scale_trans
        else:
            print('PANDA ERROR: using unsupported mode')
        self._reset_input()
        return dict(
            dpos=dpos,
            dquat=dquat,
            grasp=self.grasp,
            reset=self.reset_state,
        )
