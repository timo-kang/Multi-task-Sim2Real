from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
import cv2
from sawyer_control.core.image_env import ImageEnv
from datetime import datetime
import csv
import numpy as np
import os

class Generator (object):


    def __init__(self, ep_num=0):
        self.env = ImageEnv(
            SawyerReachXYZEnv(
            action_mode='position',
            position_action_scale = 0.1
        ))
        # sawyer z : 0.93
        self.env.reset()
        self.ep_num = ep_num
        self.cnt = 0
        self.img_name = ''
        self.status = STATUS_LIST[0]
        self.cube_pos = np.zeros(3)
        self.goal_pos = np.zeros(3)
        self.velocities = np.zeros(3)
        self.destination = np.zeros(3)
        self.joint_angles = np.zeros(7)
        self.ee_pose = np.zeros(3)
        self.gripper_action = [0,0,1]

    def __str__(self):
        return 'Generator ep.' + str(self.ep_num) + ' with ' +  self.img_name

    def __set__(self, instance, value):
        # instance(self.cnt + 1)
        pass

    def __get__(self, instance, value):
        pass

    def __len__(self):
        return self.cnt

    def init_label(self):
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        timestamp = (timestamp)
        self.img_name = "ep_" + str(self.ep_num) + "_" + str(timestamp) + ".png"
        print('Generator init : ', self.img_name)

    def set_next_episode(self):
        self.reset()
        self.env.set_state('bowl')
        self.env.set_state('wood_cube_2_5cm')
        ## TODO :: put cube and bowl randomly distributed.
        ## TODO :: set sawyer position randomly distributed.
        ##
        self.ep_num += 1
        self.cnt = 0

    def reset(self):
        self.env.set_to_goal(np.zeros(3))
        self.img_name = ''
        self.status = STATUS_LIST[0]
        self.cube_pos = np.zeros(3)
        self.velocities = np.zeros(3)
        self.destination = np.zeros(3)
        self.goal_pos = np.zeros(3)
        self.joint_angles = np.zeros(7)
        self.ee_pose = np.zeros(3)
        self.gripper_action = [0,0,1]

    def get_env_state(self):
        try:
            # gripper action : [open, close, no-op]
            ## TODO :: get cube, place position from env.
            c = self.env.get_state('wood_cube_2_5cm')
            angles, velocities, endpoint_pose = self.env.request_observation()

            self.cube_pos = c
            self.joint_angles = angles
            self.velocities = velocities
            self.ee_pose = endpoint_pose[:3]
        except Exception as e:
            print(e)

    def save(self):
        img = self.env.get_image()
        self.init_label()
        cv2.imwrite("./data/train/images/"+ self.img_name , img)
        print('successfully saved image: ', self.img_name)
        path = os.path.join("data","train","physics", "sawyer_sim_ep_" + str(self.ep_num) + ".csv")
        with open(path, "a", newline='') as f:
            csvWriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            data = [self.img_name] + self.velocities.tolist() + self.gripper_action \
                    + self.cube_pos.tolist() + self.ee_pose.tolist() + self.joint_angles.tolist()
            # print(data)
            csvWriter.writerow(data)
        f.close()
        print('ep.', self.ep_num,'stage:',self.status, 'scene no.', self.cnt, 'saved.')
        self.cnt += 1

    def run_episode(self, callback):
        if self.status != STATUS_LIST[0]:
            return
        self.get_env_state()
        self.destination = self.env.get_state('bowl')
        self.goal_pos = self.cube_pos
        self.status = STATUS_LIST[1]

        return callback()

    def stage_callback(self):
        if self.status == STATUS_LIST[0]:
            # init stage
            print('INVALID_CALLBACK_STATUS', STATUS_LIST[0])
            return
        elif self.status == STATUS_LIST[1]:
            # reaching cube stage
            self.next_stage(1, self.stage_callback)
        elif self.status == STATUS_LIST[2]:
            # pick stage
            self._gripper_act()
        elif self.status == STATUS_LIST[3]:
            # reaching goal stage
            self.next_stage(3, self.stage_callback)
        elif self.status == STATUS_LIST[4]:
            # place stage
            self._gripper_act()

    def _gripper_act(self):
        if self.status == STATUS_LIST[2]:
            self.gripper_action = [0,1,0]
            self.env._act('close')
            for _ in range(10):
                self.get_env_state()
                self.save()
            self.status = STATUS_LIST[3]
            self.goal_pos = self.destination
            self.gripper_action = [0,0,1]
            return self.stage_callback()
        elif self.status == STATUS_LIST[4]:
            self.gripper_action = [1,0,0]
            self.env._act('open')
            for _ in range(10):
                self.get_env_state()
                self.save()
            self.status = STATUS_LIST[5]
            self.gripper_action = [0,0,1]
            return
        else:
            print('INVALID_ACTION_STATUS')
            return

    def next_stage(self, status, callback):
        if self.status != STATUS_LIST[status] :
            return False
        for _ in range(100):
            self.env._act(self.goal_pos - self.ee_pose)
            self.get_env_state()
            self.save()
            if self._check_stop_complete() and self.stage_clear_check():
                break

        self.status = STATUS_LIST[status+1]
        return callback()

    def _check_stop_complete(self):
        close_to_desired_reset_pos = self._check_stop_angles_within_threshold()
        velocities = self.velocities
        velocities = np.abs(np.array(velocities))
        VELOCITY_THRESHOLD = .002 * np.ones(7)
        no_velocity = (velocities < VELOCITY_THRESHOLD).all()
        return close_to_desired_reset_pos and no_velocity

    def _check_stop_angles_within_threshold(self):
        desired_neutral = self.env.AnglePDController._des_angles
        desired_neutral = np.array([desired_neutral[joint] for joint in self.env.config.JOINT_NAMES])
        actual_neutral = (self.joint_angles)
        errors = self.compute_angle_difference(desired_neutral, actual_neutral)
        is_within_threshold = (errors < self.env.config.RESET_ERROR_THRESHOLD).all()
        return is_within_threshold

    def compute_angle_difference(self, angles1, angles2):
        deltas = np.abs(angles1 - angles2)
        differences = np.minimum(2 * np.pi - deltas, deltas)
        return differences

    def failure_check(self):
        if self.status != STATUS_LIST[-1]:
            print('NOT_FINISHED, UNEXPECTED_ERROR_OCCURED')
            return True
        EPISODE_THRESHOLD = .01 * np.ones(3)
        offset = np.abs(self.cube_pos - self.destination)
        failed = (offset > EPISODE_THRESHOLD).all()
        return failed
    def stage_clear_check(self):
        STAGE_CLEAR_THRESHOLD = .02 * np.ones(3)
        offset = np.abs(self.ee_pose - self.goal_pos)
        failed = (offset < STAGE_CLEAR_THRESHOLD).all()
        return failed


if __name__ == '__main__':

    STATUS_LIST = [
        'init',
        'reaching_cube',
        'pick',
        'reaching_place',
        'place',
        'finish'
    ]


    g = Generator(0)
    g.get_env_state()
    g.reset()
    # g.env.set_to_goal(np.zeros(3))
    # g.run_episode(g.stage_callback)
    # if g.status == 'finish':
    #     print('first episode finished!!')
    #     if g.failure_check():
    #         print('EP.',g.ep_num, ' is failed!')
    #     else:
    #         g.set_next_episode()
